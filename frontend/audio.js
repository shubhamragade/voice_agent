import { VAD } from './vad.js';

const WS_URL = 'ws://localhost:8000/ws/audio';

let ws;
let audioContext;
let scriptProcessor;
let mediaStream;
let mediaStreamSource;

// Robust state management
let isRecording = false;
let isPlaying = false;
let isStopping = false;

let playbackQueue = [];
let currentSource = null;

const vad = new VAD(0.04, 3);
const btn = document.getElementById('session-button');

const TARGET_SAMPLE_RATE = 16000;
const BUFFER_SIZE = 4096;

let hasInterruptedDuringCurrentSpeech = false;
let recordStartTime = 0;

function connectWebSocket() {
    console.log('Attemping to connect WebSocket...');
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        console.log('WebSocket Connected');
        window.updateStatus(true);
        window.logMessage('system', 'Connected to backend. Session ready.');
        // Notify any pending startRecording call that we are ready
        if (window._wsResolve) window._wsResolve();
    };

    ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
            try {
                const audioBuffer = await audioContext.decodeAudioData(event.data);
                playbackQueue.push(audioBuffer);
                playNextInQueue();
            } catch (err) {
                console.error('Error decoding audio chunk:', err);
            }
        } else if (typeof event.data === 'string') {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'user_transcript') {
                    window.logMessage('user', data.text);
                } else if (data.type === 'ai_response') {
                    window.logMessage('ai', data.text);
                }
            } catch (err) {
                console.error('Error parsing JSON from server:', err);
            }
        }
    };

    ws.onclose = () => {
        console.warn('WebSocket Closed.');
        window.updateStatus(false);
    };

    ws.onerror = (err) => console.error('WS Connection Error:', err);
}

function playNextInQueue() {
    if (isPlaying || playbackQueue.length === 0) return;

    isPlaying = true;
    const buffer = playbackQueue.shift();

    currentSource = audioContext.createBufferSource();
    currentSource.buffer = buffer;
    currentSource.connect(audioContext.destination);

    currentSource.onended = () => {
        isPlaying = false;
        currentSource = null;
        if (playbackQueue.length > 0) {
            playNextInQueue();
        } else {
            hasInterruptedDuringCurrentSpeech = false;
        }
    };

    currentSource.start(0);
}

function interruptPlayback() {
    if (!isPlaying) return;
    
    console.log('Interruption triggered!');
    if (currentSource) {
        currentSource.onended = null;
        currentSource.stop();
        currentSource = null;
    }
    playbackQueue = [];
    isPlaying = false;
    hasInterruptedDuringCurrentSpeech = true;

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'interrupt' }));
        window.logMessage('system', 'You interrupted.');
    }
}

async function downsampleAndConvertToInt16(inputData, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) {
        return floatTo16BitPCM(inputData);
    }

    const duration = inputData.length / inputSampleRate;
    const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, outputSampleRate * duration, outputSampleRate);
    
    const audioBuffer = offlineCtx.createBuffer(1, inputData.length, inputSampleRate);
    audioBuffer.copyToChannel(inputData, 0);

    const source = offlineCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineCtx.destination);
    source.start(0);

    const renderedBuffer = await offlineCtx.startRendering();
    const resampledData = renderedBuffer.getChannelData(0);

    return floatTo16BitPCM(resampledData);
}

function floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output;
}

async function startRecording() {
    if (isRecording || isStopping) return;
    
    console.log('Starting session...');
    
    // 1. Establish WebSocket FIRST
    try {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            connectWebSocket();
            // Wait for open
            await new Promise((resolve, reject) => {
                window._wsResolve = resolve;
                setTimeout(() => reject(new Error('Connection timeout')), 5000);
            });
        }
    } catch (err) {
        console.error('WebSocket connection failed:', err);
        window.logMessage('system', 'Connection failed. Please try again.');
        return;
    }

    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    try {
        isRecording = true;
        btn.classList.add('recording');
        btn.innerText = 'Stop Conversation';
        hasInterruptedDuringCurrentSpeech = false;
        recordStartTime = Date.now();

        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);

        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0;

        scriptProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(gainNode);
        gainNode.connect(audioContext.destination);

        window._scriptProcessor = scriptProcessor;

        scriptProcessor.onaudioprocess = async (e) => {
            if (!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);

            const isSpeakingNow = vad.processFrame(inputData);
            const timeSinceStart = Date.now() - recordStartTime;
            
            if (isSpeakingNow && isPlaying && !hasInterruptedDuringCurrentSpeech && timeSinceStart > 500) {
                interruptPlayback();
            }

            if (ws && ws.readyState === WebSocket.OPEN) {
                const int16Data = await downsampleAndConvertToInt16(
                    inputData,
                    audioContext.sampleRate,
                    TARGET_SAMPLE_RATE
                );
                ws.send(int16Data.buffer);
            }
        };

        window.logMessage('system', 'Microphone active. You can speak now.');

    } catch (err) {
        console.error('Failed to access microphone', err);
        isRecording = false;
        btn.classList.remove('recording');
        btn.innerText = 'Start Conversation';
        window.logMessage('system', 'Microphone access denied.');
    }
}

async function stopRecording() {
    if (!isRecording || isStopping) return;
    isStopping = true;
    
    console.log('Stopping session...');
    try {
        if (scriptProcessor) {
            scriptProcessor.disconnect();
            scriptProcessor.onaudioprocess = null;
            scriptProcessor = null;
        }
        if (mediaStreamSource) {
            mediaStreamSource.disconnect();
            mediaStreamSource = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        isRecording = false;
        btn.classList.remove('recording');
        btn.innerText = 'Start Conversation';
        
        // Close WebSocket to end backend session cleanly
        if (ws) {
            ws.close();
            ws = null;
        }
        
        window.logMessage('system', 'Session stopped.');
    } finally {
        isStopping = false;
    }
}

function toggleRecording(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

btn.addEventListener('click', toggleRecording);
