/**
 * Simple Energy-based Voice Activity Detection
 * Calculates the Root Mean Square (RMS) of audio frames
 */
export class VAD {
    constructor(threshold = 0.04, minFrames = 5) {
        this.threshold = threshold;
        this.minFrames = minFrames;
        this.consecutiveSpeechFrames = 0;
        this.speaking = false;
        
        // Settings for silencing
        this.silenceFrames = 0;
        // ~500ms of silence at 100ms/frame => 5 frames
        this.silenceThresholdFrames = 5; 
    }

    processFrame(pcmFloatArray) {
        if (!pcmFloatArray || pcmFloatArray.length === 0) return this.speaking;

        let sumSquares = 0.0;
        for (let i = 0; i < pcmFloatArray.length; i++) {
            sumSquares += pcmFloatArray[i] * pcmFloatArray[i];
        }
        
        const rms = Math.sqrt(sumSquares / pcmFloatArray.length);

        if (rms > this.threshold) {
            this.consecutiveSpeechFrames++;
            this.silenceFrames = 0;
            
            if (this.consecutiveSpeechFrames >= this.minFrames) {
                this.speaking = true;
            }
        } else {
            this.consecutiveSpeechFrames = 0;
            this.silenceFrames++;
            
            if (this.silenceFrames >= this.silenceThresholdFrames) {
                this.speaking = false;
            }
        }

        return this.speaking;
    }
}
