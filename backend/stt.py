import asyncio
import logging
import os
import json
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
from .session import Session

logger = logging.getLogger(__name__)

async def stt_worker(session: Session):
    """Reads audio bytes from the session queue and streams to Deepgram STT."""
    logger.info(f"[{session.session_id}] Starting STT worker")
    
    try:
        # Initialize Deepgram client
        api_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not api_key:
            logger.error("DEEPGRAM_API_KEY is missing")
            return
            
        deepgram = DeepgramClient(api_key)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        # Grouping logic variables
        accumulated_transcript = []
        last_transcript_time = 0
        grouping_timeout = 1.0 # Reduced for better responsiveness

        # Define event handlers
        async def on_message(self, result, **kwargs):
            nonlocal last_transcript_time
            if session.interrupted:
                return

            if not result.channel.alternatives:
                return
                
            sentence = result.channel.alternatives[0].transcript
            if not sentence.strip():
                return

            if result.is_final:
                logger.debug(f"[{session.session_id}] Component Transcript: {sentence}")
                accumulated_transcript.append(sentence)
                last_transcript_time = asyncio.get_event_loop().time()

        async def on_error(self, error, **kwargs):
            logger.error(f"[{session.session_id}] Deepgram STT error: {error}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # ... (options and start logic remains same)
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            punctuate=True,
            # Improve end-of-speech detection
            endpointing=300 
        )

        if not await dg_connection.start(options):
            logger.error(f"[{session.session_id}] Failed to connect to Deepgram")
            return

        # Main reading loop
        while session.active:
            try:
                # Check if we should flush the accumulated transcript
                current_time = asyncio.get_event_loop().time()
                if accumulated_transcript and (current_time - last_transcript_time) > grouping_timeout:
                    full_query = " ".join(accumulated_transcript).strip()
                    if full_query:
                        logger.info(f"[{session.session_id}] STT Final Grouped Transcript: {full_query}")
                        await session.text_queue.put(full_query)
                        # Sync with frontend UI
                        await session.text_out_queue.put(json.dumps({
                            "type": "user_transcript", 
                            "text": full_query
                        }))
                    accumulated_transcript = []

                # Wait for audio from the frontend
                # Using a shorter timeout to allow checking the grouping timer frequently
                try:
                    audio_data = await asyncio.wait_for(session.audio_in_queue.get(), timeout=0.1)
                    await dg_connection.send(audio_data)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    logger.info(f"[{session.session_id}] Deepgram connection issue. Reconnecting...")
                    # Reconnection logic...
                    try:
                        await dg_connection.finish()
                    except Exception:
                        pass
                    dg_connection = deepgram.listen.asyncwebsocket.v("1")
                    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
                    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
                    if await dg_connection.start(options):
                        # skip sending the failed chunk for now or retry
                        pass
                        
            except Exception as e:
                logger.error(f"[{session.session_id}] Loop error: {e}")
                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logger.info(f"[{session.session_id}] stt_worker task cancelled")
    except Exception as e:
        logger.error(f"[{session.session_id}] stt_worker encountered an error: {e}", exc_info=True)
    finally:
        try:
            if 'dg_connection' in locals():
                await dg_connection.finish()
        except Exception:
            pass
        logger.info(f"[{session.session_id}] stt_worker exiting")
