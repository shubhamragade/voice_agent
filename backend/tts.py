import asyncio
import logging
import os
from openai import AsyncOpenAI

from .session import Session

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

async def tts_worker(session: Session):
    """
    Reads finalized sentences from tts_queue, generates audio via OpenAI TTS,
    and yields MP3 chunks to the audio_out_queue.
    """
    logger.info(f"[{session.session_id}] Starting TTS worker")
    
    try:
        while session.active:
            try:
                # Wait for the next sentence
                sentence = await asyncio.wait_for(session.tts_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if session.interrupted:
                continue

            logger.info(f"[{session.session_id}] Synthesizing TTS: {sentence[:30]}...")

            try:
                # Collect all chunks into a single complete MP3 file in memory for this sentence
                sentence_mp3 = b""
                async with client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="nova",
                    response_format="mp3",
                    input=sentence
                ) as response:
                    
                    async for chunk in response.iter_bytes(chunk_size=4096):
                        if session.interrupted:
                            logger.info(f"[{session.session_id}] TTS generation interrupted mid-stream")
                            session.flush_audio_out()
                            break
                        
                        if chunk:
                            sentence_mp3 += chunk
                
                # Send the completely formed MP3 for this sentence to the frontend
                # decodeAudioData requires complete standard files to prevent skipping frames
                if sentence_mp3 and not session.interrupted:
                    await session.audio_out_queue.put(sentence_mp3)

            except Exception as e:
                logger.error(f"[{session.session_id}] OpenAI TTS API error: {e}", exc_info=True)
                
    except asyncio.CancelledError:
        logger.info(f"[{session.session_id}] tts_worker task cancelled")
    except Exception as e:
        logger.error(f"[{session.session_id}] tts_worker error: {e}", exc_info=True)
    finally:
        logger.info(f"[{session.session_id}] tts_worker exiting")
