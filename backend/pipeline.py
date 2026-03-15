import asyncio
import logging
from fastapi import WebSocket

from .session import Session
from .stt import stt_worker
from .llm import llm_worker
from .tts import tts_worker

logger = logging.getLogger(__name__)

async def sender(session: Session, websocket: WebSocket):
    """
    Reads finalized TTS audio chunks and text updates from queues 
    and sends them to the client.
    """
    logger.info(f"[{session.session_id}] Starting sender worker")
    
    while session.active:
        audio_task = asyncio.create_task(session.audio_out_queue.get())
        text_task = asyncio.create_task(session.text_out_queue.get())
        
        done, pending = await asyncio.wait(
            [audio_task, text_task], 
            return_when=asyncio.FIRST_COMPLETED,
            timeout=1.0
        )
        
        for task in pending:
            task.cancel()
            
        for task in done:
            try:
                result = task.result()
                if task == audio_task:
                    # Send binary audio
                    await websocket.send_bytes(result)
                else:
                    # Send JSON text message
                    await websocket.send_text(result)
            except Exception as e:
                logger.error(f"[{session.session_id}] Error in sender loop: {e}")

    logger.info(f"[{session.session_id}] sender exiting")


async def run_pipeline(session: Session, websocket: WebSocket):
    """
    Main orchestrator for the real-time AI pipeline.
    Spawns all worker coroutines to run concurrently and handles cleanup on exit.
    """
    logger.info(f"[{session.session_id}] Starting pipeline coroutines")
    
    try:
        # We run the coroutines concurrently.
        # If any of them crash or complete (e.g., session.active becomes false), the others continue
        # until the pipeline exits.
        await asyncio.gather(
            stt_worker(session),
            llm_worker(session),
            tts_worker(session),
            sender(session, websocket),
        )
    except asyncio.CancelledError:
        logger.info(f"[{session.session_id}] Pipeline execution cancelled")
    except Exception as e:
        logger.error(f"[{session.session_id}] Pipeline exception: {e}", exc_info=True)
    finally:
        session.active = False
        logger.info(f"[{session.session_id}] Pipeline cleanup finished")
