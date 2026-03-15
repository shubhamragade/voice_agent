import asyncio
import json
import logging
import uuid
import os
import shutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .session import session_manager
from .pipeline import run_pipeline
from .rag import add_document

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice AI Conversation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
    os.makedirs(base_dir, exist_ok=True)
    
    file_path = os.path.join(base_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Dynamically inject into FAISS
    success = add_document(file_path)
    
    if success:
        return {"filename": file.filename, "status": "File successfully uploaded and indexed."}
    else:
        return {"filename": file.filename, "status": "File uploaded but indexing failed."}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = session_manager.create_session(session_id)
    
    logger.info(f"[{session_id}] WebSocket connected")
    
    # Start the processing pipeline
    pipeline_task = asyncio.create_task(run_pipeline(session, websocket))
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Raw PCM audio from client
                await session.audio_in_queue.put(message["bytes"])
                
            elif "text" in message:
                # Control messages (JSON)
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "interrupt":
                        logger.info(f"[{session_id}] Interruption received from client")
                        session.interrupted = True
                        session.flush_audio_out()
                        session.flush_tts_queue()
                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] Received invalid JSON: {message['text']}")
                    
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected")
    except RuntimeError as e:
        if 'Cannot call "receive" once a disconnect' in str(e):
            logger.info(f"[{session_id}] WebSocket disconnected abruptly")
        else:
            logger.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
    finally:
        session_manager.remove_session(session_id)
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            logger.info(f"[{session_id}] Pipeline task cancelled gracefully")
