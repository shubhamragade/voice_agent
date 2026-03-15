import asyncio
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.audio_in_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()
        self.text_out_queue = asyncio.Queue()
        self.interrupted = False
        self.active = True
        self.history = []
        
    def flush_audio_out(self):
        """Clears the audio output queue."""
        while not self.audio_out_queue.empty():
            try:
                self.audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"[{self.session_id}] Flushed audio_out_queue")

    def flush_tts_queue(self):
        """Clears the text-to-speech queue."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"[{self.session_id}] Flushed tts_queue")

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(self, session_id: str) -> Session:
        session = Session(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].active = False
            del self._sessions[session_id]

session_manager = SessionManager()
