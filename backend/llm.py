import asyncio
import logging
import re
import json
from openai import AsyncOpenAI
import os

from .session import Session
from .rag import retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful voice assistant. Keep responses conversational and concise. Respond in 1-3 sentences when possible."""

from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Determine sentence boundary: ends with . ! or ? optionally followed by quotes
SENTENCE_BOUNDARY = re.compile(r'([.!?]["\']?)\s')

# We'll store history manually in the session, but since we didn't add it in Session class initially, 
# we can just inject it here dynamically or augment the session object.

async def llm_worker(session: Session):
    """
    Reads transcribed user sentences from text_queue, enriches with RAG context, 
    calls GPT-4o streaming API, chunks the streaming text into sentences, 
    and adds complete sentences to audio_out_queue for TTS.
    """
    logger.info(f"[{session.session_id}] Starting LLM worker")
    
    # Initialize session history if it's empty
    if not session.history:
        session.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while session.active:
            try:
                user_message = await asyncio.wait_for(session.text_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if user_message:
                session.interrupted = False
                logger.info(f"[{session.session_id}] User asks: {user_message}")
            else:
                continue

            # RAG Integration
            # Use a slightly more descriptive query for RAG based on the full grouped transcript
            context_chunks = retrieve(user_message, k=5)
            
            # Refine System Prompt to handle STT hallucinations/misrecognitions
            corrected_instruction = """
            IMPORTANT: You are receiving transcripts from a live STT engine. 
            If a word sounds like it might be a mistake (e.g., 'reward' instead of 'river', 'nile' instead of 'nil'), 
            use the context of the conversation and provide the most logical answer.
            If the user is asking about the longest river, they likely mean the Nile.
            """
            
            current_system_prompt = SYSTEM_PROMPT + corrected_instruction
            if context_chunks:
                context_str = "\n".join([f"- {c}" for c in context_chunks])
                current_system_prompt += f"\n\nRELEVANT DATA FROM KNOWLEDGE BASE:\n{context_str}\n\nUse this data to answer accurately. If the data is irrelevant, ignore it."
            
            # Update system prompt dynamically
            session.history[0] = {"role": "system", "content": current_system_prompt}

            # Add user message
            session.history.append({"role": "user", "content": user_message})

            # Maintain a slightly longer history for better context (15 turns)
            if len(session.history) > 15:
                session.history = [session.history[0]] + session.history[-14:]

            assistant_reply_full = ""
            buffer = ""
            
            try:
                # Notify UI that AI is thinking/responding
                await session.text_out_queue.put(json.dumps({
                    "type": "ai_status", 
                    "text": "..."
                }))

                # Stream from OpenAI
                stream = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=session.history,
                    stream=True,
                )
                
                async for chunk in stream:
                    if session.interrupted:
                        logger.info(f"[{session.session_id}] LLM generation interrupted by user")
                        break

                    content = chunk.choices[0].delta.content
                    if content is not None:
                        buffer += content
                        assistant_reply_full += content

                        # Check for sentence boundaries
                        # We consider it a sentence if it matches boundary and is >= 20 chars
                        match = SENTENCE_BOUNDARY.search(buffer)
                        if match and len(buffer) >= 20:
                            # Split at the boundary
                            end_pos = match.end()
                            sentence = buffer[:end_pos].strip()
                            buffer = buffer[end_pos:] # keep the remainder
                            
                            if sentence:
                                logger.debug(f"[{session.session_id}] LLM emitted sentence: {sentence}")
                                await session.tts_queue.put(sentence)
                
                # Flush the final remainder of the buffer if any
                if buffer.strip() and not session.interrupted:
                    logger.debug(f"[{session.session_id}] LLM emitted sentence: {buffer.strip()}")
                    await session.tts_queue.put(buffer.strip())

                if assistant_reply_full and not session.interrupted:
                    session.history.append({"role": "assistant", "content": assistant_reply_full})
                    # Sync full response with UI
                    await session.text_out_queue.put(json.dumps({
                        "type": "ai_response", 
                        "text": assistant_reply_full
                    }))

            except Exception as e:
                logger.error(f"[{session.session_id}] OpenAI API error: {e}", exc_info=True)

    except asyncio.CancelledError:
        logger.info(f"[{session.session_id}] llm_worker task cancelled")
    except Exception as e:
        logger.error(f"[{session.session_id}] llm_worker error: {e}", exc_info=True)
    finally:
        logger.info(f"[{session.session_id}] llm_worker exiting")
