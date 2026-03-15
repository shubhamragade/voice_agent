import asyncio
import logging
import re
import json
from openai import AsyncOpenAI
import os

from .session import Session
from .rag import retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a professional Voice AI Assistant. 
Your primary goal is to provide accurate information based on the PROVIDED CONTEXT.
Keep responses conversational, friendly, and concise (1-3 sentences).
If you find relevant info in the context, use it! If not, answer generally but mention you don't have that specific data in your files."""

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
            # Use context-aware retrieval: Combine current message with previous user turn 
            # if current query is short or contains pronouns, to maintain entity context.
            rag_query = user_message
            if len(user_message) < 50 and len(session.history) > 2:
                prev_user_msg = session.history[-2]["content"]
                rag_query = f"{prev_user_msg} {user_message}"
                logger.info(f"[{session.session_id}] Using contextual RAG query: {rag_query}")

            context_chunks = retrieve(rag_query, k=5)
            
            # Refine System Prompt to handle STT hallucinations/misrecognitions
            corrected_instruction = """
            IMPORTANT: You are receiving transcripts from a live STT engine. 
            If a word sounds like it might be a mistake (e.g., 'reward' instead of 'river', 'nile' instead of 'nil'), 
            use the context of the conversation and provide the most logical answer.
            If the user is asking about the longest river, they likely mean the Nile.
            """
            
            current_system_prompt = SYSTEM_PROMPT + corrected_instruction
            if context_chunks:
                logger.info(f"[{session.session_id}] Injecting {len(context_chunks)} RAG chunks into prompt")
                context_str = "\n".join([f"- {c}" for c in context_chunks])
                current_system_prompt += f"\n\nCRITICAL KNOWLEDGE BASE DATA:\n{context_str}\n\nINSTRUCTION: Priority #1 is answering using the data above. If the STT transcript looks like a misspelling of a company or name in this context, correct it and answer."
            else:
                logger.warning(f"[{session.session_id}] No RAG context found for this turn.")
            
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
