import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import List

# Use late imports or simply initialize components here
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Global in-memory vector store
vector_store: FAISS = None
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def load_file(filepath: str) -> List:
    """Helper to load a file based on extension."""
    docs = []
    try:
        if filepath.lower().endswith(".txt"):
            loader = TextLoader(filepath)
            docs.extend(loader.load())
        elif filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        elif filepath.lower().endswith((".doc", ".docx")):
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
        else:
            logger.warning(f"Unsupported file type: {filepath}")
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
    return docs

def init_rag():
    global vector_store
    logger.info("Initializing RAG vector store")
    
    docs = []
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
    
    # Initialize an empty vector store to avoid NoneType errors
    vector_store = FAISS.from_texts([""], embeddings)
    
    if not os.path.exists(base_dir):
        logger.warning(f"Knowledge base directory not found at {base_dir}")
        return

    # Load all txt, pdf, docx files
    for filepath in os.listdir(base_dir):
        full_path = os.path.join(base_dir, filepath)
        docs.extend(load_file(full_path))

    if not docs:
        logger.info("No documents found in knowledge base.")
        return

    try:
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(splits)
        logger.info(f"Loaded {len(splits)} document chunks into FAISS.")
    except Exception as e:
        logger.error(f"Failed to initialize FAISS vector store: {e}")

def add_document(filepath: str) -> bool:
    """Takes a file path and dynamically adds it to the RAG vector store."""
    global vector_store
    docs = load_file(filepath)
    if not docs:
        return False
        
    try:
        splits = text_splitter.split_documents(docs)
        if vector_store is None:
            vector_store = FAISS.from_documents(splits, embeddings)
        else:
            vector_store.add_documents(splits)
        logger.info(f"Dynamically added {len(splits)} chunks from {filepath} to FAISS.")
        return True
    except Exception as e:
        logger.error(f"Failed to add document to index: {e}")
        return False

def retrieve(query: str, k: int = 5) -> List[str]:
    """Retrieve top-k relevant text chunks for the query with a minimal similarity threshold."""
    if vector_store is None:
        return []
    
    try:
        # We use similarity_search_with_score to filter out low-quality matches
        # Langchain FAISS returns L2 distance (lower is better)
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        
        # Threshold: Distance < 1.5 (Very permissive to handle misspellings)
        # GPT-4o will filter irrelevant data via system prompt.
        results = []
        for doc, score in docs_with_scores:
            if score < 1.5: 
                if doc.page_content.strip():
                    results.append(doc.page_content)
        
        if results:
            logger.info(f"RAG retrieved {len(results)} chunks for query: '{query}'")
        else:
            logger.warning(f"RAG found NO relevant chunks for query: '{query}'")
        
        return results
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []

# Initialize on import
init_rag()
