# multimodal_rag/main.py

import os
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Annotated, Sequence, TypedDict

# --- LangGraph and State Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# --- Our RAG Modules ---
from data_extraction.youtube_extractor import extract_youtube_data
from data_extraction.pdf_extractor import extract_pdf_data
from embedding.embedder import Embedder
from vector_store.chroma_store import ChromaStore
from retrieval.retriever import Retriever
from generation.generator import Generator

# --- Global App Components ---
app_graph = None
embedder = Embedder()
store = ChromaStore(db_path="output/web_chroma_db")
retriever = Retriever(embedder, store.text_collection, store.image_collection)
generator = Generator()


# --- Lifespan Management for the Checkpointer ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the checkpointer's lifecycle."""
    global app_graph
    with SqliteSaver.from_conn_string(":memory:") as memory:
        # Define the Graph State
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # Define Graph Nodes
        def retrieve_node(state: AgentState):
            last_message = state['messages'][-1].content
            retrieved_results = retriever.retrieve(last_message, n_results=3)
            context_str = f"--- CONTEXT ---\nText: {retrieved_results['text']['documents']}\nImages: {retrieved_results['image']['documents']}"
            return {"messages": [("system", context_str)]}

        def generate_node(state: AgentState):
            query = state['messages'][-2].content
            context = state['messages'][-1].content
            response = generator.model.generate_content(f"Based on this context:\n{context}\n\nAnswer this query: {query}").text
            return {"messages": [("ai", response)]}

        # Build the Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        app_graph = workflow.compile(checkpointer=memory)
        print("Application startup complete. Graph is compiled with active checkpointer.")
        yield
    print("Application shutdown complete.")


# --- FastAPI App Definition with Lifespan ---
app = FastAPI(lifespan=lifespan)

# --- Helper Functions for Data Processing (with better error handling) ---
def process_youtube_video(url: str, store: ChromaStore, embedder: Embedder):
    """Helper function to process a YouTube video and add to the vector store."""
    print(f"\n--- Processing YouTube Video: {url} ---")
    try:
        video_path, transcript, frame_paths = extract_youtube_data(url)
        if not transcript:
            raise ValueError("Failed to get transcript. The video may be private, have transcripts disabled, or the URL is invalid.")

        text_chunks = [segment['text'] for segment in transcript]
        metadatas = [{'source': url, 'timestamp': segment['start']} for segment in transcript]
        ids = [str(uuid.uuid4()) for _ in text_chunks]
        if text_chunks:
            text_embeddings = embedder.embed_text(text_chunks)
            store.text_collection.add(embeddings=text_embeddings.tolist(), documents=text_chunks, metadatas=metadatas, ids=ids)

        if frame_paths:
            image_metadatas = [{'image_path': path, 'source': url} for path in frame_paths]
            image_ids = [str(uuid.uuid4()) for _ in frame_paths]
            image_embeddings = embedder.embed_images(frame_paths)
            store.image_collection.add(embeddings=image_embeddings.tolist(), documents=frame_paths, metadatas=image_metadatas, ids=image_ids)
        return True, "YouTube video processed successfully."
    except Exception as e:
        print(f"ERROR in YouTube processing: {e}")
        return False, str(e)


def process_pdf_file(file_path: str, store: ChromaStore, embedder: Embedder):
    """Helper function to process a PDF file and add to the vector store."""
    print(f"\n--- Processing PDF: {file_path} ---")
    try:
        all_text, all_image_paths = extract_pdf_data(file_path)
        
        if all_text:
            text_chunks = [item['text'] for item in all_text]
            metadatas = [{'source': file_path, 'page': item['page']} for item in all_text]
            ids = [str(uuid.uuid4()) for _ in text_chunks]
            text_embeddings = embedder.embed_text(text_chunks)
            store.text_collection.add(embeddings=text_embeddings.tolist(), documents=text_chunks, metadatas=metadatas, ids=ids)

        if all_image_paths:
            image_metadatas = [{'image_path': path, 'source': file_path} for path in all_image_paths]
            image_ids = [str(uuid.uuid4()) for _ in all_image_paths]
            image_embeddings = embedder.embed_images(all_image_paths)
            store.image_collection.add(embeddings=image_embeddings.tolist(), documents=all_image_paths, metadatas=image_metadatas, ids=image_ids)
        return True, "PDF processed successfully."
    except Exception as e:
        print(f"ERROR in PDF processing: {e}")
        return False, str(e)

# --- API Data Models ---
class ChatRequest(BaseModel):
    thread_id: str
    message: str

# --- API Endpoints ---
@app.post("/process-youtube/")
async def process_youtube_endpoint(url: str = Form(...)):
    success, message = process_youtube_video(url, store, embedder)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to process YouTube video: {message}")
    return {"status": "success", "message": message}


@app.post("/process-pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    success, message = process_pdf_file(file_path, store, embedder)
    
    os.remove(file_path)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {message}")
    return {"status": "success", "message": message}


@app.post("/chat/")
async def chat(request: ChatRequest):
    if app_graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized.")
        
    config = {"configurable": {"thread_id": request.thread_id}}
    final_response = ""
    events = app_graph.stream({"messages": [("user", request.message)]}, config=config, stream_mode="values")
    for event in events:
        if "messages" in event:
            final_response = event["messages"][-1].content
    return {"response": final_response}

app.mount("/", StaticFiles(directory="static", html=True), name="static")