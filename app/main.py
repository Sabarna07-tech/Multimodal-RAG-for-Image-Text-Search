import os
import uuid
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated, Sequence, TypedDict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# --- Core App Imports ---
from .core.config import settings

# --- Our RAG Modules ---
from .data_extraction.youtube_extractor import extract_youtube_data
from .data_extraction.pdf_extractor import extract_pdf_data
from .embedding.embedder import Embedder
from .vector_store.chroma_store import ChromaStore
from .retrieval.retriever import Retriever
from .generation.generator import Generator

# --- Global App Components (Initialized on Startup) ---
embedder = Embedder()
generator = Generator(api_key=settings.GOOGLE_API_KEY)
# The graph is now stateless and compiled on-the-fly in the chat endpoint
app_graph_definition = None

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key")

def get_user_id(api_key: str = Security(api_key_header)):
    """Dependency to validate API key and return user_id."""
    if api_key in settings.API_KEYS:
        return settings.API_KEYS[api_key]
    raise HTTPException(status_code=403, detail="Invalid API Key")

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown."""
    print("--- Application Startup ---")
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
    os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def retrieve_node(state: AgentState, config):
        user_id = config["configurable"]["user_id"]
        retriever = Retriever(
            embedder,
            ChromaStore(db_path=settings.CHROMA_DB_PATH, user_id=user_id).text_collection,
            ChromaStore(db_path=settings.CHROMA_DB_PATH, user_id=user_id).image_collection
        )
        last_message = state['messages'][-1].content
        retrieved_results = retriever.retrieve(last_message, n_results=3)
        context_str = f"--- CONTEXT ---\nText: {retrieved_results['text']['documents']}\nImages: {retrieved_results['image']['documents']}"
        return {"messages": [("system", context_str)]}

    def generate_node(state: AgentState):
        query = state['messages'][-2].content
        context = state['messages'][-1].content
        response = generator.model.generate_content(f"Based on this context:\n{context}\n\nAnswer this query: {query}").text
        return {"messages": [("ai", response)]}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    global app_graph_definition
    app_graph_definition = workflow.compile()
    print("Graph definition compiled.")
    yield
    print("--- Application Shutdown ---")

# --- FastAPI App Definition ---
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# --- Helper Functions (omitted for brevity, they are unchanged) ---
def process_youtube_video(url: str, store: ChromaStore, embedder: Embedder):
    try:
        video_path, transcript, frame_paths = extract_youtube_data(url)
        if not transcript: raise ValueError("Failed to get transcript.")
        text_chunks = [seg['text'] for seg in transcript]
        metadatas = [{'source': url, 'timestamp': seg['start']} for seg in transcript]
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
        return False, str(e)

def process_pdf_file(file_path: str, store: ChromaStore, embedder: Embedder):
    all_text, all_image_paths = extract_pdf_data(file_path)
    if all_text:
        text_chunks = [item['text'] for item in all_text]
        metadatas = [{'source': os.path.basename(file_path), 'page': item['page']} for item in all_text]
        ids = [str(uuid.uuid4()) for _ in text_chunks]
        text_embeddings = embedder.embed_text(text_chunks)
        store.text_collection.add(embeddings=text_embeddings.tolist(), documents=text_chunks, metadatas=metadatas, ids=ids)
    if all_image_paths:
        image_metadatas = [{'image_path': path, 'source': os.path.basename(file_path)} for path in all_image_paths]
        image_ids = [str(uuid.uuid4()) for _ in all_image_paths]
        image_embeddings = embedder.embed_images(all_image_paths)
        store.image_collection.add(embeddings=image_embeddings.tolist(), documents=all_image_paths, metadatas=image_metadatas, ids=image_ids)

# --- API Data Models ---
class ChatRequest(BaseModel):
    thread_id: str
    message: str

# --- API Endpoints ---
@app.post("/process-youtube/")
async def process_youtube_endpoint(url: str = Form(...), user_id: str = Depends(get_user_id)):
    user_store = ChromaStore(db_path=settings.CHROMA_DB_PATH, user_id=user_id)
    success, message = process_youtube_video(url, user_store, embedder)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to process YouTube video: {message}")
    return {"status": "success", "message": message}

@app.post("/process-pdf/")
async def process_pdf_endpoint(file: UploadFile = File(...), user_id: str = Depends(get_user_id)):
    user_store = ChromaStore(db_path=settings.CHROMA_DB_PATH, user_id=user_id)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        process_pdf_file(tmp_path, user_store, embedder)
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return {"status": "success", "message": f"PDF processed: {file.filename}"}

@app.post("/chat/")
async def chat(request: ChatRequest, user_id: str = Depends(get_user_id)):
    if app_graph_definition is None:
        raise HTTPException(status_code=500, detail="Graph definition not initialized.")

    # Create a user-specific checkpointer
    checkpoint_path = os.path.join(settings.CHECKPOINT_DIR, f"{user_id}_checkpoints.db")
    memory = SqliteSaver.from_conn_string(checkpoint_path)

    # Compile the graph with the user-specific checkpointer
    app_with_memory = app_graph_definition.with_checkpointer(checkpointer=memory)

    config = {"configurable": {"thread_id": request.thread_id, "user_id": user_id}}
    final_response = ""
    events = app_with_memory.stream({"messages": [("user", request.message)]}, config=config, stream_mode="values")
    for event in events:
        if "messages" in event:
            final_response = event["messages"][-1].content
    return {"response": final_response}

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")