# multimodal_rag/main_pipeline.py

import os
import uuid
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

# Use the state management and graph components from langgraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Import our existing modules
from data_extraction.youtube_extractor import extract_youtube_data
from data_extraction.pdf_extractor import extract_pdf_data
from embedding.embedder import Embedder
from vector_store.chroma_store import ChromaStore
from retrieval.retriever import Retriever
from generation.generator import Generator

# --- 1. Define the State for our Graph ---
# The state will now manage the conversation history.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def main():
    """
    The main function to run the complete Multimodal RAG pipeline
    with conversational memory.
    """
    print("--- Initializing the Stateful Multimodal RAG Pipeline ---")
    embedder = Embedder()
    store = ChromaStore(db_path="output/main_chroma_db")
    retriever = Retriever(embedder, store.text_collection, store.image_collection)
    generator = Generator()
    
    # The checkpointer is our memory
    memory = SqliteSaver.from_conn_string(":memory:") # Use in-memory SQLite for this example

    # --- 2. Define the Nodes of our Graph ---
    
    def retrieve_node(state: AgentState):
        """Node that retrieves relevant documents."""
        last_message = state['messages'][-1].content
        retrieved_results = retriever.retrieve(last_message, n_results=3)
        
        # We will pass the retrieved info to the generator via the state
        # For simplicity, we'll format it as a string for now.
        context_str = "--- CONTEXT ---\n"
        context_str += "Relevant text:\n" + str(retrieved_results['text']['documents'])
        context_str += "\nRelevant images:\n" + str(retrieved_results['image']['documents'])
        
        # To avoid passing complex objects, we pass a simple message
        # In a more advanced setup, you could add fields to AgentState for this
        return {"messages": [("system", context_str)]}

    def generate_node(state: AgentState):
        """Node that generates the final answer."""
        query = state['messages'][-2].content # The user's query
        context = state['messages'][-1].content # The context from the retriever
        
        # Simplified call for demonstration
        # A real implementation would parse the context string more robustly
        final_answer = generator.model.generate_content(
            f"Based on this context:\n{context}\n\nAnswer this query: {query}"
        ).text

        return {"messages": [("ai", final_answer)]}


    # --- 3. Build the Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph with memory
    app = workflow.compile(checkpointer=memory)

    # --- 4. Data Processing (Same as before) ---
    # In a real app, you might do this once and save the Chroma DB
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 
    process_youtube_video(youtube_url, store, embedder)

    # --- 5. Conversational Loop ---
    # We define a thread_id for our conversation
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("\n--- Ready to Chat (type 'exit' to end) ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
            
        # Invoke the graph with the user's message
        # The checkpointer ensures the entire conversation history is loaded
        events = app.stream(
            {"messages": [("user", user_input)]}, config=config, stream_mode="values"
        )
        
        for event in events:
            # The final message from the 'generate' node is our answer
            if "messages" in event:
                event["messages"][-1].pretty_print()


# The data processing functions remain the same as the previous version
def process_youtube_video(url, store, embedder):
    """Processes a YouTube video and adds its data to the vector store."""
    print(f"\n--- Processing YouTube Video: {url} ---")
    # This function is the same as the one in the previous step
    # ...

def process_pdf(pdf_path, store, embedder):
    """Processes a PDF and adds its data to the vector store."""
    print(f"\n--- Processing PDF: {pdf_path} ---")
    # This function is the same as the one in the previous step
    # ...


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Please set the GOOGLE_API_KEY environment variable.")
    else:
        main()