# multimodal_rag/retrieval/retriever.py

import sys
import os
import numpy as np

# Add the parent directory to the system path to allow imports from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embedding.embedder import Embedder
from vector_store.chroma_store import ChromaStore

class Retriever:
    """
    Handles the retrieval of relevant text and images from the vector store
    based on a user's query.
    """
    def __init__(self, embedder, text_collection, image_collection):
        """
        Initializes the Retriever.

        Args:
            embedder (Embedder): An instance of the Embedder class.
            text_collection (chromadb.Collection): The text collection from ChromaDB.
            image_collection (chromadb.Collection): The image collection from ChromaDB.
        """
        self.embedder = embedder
        self.text_collection = text_collection
        self.image_collection = image_collection
        print("Retriever initialized.")

    def retrieve(self, query, n_results=5):
        """
        Takes a text query, embeds it, and searches for the most relevant
        text and images.

        Args:
            query (str): The user's search query.
            n_results (int): The number of results to return for each modality.

        Returns:
            dict: A dictionary containing the top text and image results.
        """
        print(f"\nRetrieving results for query: '{query}'")

        # Embed the text query
        # Note: We use the text_model for the query embedding
        query_embedding = self.embedder.text_model.encode(query)

        # Search for relevant text
        text_results = self.text_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Search for relevant images
        # For image search, we also use the same text query embedding.
        # This works because CLIP models are trained to understand text-image pairs.
        image_results = self.image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return {
            "text": text_results,
            "image": image_results
        }

# --- Example Usage ---
if __name__ == '__main__':
    # This block is for testing the module in isolation.
    
    # 1. Initialize the components the retriever depends on
    print("--- Initializing dependencies for testing ---")
    embedder = Embedder()
    store = ChromaStore(db_path="output/test_chroma_db") # Use a separate DB for testing

    # 2. Add some dummy data to the store
    print("\n--- Adding dummy data to the test store ---")
    # Text data
    dummy_texts = ["The sky is blue and the sun is bright.", "Success in life comes from hard work and dedication."]
    text_embeddings = embedder.embed_text(dummy_texts)
    text_metadatas = [{"source": "document1", "page": 1}, {"source": "youtube1", "timestamp": 120}]
    text_ids = ["text_doc1_p1", "text_yt1_t120"]
    store.add_text_embeddings(text_embeddings, dummy_texts, text_metadatas, text_ids)

    # Image data (create a dummy image first)
    try:
        from PIL import Image
        if not os.path.exists('dummy_image_blue_sky.jpg'):
            dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_array[:, :, 2] = 255 # Make it blue
            dummy_image = Image.fromarray(dummy_array)
            dummy_image.save('dummy_image_blue_sky.jpg')

        image_paths = ['dummy_image_blue_sky.jpg']
        image_embeddings = embedder.embed_images(image_paths)
        image_metadatas = [{"image_path": "dummy_image_blue_sky.jpg", "source": "document1", "page": 1}]
        image_ids = ["image_doc1_p1"]
        store.add_image_embeddings(image_embeddings, image_metadatas, image_ids)

    except (ImportError, NameError):
        print("Could not create dummy image. Skipping image data addition.")


    # 3. Initialize the retriever with the test components
    print("\n--- Initializing the Retriever ---")
    retriever = Retriever(embedder, store.text_collection, store.image_collection)

    # 4. Perform a search
    search_query = "What does it take to succeed?"
    results = retriever.retrieve(search_query, n_results=1)

    print("\n--- Search Results ---")
    print("Best text match:", results['text']['documents'])
    print("Best image match:", results['image']['documents'])

    # Clean up dummy files
    if os.path.exists('dummy_image_blue_sky.jpg'):
        os.remove('dummy_image_blue_sky.jpg')