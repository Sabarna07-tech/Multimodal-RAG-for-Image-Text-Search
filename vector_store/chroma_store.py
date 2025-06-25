# multimodal_rag/vector_store/chroma_store.py

import chromadb
import numpy as np
import os

class ChromaStore:
    """
    A class to manage storing and retrieving embeddings using ChromaDB.
    """
    def __init__(self, db_path="output/chroma_db"):
        """
        Initializes the ChromaStore by setting up the client and collections.

        Args:
            db_path (str): The directory to store the ChromaDB database files.
        """
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        self.client = chromadb.PersistentClient(path=db_path)

        # Create or get collections for text and images
        self.text_collection = self.client.get_or_create_collection(name="text_embeddings")
        self.image_collection = self.client.get_or_create_collection(name="image_embeddings")
        
        print("ChromaDB client initialized.")
        print(f"Available collections: {self.client.list_collections()}")

    def add_text_embeddings(self, embeddings, documents, metadatas, ids):
        """
        Adds text embeddings and their corresponding metadata to the collection.

        Args:
            embeddings (numpy.ndarray): The text embeddings.
            documents (list): The original text content.
            metadatas (list): A list of dictionaries with metadata for each text.
            ids (list): A list of unique string IDs for each text.
        """
        if len(embeddings) == 0:
            print("No text embeddings to add.")
            return
            
        self.text_collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(embeddings)} text embeddings to the collection.")

    def add_image_embeddings(self, embeddings, metadatas, ids):
        """
        Adds image embeddings and their corresponding metadata to the collection.
        Note: For images, the 'document' is often just the file path or a URI.

        Args:
            embeddings (numpy.ndarray): The image embeddings.
            metadatas (list): A list of dictionaries with metadata for each image.
            ids (list): A list of unique string IDs for each image.
        """
        if len(embeddings) == 0:
            print("No image embeddings to add.")
            return
            
        # Chroma needs a 'document' for each embedding, we can use the image path (from metadata)
        documents = [meta['image_path'] for meta in metadatas]

        self.image_collection.add(
            embeddings=embeddings,
            documents=documents, # Storing the path as the document
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(embeddings)} image embeddings to the collection.")

    def search_text(self, query_embedding, n_results=5):
        """
        Searches the text collection for the most similar embeddings.

        Args:
            query_embedding (numpy.ndarray): The embedding of the search query.
            n_results (int): The number of results to return.

        Returns:
            dict: The search results from ChromaDB.
        """
        return self.text_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

    def search_images(self, query_embedding, n_results=5):
        """
        Searches the image collection for the most similar embeddings.

        Args:
            query_embedding (numpy.ndarray): The embedding of the search query.
            n_results (int): The number of results to return.

        Returns:
            dict: The search results from ChromaDB.
        """
        return self.image_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

# --- Example Usage ---
if __name__ == '__main__':
    # This block is for testing the module in isolation.
    store = ChromaStore()

    # Create dummy data to test
    dummy_text_embeddings = np.random.rand(2, 384).astype(np.float32) # all-MiniLM-L6-v2 has 384 dimensions
    dummy_texts = ["A sample sentence.", "Another piece of text."]
    dummy_text_metadatas = [{"source": "test"}, {"source": "test"}]
    dummy_text_ids = ["text_1", "text_2"]
    
    dummy_image_embeddings = np.random.rand(1, 512).astype(np.float32) # clip-ViT-B-32 has 512 dimensions
    dummy_image_metadatas = [{"image_path": "dummy_image.jpg", "source": "test"}]
    dummy_image_ids = ["image_1"]

    # Test adding data
    store.add_text_embeddings(dummy_text_embeddings, dummy_texts, dummy_text_metadatas, dummy_text_ids)
    store.add_image_embeddings(dummy_image_embeddings, dummy_image_metadatas, dummy_image_ids)
    
    # Test searching
    query_text_emb = np.random.rand(1, 384).astype(np.float32)
    text_results = store.search_text(query_text_emb)
    print("\nText search results:", text_results['documents'])

    query_image_emb = np.random.rand(1, 512).astype(np.float32)
    image_results = store.search_images(query_image_emb)
    print("Image search results:", image_results['documents'])