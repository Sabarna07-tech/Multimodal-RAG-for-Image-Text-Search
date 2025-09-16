# multimodal_rag/vector_store/chroma_store.py

import chromadb
import numpy as np
import os

class ChromaStore:
    """
    A class to manage storing and retrieving embeddings using ChromaDB.
    """
    def __init__(self, db_path: str, user_id: str):
        """
        Initializes the ChromaStore by setting up the client and user-specific
        collections.

        Args:
            db_path (str): The directory to store the ChromaDB database files.
            user_id (str): The unique identifier for the user.
        """
        if not user_id:
            raise ValueError("A user_id must be provided to create a ChromaStore.")

        if not os.path.exists(db_path):
            os.makedirs(db_path)

        self.client = chromadb.PersistentClient(path=db_path)

        # Create or get user-specific collections for text and images
        text_collection_name = f"{user_id}_text_embeddings"
        image_collection_name = f"{user_id}_image_embeddings"

        self.text_collection = self.client.get_or_create_collection(name=text_collection_name)
        self.image_collection = self.client.get_or_create_collection(name=image_collection_name)

        print(f"ChromaDB client initialized for user '{user_id}'.")
        print(f"Using collections: '{text_collection_name}', '{image_collection_name}'")

# No other methods needed, as the collection objects are accessed directly.