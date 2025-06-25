# multimodal_rag/generation/generator.py

import os
import sys
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
# Add the parent directory to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Generator:
    """
    Handles the final answer generation using a multimodal LLM (Gemini).
    """
    def __init__(self):
        """
        Initializes the Generator by configuring the Gemini API.
        """
        try:
            # Configure the API key from environment variables
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
            
            # Initialize the multimodal model
            self.model = genai.GenerativeModel('gemini-pro-vision')
            print("Gemini Pro Vision model initialized.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            self.model = None

    def generate_answer(self, query, retrieved_text, retrieved_images):
        """
        Generates a comprehensive answer based on the query and retrieved context.

        Args:
            query (str): The user's original query.
            retrieved_text (dict): The text search results from ChromaDB.
            retrieved_images (dict): The image search results from ChromaDB.

        Returns:
            str: The generated answer.
        """
        if not self.model:
            return "Generator model not initialized. Cannot generate answer."

        # --- 1. Build the Prompt ---
        # Start with the user's query
        prompt_parts = [f"User Query: {query}\n\n"]
        
        # Add the context from retrieved text
        prompt_parts.append("Here is the most relevant text I found:\n")
        if retrieved_text and retrieved_text.get('documents'):
            for i, doc in enumerate(retrieved_text['documents'][0]):
                metadata = retrieved_text['metadatas'][0][i]
                prompt_parts.append(f"- Text from {metadata.get('source', 'N/A')}: '{doc}' (Source details: {metadata})\n")
        else:
            prompt_parts.append("- No relevant text found.\n")
            
        # Add the context from retrieved images
        prompt_parts.append("\nHere are the most relevant images I found:\n")
        image_parts = []
        if retrieved_images and retrieved_images.get('documents'):
            for i, doc in enumerate(retrieved_images['documents'][0]):
                metadata = retrieved_images['metadatas'][0][i]
                image_path = metadata.get('image_path')
                if image_path and os.path.exists(image_path):
                    prompt_parts.append(f"- Image from {metadata.get('source', 'N/A')}. (Source details: {metadata})\n")
                    # Append the actual image data for the model to see
                    image_parts.append(Image.open(image_path))
                else:
                    prompt_parts.append(f"- Could not load image from path: {image_path}\n")
        else:
             prompt_parts.append("- No relevant images found.\n")

        # Add the final instruction
        prompt_parts.append("\n---\nBased on all the provided text and images, please provide a comprehensive answer to my user query.")

        # --- 2. Call the Model ---
        # Combine text prompt parts with image parts
        full_prompt = ["\n".join(prompt_parts)] + image_parts
        
        try:
            print("\n--- Sending request to Gemini ---")
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"


# --- Example Usage ---
if __name__ == '__main__':
    # This block is for testing the module in isolation.
    generator = Generator()

    if generator.model:
        # Create dummy retrieved data
        dummy_query = "What does it take to succeed?"
        
        dummy_text_results = {
            'documents': [['Success in life comes from hard work and dedication.']],
            'metadatas': [[{'source': 'youtube1', 'timestamp': 120}]]
        }
        
        # Create a dummy image for testing
        try:
            from PIL import Image
            import numpy as np
            if not os.path.exists('dummy_success_image.jpg'):
                dummy_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                dummy_image = Image.fromarray(dummy_array)
                dummy_image.save('dummy_success_image.jpg')

            dummy_image_results = {
                'documents': [['dummy_success_image.jpg']],
                'metadatas': [[{'image_path': 'dummy_success_image.jpg', 'source': 'document1'}]]
            }
            
            answer = generator.generate_answer(dummy_query, dummy_text_results, dummy_image_results)

            print("\n--- Generated Answer ---")
            print(answer)

            # Clean up
            os.remove('dummy_success_image.jpg')

        except (ImportError, ValueError) as e:
            print(f"Could not run example: {e}")