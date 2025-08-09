# multimodal_rag/embedding/embedder.py

from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

class Embedder:
    """
    A class to handle the embedding of text and images using pre-trained models.
    """
    def __init__(self):
        """
        Initializes the Embedder by loading the text and image embedding models.
        """
        # Use a model optimized for semantic search for text
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use a CLIP model for image embeddings, which understands both images and text
        self.image_model = SentenceTransformer('clip-ViT-B-32')

        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_model.to(self.device)
        self.image_model.to(self.device)
        print(f"Embedding models loaded on {self.device}")

    def embed_text(self, text_list):
        """
        Generates embeddings for a list of text strings.

        Args:
            text_list (list): A list of strings to be embedded.

        Returns:
            numpy.ndarray: A numpy array of embeddings.
        """
        if not text_list:
            return []
        
        print(f"Generating embeddings for {len(text_list)} text snippets...")
        embeddings = self.text_model.encode(text_list, convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def embed_images(self, image_paths):
        """
        Generates embeddings for a list of image files.

        Args:
            image_paths (list): A list of file paths to the images.

        Returns:
            numpy.ndarray: A numpy array of embeddings.
        """
        if not image_paths:
            return []
        
        print(f"Generating embeddings for {len(image_paths)} images...")
        # SentenceTransformer's CLIP model can directly take a list of PIL images
        pil_images = [Image.open(path) for path in image_paths]
        
        embeddings = self.image_model.encode(pil_images, convert_to_tensor=True)
        return embeddings.cpu().numpy()

# --- Example Usage ---
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It's useful for testing the module in isolation.
    
    # Create a dummy image for testing
    if not os.path.exists('dummy_image.jpg'):
        try:
            from PIL import Image
            import numpy as np
            dummy_array = np.zeros((100, 100, 3), dtype=np.uint8)
            dummy_image = Image.fromarray(dummy_array)
            dummy_image.save('dummy_image.jpg')
            print("Created dummy_image.jpg for testing.")
        except ImportError:
            print("Please install Pillow and numpy to create a dummy image for testing.")


    embedder = Embedder()

    # Test text embedding
    sample_texts = ["This is a test sentence.", "Life is about the journey, not the destination."]
    text_embeddings = embedder.embed_text(sample_texts)
    print("Text embeddings shape:", text_embeddings.shape)

    # Test image embedding
    if os.path.exists('dummy_image.jpg'):
        sample_images = ['dummy_image.jpg']
        image_embeddings = embedder.embed_images(sample_images)
        print("Image embeddings shape:", image_embeddings.shape)
        os.remove('dummy_image.jpg')