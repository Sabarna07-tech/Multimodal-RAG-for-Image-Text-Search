# multimodal_rag/data_extraction/pdf_extractor.py

import fitz  # PyMuPDF
import os

def extract_pdf_data(pdf_path, output_dir="output/pdf_data"):
    """
    Extracts text and images from each page of a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir (str): Directory to save extracted images.

    Returns:
        tuple: A tuple containing a list of text per page and a list of image paths.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        doc = fitz.open(pdf_path)
        all_text = []
        all_image_paths = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract text
            text = page.get_text("text")
            all_text.append({"page": page_num + 1, "text": text})
            print(f"Extracted text from page {page_num + 1}")

            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(output_dir, f"page{page_num+1}_img{img_index}.{image_ext}")
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                all_image_paths.append(image_filename)
                print(f"Saved image: {image_filename}")

        return all_text, all_image_paths

    except Exception as e:
        print(f"An error occurred in PDF extraction: {e}")
        return [], []