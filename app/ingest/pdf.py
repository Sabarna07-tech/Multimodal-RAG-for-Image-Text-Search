from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # type: ignore

from app.settings import settings
from app.utils.text_chunk import chunk_text


def extract_pdf_nodes(
    pdf_path: Path,
    user_id: str,
    doc_id: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    Extract text and image nodes from a PDF using PyMuPDF.

    # [MIGRATE] Centralized PDF ingestion returning chunk metadata for indexing.
    """
    text_nodes: List[Dict[str, object]] = []
    image_nodes: List[Dict[str, object]] = []

    with fitz.open(str(pdf_path)) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_no = page_index + 1

            raw_text = page.get_text("text").strip()
            if raw_text:
                chunks = chunk_text(
                    raw_text,
                    chunk_size=settings.chunks.size_chars,
                    overlap=settings.chunks.overlap_chars,
                )
                for chunk_index, chunk_text_value in enumerate(chunks):
                    text_nodes.append(
                        {
                            "id": f"{doc_id}:page{page_no}:chunk{chunk_index}",
                            "text": chunk_text_value,
                            "metadata": {
                                "doc_id": doc_id,
                                "user_id": user_id,
                                "modality": "text",
                                "source": "pdf",
                                "page_no": page_no,
                                "chunk_index": chunk_index,
                            },
                        }
                    )

            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image.get("image")
                ext = base_image.get("ext", "png")

                media_root = Path(settings.paths.media_dir) / "pdf_images" / user_id / doc_id
                media_root.mkdir(parents=True, exist_ok=True)

                filename = f"{doc_id}_page{page_no:03d}_img{image_index:03d}.{ext}"
                file_path = media_root / filename
                with open(file_path, "wb") as fp:
                    fp.write(image_bytes)

                image_nodes.append(
                    {
                        "id": f"{doc_id}:img{page_no}:{image_index}",
                        "metadata": {
                            "doc_id": doc_id,
                            "user_id": user_id,
                            "modality": "image",
                            "source": "pdf",
                            "page_no": page_no,
                            "file_path": str(file_path),
                        },
                    }
                )

    return text_nodes, image_nodes


__all__ = ["extract_pdf_nodes"]

