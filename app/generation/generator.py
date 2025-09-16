from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from PIL import Image


class Generator:
    """Wrapper around Gemini for generating answers with text + image context."""

    def __init__(self, api_key: str, model_name: str = "gemini-pro-vision") -> None:
        if not api_key:
            raise ValueError("A valid Google API key must be provided.")

        self.model_name = model_name
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception as exc:  # pragma: no cover - depends on external service
            # Surface a helpful error at call time rather than import time.
            print(f"Error initializing Gemini model: {exc}")
            self.model = None

    def _format_text_context(self, text_results: Optional[Dict[str, Any]]) -> List[str]:
        output: List[str] = ["Here is the most relevant text I found:"]
        if not text_results:
            output.append("- No relevant text found.")
            return output

        documents = text_results.get("documents") or []
        metadatas = text_results.get("metadatas") or []
        if not documents:
            output.append("- No relevant text found.")
            return output

        docs = documents[0]
        metas = metadatas[0] if metadatas else [{} for _ in docs]
        for doc, meta in zip(docs, metas):
            source = meta.get("source") or meta.get("file_name") or "unknown"
            output.append(f"- From {source}: '{doc}' (metadata={meta})")
        return output

    def _prepare_image_parts(self, image_results: Optional[Dict[str, Any]], prompt_parts: List[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        prompt_parts.append("\nHere are the most relevant images I found:")
        if not image_results:
            prompt_parts.append("- No relevant images found.")
            return images

        documents = image_results.get("documents") or []
        metadatas = image_results.get("metadatas") or []
        if not documents:
            prompt_parts.append("- No relevant images found.")
            return images

        docs = documents[0]
        metas = metadatas[0] if metadatas else [{} for _ in docs]
        for doc, meta in zip(docs, metas):
            image_path = meta.get("image_path") or doc
            if image_path and os.path.exists(image_path):
                prompt_parts.append(f"- Image from {meta.get('source', 'unknown')} (metadata={meta})")
                with Image.open(image_path) as img:
                    # Convert to RGB to avoid surprises with palette/alpha modes.
                    images.append(img.convert("RGB"))
            else:
                prompt_parts.append(f"- Could not load image from path: {image_path}")
        return images

    def generate_answer(
        self,
        query: str,
        retrieved_text: Optional[Dict[str, Any]],
        retrieved_images: Optional[Dict[str, Any]],
    ) -> str:
        if not self.model:
            return "Generator model not initialized. Cannot generate answer."

        prompt_parts = [f"User Query: {query}\n"]
        prompt_parts.extend(self._format_text_context(retrieved_text))

        image_parts = self._prepare_image_parts(retrieved_images, prompt_parts)
        prompt_parts.append("\n---\nBased on all the provided text and images, answer the user query.")

        # Gemini expects an alternating list of strings and images.
        payload: List[Any] = ["\n".join(prompt_parts)]
        payload.extend(image_parts)

        try:
            response = self.model.generate_content(payload)
            return getattr(response, "text", "") or ""
        except Exception as exc:  # pragma: no cover - service call may fail
            return f"An error occurred while generating the answer: {exc}"