from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import google.generativeai as genai

from app.settings import settings

_GEN_MODEL = None


def _ensure_model():
    global _GEN_MODEL
    if _GEN_MODEL is None:
        genai.configure(api_key=settings.gemini.api_key)
        _GEN_MODEL = genai.GenerativeModel(settings.gemini.model)
    return _GEN_MODEL


def _format_citation(meta: Dict[str, Any]) -> str:
    doc_id = meta.get("doc_id", "unknown")
    page_no = meta.get("page_no")
    start_ts = meta.get("start_ts")
    end_ts = meta.get("end_ts")
    if page_no is not None:
        return f"[doc:{doc_id} p:{page_no}]"
    if start_ts is not None and end_ts is not None:
        return f"[ts:{int(start_ts)}-{int(end_ts)}]"
    return f"[doc:{doc_id}]"


def _build_prompt(query: str, items: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    text_sections: List[str] = []
    image_paths: List[str] = []

    for item in items:
        meta = item["metadata"]
        citation = _format_citation(meta)
        snippet = item.get("text") or meta.get("summary") or ""
        if item["modality"] == "image" and meta.get("file_path"):
            image_paths.append(meta["file_path"])
        if snippet:
            text_sections.append(f"{citation} {snippet}")

    prompt = (
        "You are a grounded assistant. Use only the provided evidence to answer the user's question.\n"
        "Cite sources inline using the provided citation tokens (e.g., [doc:abc p:2]).\n"
        "If the evidence is insufficient, clearly state that.\n\n"
        f"User Question:\n{query}\n\n"
        "Evidence:\n" + "\n".join(f"- {section}" for section in text_sections)
    )
    return prompt, image_paths


def _confidence_low(items: List[Dict[str, Any]]) -> bool:
    if not items:
        return True
    top_score = max(item.get("combined_score", item.get("score", 0.0)) for item in items)
    return top_score < settings.retrieval.confidence_tau


def generate_response(query: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a grounded answer with inline citations, abstaining when evidence is weak.
    """
    citations = {
        "text": [item["metadata"] for item in items if item["modality"] == "text"],
        "images": [item["metadata"] for item in items if item["modality"] == "image"],
    }

    if _confidence_low(items):
        snippets = []
        for item in items[:3]:
            snippet = item.get("text")
            if snippet:
                snippets.append(f"{_format_citation(item['metadata'])} {snippet}")
        answer = "I'm not confident enough to answer with the available evidence."
        if snippets:
            answer += "\nRelevant snippets:\n" + "\n".join(f"- {snippet}" for snippet in snippets)
        return {"response": answer, "citations": citations}

    prompt, image_paths = _build_prompt(query, items)
    payload: List[Any] = [prompt]
    for path in image_paths[: settings.retrieval.final_n]:
        if not os.path.exists(path):
            continue
        mime = "image/jpeg"
        if path.lower().endswith(".png"):
            mime = "image/png"
        with open(path, "rb") as fp:
            payload.append({"mime_type": mime, "data": fp.read()})

    model = _ensure_model()
    response = model.generate_content(payload)
    answer = getattr(response, "text", "") or ""
    return {"response": answer, "citations": citations}


__all__ = ["generate_response"]
