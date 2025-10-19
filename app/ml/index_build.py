from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.settings import settings
from app.storage.lancedb_store import LanceDBStore, VectorRow
from app.ml.embeddings import embed_text_batch, embed_images_batch, embed_query_for_images

_SPLITTER = SentenceSplitter(chunk_size=512, chunk_overlap=64)
_LANCEDB_STORE = LanceDBStore(settings.paths.lancedb_dir)
_VERSION_FILE = Path(settings.paths.lancedb_dir) / "index_versions.json"


def _load_versions() -> Dict[str, int]:
    if not _VERSION_FILE.exists():
        return {}
    try:
        return json.loads(_VERSION_FILE.read_text())
    except Exception:
        return {}


def _save_versions(versions: Dict[str, int]) -> None:
    _VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _VERSION_FILE.write_text(json.dumps(versions))


def _bump_version(user_id: str) -> int:
    versions = _load_versions()
    versions[user_id] = versions.get(user_id, 0) + 1
    _save_versions(versions)
    return versions[user_id]


def get_index_version(user_id: str) -> int:
    """Return the current index version for the given user."""
    versions = _load_versions()
    return versions.get(user_id, 0)


def index_text_nodes(user_id: str, nodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Chunk and index text nodes into LanceDB.

    Each node requires fields: id, text, metadata(dict).
    """
    documents: List[Document] = []
    for node in nodes:
        text = str(node.get("text") or "").strip()
        if not text:
            continue
        metadata = dict(node.get("metadata", {}))
        doc_id = str(node.get("id"))
        documents.append(Document(text=text, metadata=metadata, doc_id=doc_id))

    if not documents:
        return []

    parsed_nodes = _SPLITTER.get_nodes_from_documents(documents)
    texts = [item.get_content(metadata_mode="all") for item in parsed_nodes]
    if not texts:
        return []

    embeddings = embed_text_batch(texts)
    rows: List[VectorRow] = []
    stored: List[Dict[str, object]] = []
    for parsed, embedding in zip(parsed_nodes, embeddings):
        meta = dict(parsed.metadata)
        meta.update(
            {
                "doc_id": parsed.ref_doc_id or parsed.node_id,
                "user_id": user_id,
                "modality": "text",
                "source": meta.get("source"),
            }
        )
        rows.append(
            VectorRow(
                chunk_id=parsed.node_id,
                user_id=user_id,
                document_id=meta["doc_id"],
                modality="text",
                embedding=embedding.tolist(),
                meta=meta,
            )
        )
        stored.append(
            {
                "chunk_id": parsed.node_id,
                "metadata": meta,
                "text": parsed.get_content(metadata_mode="none"),
            }
        )

    if rows:
        _LANCEDB_STORE.upsert_text_vectors(rows)
        _bump_version(user_id)
    return stored


def index_image_nodes(user_id: str, nodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Index image nodes in LanceDB using CLIP embeddings.

    Each node requires: metadata[file_path].
    """
    paths: List[Path] = []
    rows: List[VectorRow] = []
    for node in nodes:
        metadata = dict(node.get("metadata", {}))
        file_path = Path(str(metadata.get("file_path", "")))
        if not file_path.exists():
            continue
        chunk_id = str(node.get("id"))
        metadata.update(
            {
                "doc_id": metadata.get("doc_id", chunk_id),
                "user_id": user_id,
                "modality": "image",
                "source": metadata.get("source"),
            }
        )
        paths.append(file_path)
        rows.append(
            VectorRow(
                chunk_id=chunk_id,
                user_id=user_id,
                document_id=metadata["doc_id"],
                modality="image",
                embedding=[],  # placeholder, filled after embeddings computed
                meta=metadata,
            )
        )

    if not rows:
        return []

    embeddings = embed_images_batch(paths)
    for row, embedding in zip(rows, embeddings):
        row.embedding = embedding.tolist()

    _LANCEDB_STORE.upsert_image_vectors(rows)
    _bump_version(user_id)
    return [
        {
            "chunk_id": row.chunk_id,
            "metadata": row.meta,
        }
        for row in rows
    ]


__all__ = [
    "index_text_nodes",
    "index_image_nodes",
    "embed_query_for_images",
]
