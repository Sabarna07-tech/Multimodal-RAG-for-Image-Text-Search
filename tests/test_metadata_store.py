from __future__ import annotations

from pathlib import Path

from app.storage.schema import Chunk, Document, MetadataStore


def test_metadata_store_crud(tmp_path: Path) -> None:
    db_path = tmp_path / "metadata.db"
    store = MetadataStore(str(db_path))

    try:
        document = Document(
            id="doc-1",
            user_id="user-123",
            source_type="pdf",
            source_uri="file:///docs/doc-1.pdf",
            title="Doc One",
            status="processing",
        )

        stored_doc = store.upsert_document(document)
        assert stored_doc.status == "processing"

        ready_doc = document.copy(update={"status": "ready"})
        stored_doc = store.upsert_document(ready_doc)
        assert stored_doc.status == "ready"

        chunk = Chunk(
            id="chunk-1",
            document_id=document.id,
            modality="text",
            text="Sample text",
            page_no=1,
            meta={"doc_id": document.id},
        )

        store.upsert_chunks([chunk])
        saved_chunk = store.get_chunk(chunk.id)
        assert saved_chunk is not None
        assert saved_chunk.id == chunk.id

        metadata = store.get_metadata(document.id)
        assert metadata is not None
        assert metadata.document.id == document.id
        assert len(metadata.chunks) == 1

        store.delete_chunk(chunk.id)
        assert store.list_chunks(document.id) == []

        store.delete_document(document.id)
        assert store.get_document(document.id) is None
    finally:
        store.close()
