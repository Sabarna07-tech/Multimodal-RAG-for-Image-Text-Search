from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple char-based chunker (robust & dependency-free). 
    Use tokens if you want tighter control later.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end == n: break
        i = end - overlap if end - overlap > i else end
    return chunks

def chunk_pages(pages: List[Dict], chunk_size: int = 1200, overlap: int = 200) -> List[Dict]:
    """
    pages: [{"page": int, "text": str}, ...]
    returns: [{"page": int, "chunk_index": int, "text": str}, ...]
    """
    out = []
    for page in pages:
        pno = page.get("page", None)
        txt = page.get("text", "")
        parts = chunk_text(txt, chunk_size, overlap)
        for idx, part in enumerate(parts):
            out.append({"page": pno, "chunk_index": idx, "text": part})
    return out
