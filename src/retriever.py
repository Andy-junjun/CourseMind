from src.bm25_store import keyword_score
from src.embedder import embed_text
from src.schemas import Chunk, RetrievedChunk
from src.vector_store import cosine


def retrieve(query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievedChunk]:
    query_vector = embed_text(query)
    candidates = []
    for chunk in chunks:
        dense = cosine(query_vector, embed_text(chunk.text))
        bm25 = keyword_score(query, chunk)
        candidates.append(RetrievedChunk(chunk=chunk, dense_score=dense, bm25_score=bm25))
    candidates.sort(key=lambda item: (item.bm25_score, item.dense_score), reverse=True)
    return candidates[:top_k]

