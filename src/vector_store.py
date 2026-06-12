import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.embedder import embed_text, get_embedding_model_path
from src.schemas import Chunk


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX_PATH = Path("data/indexes/faiss.index")
INDEX_VERSION = 1


@dataclass(frozen=True)
class FaissVectorIndex:
    index: Any
    chunk_ids: list[str]
    dim: int


VectorIndex = FaissVectorIndex


def build_index(chunks: list[Chunk]) -> VectorIndex:
    vectors = embed_chunks(chunks)
    dim = int(vectors.shape[1]) if len(vectors) else 0
    faiss = import_faiss()
    index = faiss.IndexFlatIP(dim)
    if len(vectors):
        index.add(vectors)
    return FaissVectorIndex(index=index, chunk_ids=[chunk.chunk_id for chunk in chunks], dim=dim)


def search_index(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
    index: VectorIndex | None = None,
) -> list[tuple[Chunk, float]]:
    if top_k <= 0 or not chunks:
        return []

    vector_index = index or build_index(chunks)
    if vector_index.index.ntotal == 0:
        return []

    query_vector = vector_matrix([embed_text(query)])
    distances, positions = vector_index.index.search(query_vector, min(top_k, len(chunks)))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    results: list[tuple[Chunk, float]] = []
    for position, score in zip(positions[0], distances[0]):
        if position < 0 or position >= len(vector_index.chunk_ids):
            continue
        chunk_id = vector_index.chunk_ids[int(position)]
        chunk = chunk_by_id.get(chunk_id)
        if chunk is not None:
            results.append((chunk, float(score)))
    return results


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype="float32")
    return vector_matrix([embed_text(chunk.text) for chunk in chunks])


def vector_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    return np.ascontiguousarray(matrix)


def save_chunks(chunks: list[Chunk], path: Path | str = DEFAULT_CHUNKS_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path | str = DEFAULT_CHUNKS_PATH) -> list[Chunk]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    chunks: list[Chunk] = []
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            chunks.append(chunk_from_dict(json.loads(line)))
    return chunks


def save_index(
    index: VectorIndex,
    path: Path | str = DEFAULT_INDEX_PATH,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    faiss = import_faiss()
    faiss.write_index(index.index, str(target))
    meta_payload = {
        "version": INDEX_VERSION,
        "backend": "faiss",
        "metric": "inner_product",
        "dim": index.dim,
        "chunk_ids": index.chunk_ids,
        "metadata": metadata or {},
        "embedding": embedding_metadata(),
    }
    index_metadata_path(target).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_index(path: Path | str = DEFAULT_INDEX_PATH) -> VectorIndex:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    meta_path = index_metadata_path(source)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    faiss = import_faiss()
    index = faiss.read_index(str(source))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_ids = [str(chunk_id) for chunk_id in metadata.get("chunk_ids", [])]
    dim = int(metadata.get("dim", index.d))
    if len(chunk_ids) != index.ntotal:
        raise ValueError(
            f"FAISS index metadata count mismatch: {len(chunk_ids)} ids vs {index.ntotal} vectors"
        )
    return FaissVectorIndex(index=index, chunk_ids=chunk_ids, dim=dim)


def persist_vector_store(
    chunks: list[Chunk],
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
    *,
    metadata: dict[str, Any] | None = None,
) -> VectorIndex:
    index = build_index(chunks)
    save_chunks(chunks, chunks_path)
    save_index(index, index_path, metadata=metadata)
    return index


def load_vector_store(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> tuple[list[Chunk], VectorIndex]:
    chunks = load_chunks(chunks_path)
    index = load_index(index_path)
    return chunks, index


def vector_store_exists(
    chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> bool:
    return (
        Path(chunks_path).exists()
        and Path(index_path).exists()
        and index_metadata_path(index_path).exists()
    )


def index_metadata_path(index_path: Path | str) -> Path:
    return Path(index_path).with_suffix(".meta.json")


def chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(data["chunk_id"]),
        file_name=str(data["file_name"]),
        page=int(data["page"]),
        text=str(data["text"]),
        chapter=data.get("chapter"),
        concepts=list(data.get("concepts") or []),
    )


def import_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "FAISS vector search requires faiss-cpu. Install it with `pip install faiss-cpu`."
        ) from exc
    return faiss


def embedding_metadata() -> dict[str, str]:
    import os

    provider = os.getenv("EMBEDDING_PROVIDER", "lite").strip().lower()
    payload = {
        "mode": os.getenv("COURSEMIND_MODE", "mock"),
        "provider": provider,
    }
    if provider in {"sentence_transformers", "sentence-transformer", "st"}:
        payload["model_path"] = get_embedding_model_path()
    elif provider in {"lite", "local_lite", "ngram"}:
        payload["dim"] = os.getenv("EMBEDDING_DIM", "384")
    return payload


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb or 1.0)
