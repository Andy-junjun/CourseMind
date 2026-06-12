from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import chunk_pages
from src.document_loader import SUPPORTED_TEXT_SUFFIXES, load_pdf
from src.schemas import Chunk, DocumentPage
from src.vector_store import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_PATH,
    persist_vector_store,
)


SUPPORTED_SUFFIXES = {".pdf", *SUPPORTED_TEXT_SUFFIXES}


def collect_documents(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    return sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.relative_to(raw_dir).as_posix(),
    )


def build_vector_database(
    raw_dir: Path = Path("data/raw"),
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    chunk_size: int = 400,
    overlap: int = 80,
) -> tuple[list[DocumentPage], list[Chunk]]:
    documents = collect_documents(raw_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {raw_dir}")

    pages: list[DocumentPage] = []
    for document in documents:
        relative_name = document.relative_to(raw_dir).as_posix()
        pages.extend(
            replace(page, file_name=relative_name) for page in load_pdf(str(document))
        )

    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    persist_vector_store(
        chunks,
        chunks_path=chunks_path,
        index_path=index_path,
        metadata={
            "raw_dir": str(raw_dir),
            "source_files": [path.relative_to(raw_dir).as_posix() for path in documents],
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    )
    return pages, chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CourseMind FAISS vector index.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages, chunks = build_vector_database(
        raw_dir=args.raw_dir,
        chunks_path=args.chunks_path,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Indexed {len(pages)} pages into {len(chunks)} chunks")
    print(f"Chunks: {args.chunks_path}")
    print(f"FAISS index: {args.index_path}")


if __name__ == "__main__":
    main()
