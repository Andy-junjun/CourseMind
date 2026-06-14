from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_store import keyword_score
from src.vector_store import DEFAULT_CHUNKS_PATH, load_chunks


DEFAULT_EVAL_DIR = Path("data/eval")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/training/embedding_pairs.jsonl")


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    expected_file: str = ""
    expected_topic: str = ""
    query_type: str = ""
    answer_keywords: str = ""
    source_file: str = ""
    source_prefix: str = ""


@dataclass(frozen=True)
class EmbeddingPair:
    query: str
    positive_chunk_id: str
    positive_file: str
    positive_text: str
    negative_chunk_id: str
    negative_file: str
    negative_text: str
    expected_topic: str
    source_query_file: str


def find_query_files(eval_dir: Path, raw_dir: Path = DEFAULT_RAW_DIR) -> list[Path]:
    paths: list[Path] = []
    if eval_dir.exists():
        paths.extend(eval_dir.glob("retrieval_queries*.csv"))
    if raw_dir.exists():
        paths.extend(raw_dir.rglob("questions.csv"))
    return sorted(paths, key=lambda path: path.as_posix())


def load_retrieval_queries(paths: list[Path]) -> list[RetrievalQuery]:
    queries: list[RetrievalQuery] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                query = (row.get("query") or "").strip()
                if not query:
                    continue
                query_type = (row.get("query_type") or "").strip()
                if query_type == "out_of_scope":
                    continue
                queries.append(
                    RetrievalQuery(
                        query=query,
                        expected_file=(row.get("expected_file") or "").strip(),
                        expected_topic=(row.get("expected_topic") or "").strip(),
                        query_type=query_type,
                        answer_keywords=(row.get("answer_keywords") or "").strip(),
                        source_file=path.as_posix(),
                        source_prefix=infer_source_prefix(path),
                    )
                )
    return queries


def build_pairs(
    queries: list[RetrievalQuery],
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    *,
    target_count: int = 200,
    negatives_per_query: int = 12,
) -> list[EmbeddingPair]:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    pairs: list[EmbeddingPair] = []
    for query in queries:
        positive = select_positive_chunk(query, chunks)
        if positive is None:
            continue
        negatives = select_negative_chunks(query, chunks, positive.chunk_id)
        for negative in negatives[:negatives_per_query]:
            pairs.append(
                EmbeddingPair(
                    query=query.query,
                    positive_chunk_id=positive.chunk_id,
                    positive_file=positive.file_name,
                    positive_text=positive.text,
                    negative_chunk_id=negative.chunk_id,
                    negative_file=negative.file_name,
                    negative_text=negative.text,
                    expected_topic=query.expected_topic,
                    source_query_file=query.source_file,
                )
            )
            if len(pairs) >= target_count:
                return pairs
    return pairs


def select_positive_chunk(query: RetrievalQuery, chunks):
    candidates = filter_chunks_for_query_source(query, chunks)
    if query.expected_file:
        matching = [chunk for chunk in candidates if file_matches(query.expected_file, chunk.file_name)]
        if not matching:
            return None
        candidates = matching

    scored = [(positive_score(query, chunk), chunk) for chunk in candidates]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    score, chunk = scored[0]
    return chunk if score > 0 else None


def select_negative_chunks(query: RetrievalQuery, chunks, positive_chunk_id: str):
    candidates = [
        chunk
        for chunk in filter_chunks_for_query_source(query, chunks)
        if chunk.chunk_id != positive_chunk_id
    ]
    scored = [(negative_score(query, chunk), chunk) for chunk in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored]


def positive_score(query: RetrievalQuery, chunk) -> float:
    score = keyword_score(query.query, chunk)
    expected_file = normalize(query.expected_file)
    chunk_file = normalize(chunk.file_name)
    if expected_file and file_matches(query.expected_file, chunk.file_name):
        score += 10.0
    topic = normalize(query.expected_topic)
    if topic and topic != normalize("资料外"):
        if topic in normalize(" ".join(chunk.concepts)):
            score += 5.0
        if topic in normalize(chunk.text):
            score += 2.0
    for keyword in split_keywords(query.answer_keywords):
        if normalize(keyword) in normalize(chunk.text):
            score += 1.5
    return score


def negative_score(query: RetrievalQuery, chunk) -> float:
    topic = normalize(query.expected_topic)
    score = keyword_score(query.query, chunk)
    if topic and topic in normalize(" ".join(chunk.concepts)):
        score += 1.0
    return score


def split_keywords(value: str) -> list[str]:
    return [part.strip() for part in value.replace("；", ";").split(";") if part.strip()]


def infer_source_prefix(path: Path) -> str:
    parts = list(path.parts)
    if len(parts) >= 3 and parts[-1].lower() == "questions.csv":
        for index in range(len(parts) - 2):
            if parts[index] == "data" and parts[index + 1] == "raw":
                return Path(*parts[index + 2 : -1]).as_posix()
    return ""


def filter_chunks_for_query_source(query: RetrievalQuery, chunks):
    if not query.source_prefix:
        return chunks
    prefix = normalize_path(query.source_prefix) + "/"
    return [chunk for chunk in chunks if normalize_path(chunk.file_name).startswith(prefix)]


def file_matches(expected_file: str, chunk_file: str) -> bool:
    expected = normalize_path(expected_file)
    actual = normalize_path(chunk_file)
    expected_stem = normalize_path(Path(expected_file).stem)
    return bool(
        expected
        and (
            expected in actual
            or actual.endswith("/" + expected)
            or (expected_stem and expected_stem in actual)
        )
    )


def normalize_path(value: str) -> str:
    return normalize(value).replace("\\", "/")


def normalize(value: str) -> str:
    return "".join(value.lower().split())


def save_pairs(pairs: list[EmbeddingPair], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for pair in pairs:
            file.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build query-positive-negative training pairs for embedding fine-tuning."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--negatives-per-query", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_files = find_query_files(args.eval_dir, args.raw_dir)
    if not query_files:
        raise FileNotFoundError(f"No retrieval_queries*.csv found in {args.eval_dir}")
    queries = load_retrieval_queries(query_files)
    pairs = build_pairs(
        queries,
        chunks_path=args.chunks_path,
        target_count=args.target_count,
        negatives_per_query=args.negatives_per_query,
    )
    save_pairs(pairs, args.output_path)
    print(f"Loaded query files: {', '.join(path.as_posix() for path in query_files)}")
    print(f"Built {len(pairs)} embedding training pairs")
    print(f"Output: {args.output_path}")
    if len(pairs) < args.target_count:
        print(
            f"Warning: target_count={args.target_count}, but only {len(pairs)} pairs were built. "
            "Add more retrieval queries or chunks for a stronger fine-tuning set."
        )


if __name__ == "__main__":
    main()
