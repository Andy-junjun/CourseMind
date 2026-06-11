from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentPage:
    file_name: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    file_name: str
    page: int
    text: str
    chapter: str | None = None
    concepts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    ranker_score: float = 0.0


@dataclass(frozen=True)
class QuizItem:
    question: str
    options: list[str]
    answer: str
    explanation: str
    concept: str
    source_chunk_id: str


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)

