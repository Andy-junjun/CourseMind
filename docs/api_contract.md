# API Contract

These signatures are frozen for the first sprint.

```python
load_pdf(file_path: str) -> list[DocumentPage]
chunk_pages(pages: list[DocumentPage], chunk_size: int = 400, overlap: int = 80) -> list[Chunk]
retrieve(query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievedChunk]
expand_with_graph(seed_chunks: list[RetrievedChunk], chunks: list[Chunk], hops: int = 1) -> list[RetrievedChunk]
rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RankedChunk]
answer_question(query: str, evidence: list[RankedChunk]) -> dict
generate_quiz(chunks: list[Chunk], num_questions: int = 3) -> list[QuizItem]
update_feedback(concept: str, correct: bool) -> None
recommend_concept() -> dict
```

Shared schemas live in `src/schemas.py`.

