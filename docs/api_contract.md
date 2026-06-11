# API 契约

以下函数签名是第一轮冲刺的强约束。组员可以替换函数内部实现，但不要私自修改参数和返回值。

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

共享数据结构定义在 `src/schemas.py`。如果必须修改契约，先同步所有成员，再同时更新代码、测试和文档。

