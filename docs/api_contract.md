# API 契约

以下函数签名是当前项目内部协作的主要契约。组员可以替换函数内部实现，但不要私自修改参数、返回值或共享数据结构字段。

## 数据结构

共享数据结构定义在 `src/schemas.py`：

```python
DocumentPage(file_name: str, page: int, text: str)
Chunk(chunk_id: str, file_name: str, page: int, text: str, chapter: str | None, concepts: list[str])
RetrievedChunk(chunk: Chunk, dense_score: float, bm25_score: float, graph_score: float)
RankedChunk(chunk: Chunk, dense_score: float, bm25_score: float, graph_score: float, ranker_score: float)
QuizItem(question: str, options: list[str], answer: str, explanation: str, concept: str, source_chunk_id: str)
```

## 核心流程接口

```python
load_pdf(file_path: str) -> list[DocumentPage]
chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[Chunk]
retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
    index: VectorIndex | None = None,
) -> list[RetrievedChunk]
expand_with_graph(
    seed_chunks: list[RetrievedChunk],
    chunks: list[Chunk],
    hops: int = 1,
) -> list[RetrievedChunk]
rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RankedChunk]
answer_question(
    query: str,
    evidence: list[RankedChunk],
) -> dict
generate_quiz(
    chunks: list[Chunk],
    num_questions: int = 3,
    target_concept: str | None = None,
) -> list[QuizItem]
record_quiz_result(
    quiz_item: QuizItem,
    is_wrong: bool,
) -> None
recommend_concept(
    chunk_lookup: dict[str, Chunk] | None = None,
) -> dict
```

## 索引接口

```python
build_index(chunks: list[Chunk]) -> VectorIndex
search_index(query: str, chunks: list[Chunk], top_k: int = 5, index: VectorIndex | None = None) -> list[tuple[Chunk, float]]
persist_vector_store(chunks: list[Chunk], chunks_path=DEFAULT_CHUNKS_PATH, index_path=DEFAULT_INDEX_PATH, metadata=None) -> VectorIndex
load_vector_store(chunks_path=DEFAULT_CHUNKS_PATH, index_path=DEFAULT_INDEX_PATH) -> tuple[list[Chunk], VectorIndex]
```

## 修改规则

1. 修改接口签名前，先同步调用方、测试和文档。
2. `chunk_id` 是全系统主键，不能在中间流程重新生成。
3. `mock` 模式必须保持可运行，方便没有模型和 API key 的成员测试。
4. 新增字段时优先向后兼容，不要让旧数据无法读取。
