# Chinese Retrieval Notes

CourseMind targets Chinese course materials. The public interfaces stay the same, but several internal modules should be implemented with Chinese retrieval behavior in mind.

## Chunking

Current fallback:

```text
fixed character window
chunk_size = 400
overlap = 80
```

Recommended real implementation:

```text
1. split by page
2. split by Chinese headings when detectable
3. keep formulas, code blocks, and bullet lists together when possible
4. keep page number and chunk_id stable
```

Do not remove these fields:

```text
chunk_id
file_name
page
text
chapter
concepts
```

## Keyword Retrieval

Current fallback in `src/bm25_store.py`:

```text
jieba tokenization when installed
Chinese character + bigram fallback when jieba is unavailable
```

Recommended real implementation:

```text
rank-bm25 + jieba
domain glossary boosting
synonym mapping for course terms
```

Examples:

```text
大语言模型 -> LLM
检索增强生成 -> RAG
重排序 -> rerank / MiniRanker
多臂老虎机 -> Bandit
```

## Embedding

Recommended Chinese-capable embedding models:

```text
bge-small-zh / bge-base-zh
bge-m3
m3e
text-embedding API with Chinese support
```

Keep the public retrieval interface:

```python
retrieve(query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievedChunk]
```

## Reranking

MiniRanker can stay as the lightweight feature model:

```text
dense_score
bm25_score
graph_score
same_page_bonus
same_chapter_bonus
chunk_length_norm
```

If a Chinese cross-encoder is added later, keep it behind the rerank interface and do not break `RankedChunk`.

