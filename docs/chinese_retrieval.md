# 中文检索说明

CourseMind 面向中文课程资料。公共接口保持不变，但内部实现要考虑中文分词、中文术语映射和中文 embedding。

## Chunk 切分

当前 fallback：

```text
固定字符窗口切分
chunk_size = 400
overlap = 80
```

推荐真实实现：

```text
1. 先按页切分。
2. 能识别标题时，优先按中文标题和章节切分。
3. 尽量保持公式、代码块和项目符号列表完整。
4. 保持 page 和 chunk_id 稳定，避免索引和引用失效。
```

不要删除这些字段：

```text
chunk_id
file_name
page
text
chapter
concepts
```

## 关键词检索

当前 `src/bm25_store.py` fallback：

```text
安装 jieba 时：使用 jieba 中文分词。
没有 jieba 时：使用中文单字 + bigram fallback。
```

推荐真实实现：

```text
rank-bm25 + jieba
课程术语词表加权
同义词映射
```

术语映射示例：

```text
大语言模型 -> LLM
检索增强生成 -> RAG
重排序 -> rerank / MiniRanker
多臂老虎机 -> Bandit
知识图谱扩展 -> GraphRAG-lite
```

## Embedding

推荐可选中文向量模型：

```text
bge-small-zh / bge-base-zh
bge-m3
m3e
支持中文的 text-embedding API
```

保持公共检索接口不变：

```python
retrieve(query: str, chunks: list[Chunk], top_k: int = 5) -> list[RetrievedChunk]
```

## 重排序

MiniRanker 可以保持轻量特征模型：

```text
dense_score
bm25_score
graph_score
same_page_bonus
same_chapter_bonus
chunk_length_norm
```

如果后续加入中文 Cross-Encoder，也应封装在 `rerank(...)` 接口后面，不要破坏 `RankedChunk` 数据结构。

