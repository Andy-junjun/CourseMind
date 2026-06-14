# Embedding 训练数据

`embedding_pairs.jsonl` 用于 sentence-transformers 风格的 embedding 微调。

每行格式：

```json
{"query":"问题","positive_chunk_id":"...","positive_text":"...","negative_chunk_id":"...","negative_text":"..."}
```

生成命令：

```powershell
python scripts/build_training_pairs.py --target-count 200
```

## MiniRanker

`miniranker.pt` is a small PyTorch MLP used by `src/miniranker.py` in `real` mode.
It reranks retrieved and GraphRAG-expanded chunks from six features:

```text
dense_score
bm25_score
graph_score
same_page_bonus
same_chapter_bonus
chunk_length_norm
```

Regenerate it after changing query labels, chunking, or scoring features:

```powershell
python scripts/train_miniranker.py --epochs 260 --max-examples 3000
```
