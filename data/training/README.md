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
