# Embedding 模型目录

本目录用于放本地 embedding 模型。

推荐结构：

```text
models/embedding/finetuned/
```

模型权重通常较大，默认不提交到 Git。X4 任务需要提交训练脚本、训练样本和报告；实际模型目录在本地生成后，通过 `.env` 或环境变量加载：

```powershell
$env:COURSEMIND_MODE="real"
$env:EMBEDDING_PROVIDER="sentence_transformers"
$env:EMBEDDING_MODEL_PATH="models/embedding/finetuned"
python scripts/ingest.py
```
