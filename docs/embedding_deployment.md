# 本地中文 Embedding 部署说明

## 目标

CourseMind 支持四种 embedding 状态：

```text
mock                 测试用哈希向量，不理解语义
lite                 轻量本地词/字 n-gram 向量，能做关键词相似
sentence_transformers 基座中文 Transformer embedding
finetuned             使用课程检索训练对微调后的本地 embedding
```

## 安装依赖

普通功能：

```powershell
pip install -r requirements.txt
```

本地 Transformer embedding 和 fine-tuning：

```powershell
pip install -r requirements-transformer.txt
```

本机验证过的稳定组合：

```text
numpy==1.26.4
torch==2.3.1+cpu
transformers==4.44.2
sentence-transformers==3.0.1
datasets==2.20.0
accelerate==0.34.2
```

不要使用 `torch 2.12.0` 和 `transformers 5.x` 这组版本；本机测试时出现过 DLL 初始化和 Trainer 兼容问题。

## 使用基座模型重建 FAISS

```powershell
$env:COURSEMIND_MODE="real"
$env:EMBEDDING_PROVIDER="sentence_transformers"
$env:EMBEDDING_MODEL_PATH="BAAI/bge-small-zh-v1.5"
python scripts/ingest.py
```

验证结果应显示：

```text
Indexed 9 pages into 189 chunks
FAISS index: data\indexes\faiss.index
```

`data/indexes/faiss.meta.json` 中应该包含：

```json
{
  "embedding": {
    "mode": "real",
    "provider": "sentence_transformers",
    "model_path": "BAAI/bge-small-zh-v1.5"
  }
}
```

## 使用微调模型重建 FAISS

先训练或生成本地模型目录：

```powershell
python scripts/build_training_pairs.py --target-count 200 --negatives-per-query 20
python scripts/train_embedding.py --max-samples 16 --epochs 1 --batch-size 4 --warmup-steps 1
```

再用微调模型入库：

```powershell
$env:COURSEMIND_MODE="real"
$env:EMBEDDING_PROVIDER="sentence_transformers"
$env:EMBEDDING_MODEL_PATH="models/embedding/finetuned"
python scripts/ingest.py
```

本机验证后，FAISS 元数据维度为 512：

```json
{
  "dim": 512,
  "embedding": {
    "mode": "real",
    "provider": "sentence_transformers",
    "model_path": "models/embedding/finetuned"
  }
}
```

## 模型目录

本地微调模型默认输出到：

```text
models/embedding/finetuned/
```

模型权重较大，默认不提交到 GitHub。仓库提交训练脚本、训练样本和报告，模型目录由负责人本地生成。
