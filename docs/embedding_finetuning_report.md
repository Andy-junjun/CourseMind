# Embedding Fine-tuning 报告

## 当前结论

X3 和 X4 已经形成可运行闭环：

```text
检索评测问题 / 组员 questions.csv
-> data/training/embedding_pairs.jsonl
-> scripts/train_embedding.py
-> models/embedding/finetuned/
-> EMBEDDING_MODEL_PATH=models/embedding/finetuned
-> python scripts/ingest.py
-> data/indexes/faiss.index
```

## 已实现内容

代码改动：

```text
src/embedder.py                 支持 EMBEDDING_MODEL_PATH
src/vector_store.py             FAISS 元数据记录 embedding provider 和模型路径
scripts/build_training_pairs.py 构造 query-positive-negative 训练对
scripts/train_embedding.py      支持 dry-run 和真实 TripletLoss 微调
```

训练数据：

```text
data/training/embedding_pairs.jsonl
```

生成命令：

```powershell
python scripts/build_training_pairs.py --target-count 200 --negatives-per-query 20
```

生成结果：

```text
Built 200 embedding training pairs
```

训练对构造规则：

```text
1. 读取 data/eval/retrieval_queries*.csv
2. 读取 data/raw/*/questions.csv
3. 跳过 out_of_scope 问题
4. expected_file 匹配不到时不构造正样本
5. 如果问题来自 data/raw/xujiaze/questions.csv，正负样本都限制在 xujiaze/ 资料目录内
```

这个规则是为了避免把本地未提交资料或项目说明 PDF 混入训练样本。

## 本机真实训练记录

依赖版本：

```text
numpy==1.26.4
torch==2.3.1+cpu
transformers==4.44.2
sentence-transformers==3.0.1
datasets==2.20.0
accelerate==0.34.2
```

基座模型：

```text
BAAI/bge-small-zh-v1.5
```

基座模型验证：

```text
输入：Transformer 注意力机制
输出维度：512
向量范数：1.0
```

小样本真实训练命令：

```powershell
python scripts/train_embedding.py --max-samples 16 --epochs 1 --batch-size 4 --warmup-steps 1
```

训练结果：

```text
train_runtime: 4.3681
train_samples_per_second: 3.663
train_steps_per_second: 0.916
train_loss: 4.769443988800049
epoch: 1.0
```

输出目录：

```text
models/embedding/finetuned/
```

该目录包含 `model.safetensors`、`tokenizer.json`、`modules.json`、`training_manifest.json` 等文件，可被 sentence-transformers 加载。

## 微调模型 FAISS 入库验证

命令：

```powershell
$env:COURSEMIND_MODE="real"
$env:EMBEDDING_PROVIDER="sentence_transformers"
$env:EMBEDDING_MODEL_PATH="models/embedding/finetuned"
python scripts/ingest.py
```

结果：

```text
Indexed 9 pages into 189 chunks
Chunks: data\processed\chunks.jsonl
FAISS index: data\indexes\faiss.index
```

FAISS 元数据：

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

## 风险与下一步

当前训练只是 16 条样本的小规模验证，作用是证明链路能跑通，不代表模型效果已经明显提升。

下一步应该做：

```text
1. 增加与当前 raw 数据匹配的 retrieval_queries/questions
2. 用完整 200 条训练对训练 1-3 个 epoch
3. 实现 evaluate_retrieval.py，比较 lite、base transformer、fine-tuned transformer 的 Recall@5
4. 把训练耗时、Recall@5、响应时间整理进 docs/experiment_report.md
```
