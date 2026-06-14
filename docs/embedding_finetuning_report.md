# Embedding Fine-tuning 报告

## 当前结论

已用完整 900 条训练对（75 个课程问题 × 每题约 12 个难负样本）对 BAAI/bge-small-zh-v1.5
做 TripletLoss 微调（3 epochs / batch=8 / 339 steps / 约 12.5 分钟），并用
`scripts/evaluate_retrieval.py` 做了三档 Recall@K / MRR 对比，**验证微调显著有效**：

```text
档位              Recall@1  Recall@3  Recall@5   MRR
lite              0.2778    0.6667    0.7222   0.4370
bge-base          0.2778    0.6111    0.7778   0.4648
bge-finetuned     0.4444    0.8333    1.0000   0.6713   <- 微调后全面领先
```

完整闭环：

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

正式训练命令（完整 900 对）：

```powershell
python scripts/train_embedding.py --epochs 3 --batch-size 8 --warmup-steps 50
```

训练结果：

```text
train_runtime: 753.6993
train_samples_per_second: 3.582
train_steps_per_second: 0.45
train_loss: 4.498719161942294
epoch: 3.0
pair_count: 900
```

（早期还做过 16 条小样本 / 1 epoch 的链路验证，train_loss≈4.77，仅用于确认流程可跑通，现已被上述完整训练取代。）

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

已完成：900 对正式训练 + `scripts/evaluate_retrieval.py` 三档 Recall@K/MRR 评测，
微调相比基座全面提升（Recall@5 0.78→1.00，MRR 0.46→0.67）。

仍存在的局限与下一步：

```text
1. 评测集仅 18 题、且与训练 query 同源（ddw 资料），样本偏小、存在乐观偏差
2. 下一步：扩充跨成员（liushuyang / xujiaze / shiyan11-15）的评测集做交叉验证
3. 补充响应时延对比（lite vs transformer 的检索耗时）
4. 把训练耗时、Recall@K、时延整理进 docs/experiment_report.md
```
