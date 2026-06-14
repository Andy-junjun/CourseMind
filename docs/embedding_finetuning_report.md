# Embedding Fine-tuning 报告

## 当前结论（已纠正数据泄漏）

> 重要更正：早期一版结论曾报告"微调后 Recall@5 从 0.78→1.00、显著有效"。后来发现
> 那次评测的测试问题**参与了训练**（train/test 同源），属于数据泄漏，结论无效。
> 本节是修正后的结论。

我们用 `scripts/split_dataset.py` 把全部 197 个课程问题按 query 维度做 **80/20 不重叠
划分**（种子=42），只用 158 个训练问题构造 1896 条 TripletLoss 训练对微调
BAAI/bge-small-zh-v1.5（3 epochs / batch=8 / 约 32 分钟），再用 `scripts/evaluate_retrieval.py`
在 **39 个未参与训练的测试问题**上评测三档：

```text
档位              Recall@1  Recall@3  Recall@5   MRR
lite              0.6667    0.9231    0.9487   0.7885
bge-base          0.7436    0.9744    1.0000   0.8470   <- 原始基座最强
bge-finetuned     0.5897    0.7949    0.8718   0.7030   <- 小规模微调反而退化
```

**结论**：在独立测试集上，**原始 BGE 检索效果最好**，小规模微调因弱负样本（同主题正确
片段被当负例）与小数据过拟合而退化。对已在大规模中文语料上训练过的强基座，
**直接使用比小规模 TripletLoss 微调更稳妥**。

完整、防泄漏的实验闭环：

```text
所有 questions.csv / retrieval_queries
-> scripts/split_dataset.py  (80/20 划分, 种子固定)
-> data/eval/train_queries.csv  +  data/eval/test_queries.csv
-> scripts/build_training_pairs.py --queries-file train_queries.csv (仅训练集)
-> scripts/train_embedding.py -> models/embedding/finetuned/
-> scripts/evaluate_retrieval.py (仅 test_queries.csv 评测)
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

正式训练命令（仅训练集 158 题 → 1896 对，防泄漏）：

```powershell
python scripts/split_dataset.py --test-ratio 0.2 --seed 42
python scripts/build_training_pairs.py --queries-file data/eval/train_queries.csv --target-count 2000
python scripts/train_embedding.py --epochs 3 --batch-size 8 --warmup-steps 80
```

训练结果：

```text
train_runtime: 1919.4861
train_samples_per_second: 2.963
train_steps_per_second: 0.37
train_loss: 4.4834886596508
epoch: 3.0
pair_count: 1896
```

（早期做过 16 条小样本验证、以及一次 900 对但 train/test 同源的训练——后者评测结论因
数据泄漏作废，现已被上述防泄漏流程取代。）

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

已完成：197 题数据集、80/20 防泄漏划分、1896 对训练、独立测试集三档评测。
关键发现：**小规模微调在独立测试集上不及原始基座**（见"当前结论"）。

下一步（让微调可能真正生效的方向）：

```text
1. 换损失：MultipleNegativesRankingLoss（in-batch negatives），比 TripletLoss 更适合检索
2. 清洗负样本：剔除"同主题正确片段"被当作负例的情况（当前 hard negative 偏脏）
3. 降低过拟合：减到 1 epoch、调小学习率、加大数据量到数千条
4. 扩大评测：跨成员交叉验证 + 补充检索响应时延对比
5. 把以上整理进 docs/experiment_report.md
```
