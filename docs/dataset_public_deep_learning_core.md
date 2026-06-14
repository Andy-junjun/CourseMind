# 公开可靠深度学习核心数据集

## 目标

这个数据集用于补足 CourseMind 的核心知识覆盖和训练样本数量。内容不是复制网页原文，而是基于可靠公开资料重新组织的中文课程笔记，适合提交到 GitHub、进入本地索引、生成检索训练样本。

## 存放位置

```text
data/raw/public/deep_learning_core/
```

包含：

```text
8 个 Markdown 资料文件
1 个 metadata.csv
1 个 questions.csv
```

## 覆盖主题

```text
神经网络基础
反向传播
自动微分
优化器
训练稳定性
正则化
CNN
RNN / LSTM / GRU
Attention / Transformer
Embedding 语义检索
训练评估与消融实验
```

## 来源原则

优先使用官方教材、课程、论文和主流框架文档：

```text
Deep Learning Book
MIT 6.S191 Introduction to Deep Learning
Stanford CS231n
PyTorch Tutorials
Attention Is All You Need
Adam paper
Dropout paper
Batch Normalization paper
Hugging Face Transformers
Sentence-Transformers
FAISS
```

每个 Markdown 文件开头都列出参考链接。正文为中文改写整理，不直接复制原文。

## 评测与训练

`questions.csv` 当前包含 42 条问题：

```text
fact
compare
summary
out_of_scope
```

这些问题会被 `scripts/build_training_pairs.py` 自动读取，用于生成 embedding 微调所需的 query-positive-negative 样本。

推荐流程：

```powershell
python scripts/ingest.py
python scripts/build_training_pairs.py --target-count 600
```

## 汇报时怎么讲

可以说明：

```text
我们发现组员提交的数据较散，无法稳定训练和评测。
因此补充了一个公开可靠来源改写的数据集。
每份资料都有来源链接，每个主题都有可评测问题。
这些问题既用于检索评测，也用于构造 embedding 微调样本。
```
