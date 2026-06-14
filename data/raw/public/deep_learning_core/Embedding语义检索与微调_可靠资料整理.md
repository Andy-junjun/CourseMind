# Embedding、语义检索与微调

来源依据：Hugging Face Transformers 文档、Sentence-Transformers 文档、FAISS 文档。本文为中文改写整理，不复制原文。

参考链接：

- https://huggingface.co/docs/transformers/index
- https://www.sbert.net/
- https://faiss.ai/

## 1. Embedding 是什么

Embedding 是把离散文本映射到连续向量空间的表示。对于 CourseMind 来说，用户问题和课程 chunk 都会被编码成向量。如果两个文本语义相近，它们在向量空间中的距离应该更近。

相比关键词匹配，embedding 可以处理表达方式不同但含义相近的问题。例如“LSTM 如何解决长期依赖”和“LSTM 为什么不容易梯度消失”关键词不同，但语义相关。

## 2. 向量相似度

语义检索通常使用余弦相似度或内积衡量向量接近程度。如果向量已经归一化，内积和余弦相似度等价。CourseMind 使用 FAISS 存储向量并进行近邻搜索，适合在本地快速检索大量 chunk。

检索流程：

```text
用户问题 -> embedding 模型 -> query 向量
课程 chunk -> embedding 模型 -> chunk 向量
FAISS 搜索 query 最近邻
返回 Top-K 候选 chunk
```

## 3. 为什么需要中文模型

中文文本没有天然空格分词，表达方式和英文不同。使用中文或多语种预训练模型通常比英文模型更适合中文课程资料。BGE、m3e、text2vec 等中文 embedding 模型常用于中文语义检索。

如果使用过弱的 mock embedding，模型并不真正理解语义，只是基于哈希或词法特征生成向量，检索效果有限。真实 transformer embedding 才能体现深度学习表示能力。

## 4. 微调的目标

Embedding 微调通常需要构造正负样本：

```text
query：用户问题
positive：应命中的课程 chunk
negative：不应命中的 chunk
```

训练目标会拉近 query 与 positive 的向量距离，拉远 query 与 negative 的距离。这样模型会更适应当前课程资料和问题风格。

## 5. 小数据微调的风险

数据少时，微调可能过拟合。模型可能只记住少数问题的关键词，而不是学到稳定语义关系。因此需要：

```text
高质量 questions.csv
覆盖多个主题
保留验证集
使用较小学习率
控制 epoch
比较微调前后的 Recall@5
```

如果微调后训练问题效果变好，但新问题效果变差，就是过拟合信号。

## 6. 对 CourseMind 的意义

CourseMind 的训练数据应优先服务两个目标：

```text
让系统能检索到正确资料
让汇报能证明深度学习模块有效
```

因此每份资料必须配套问题。只有 raw 文件没有 questions.csv，无法构造 query-positive-negative 训练 pair，也无法评估检索效果。
