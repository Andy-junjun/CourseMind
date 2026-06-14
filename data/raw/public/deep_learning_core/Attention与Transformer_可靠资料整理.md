# Attention 与 Transformer

来源依据：Attention Is All You Need 论文、The Annotated Transformer、Hugging Face Transformers 文档。本文为中文改写整理，不复制原文。

参考链接：

- https://arxiv.org/abs/1706.03762
- https://nlp.seas.harvard.edu/annotated-transformer/
- https://huggingface.co/docs/transformers/index

## 1. 注意力机制的直觉

注意力机制让模型在处理某个位置时，根据相关性动态关注其他位置。与固定窗口卷积或顺序递归不同，自注意力可以直接连接序列中任意两个 token。

在缩放点积注意力中，每个 token 会产生 Query、Key、Value 三组向量：

```text
Query：当前要查询什么
Key：每个位置提供什么索引特征
Value：每个位置真正传递的信息
```

Query 与 Key 做相似度计算，再经过 softmax 得到权重，用这些权重对 Value 加权求和。

## 2. 为什么要缩放

点积注意力在维度较大时，Query 和 Key 的点积数值可能变大，softmax 容易进入饱和区，使梯度变小。Transformer 使用 `sqrt(d_k)` 进行缩放，让注意力分布更稳定。

缩放点积注意力可以写成：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

这个公式是 Transformer 的核心计算之一。

## 3. 多头注意力

单个注意力头只能在一个表示子空间中计算关系。多头注意力把表示拆成多个头，每个头学习不同关系，再把结果拼接。

多头注意力的意义：

```text
不同头可以关注不同位置关系
可以同时建模语法、语义、长距离依赖等模式
提高表示能力
```

需要注意的是，并不是每个头都一定有清晰可解释语义，但多头结构整体增强了模型表达能力。

## 4. 位置编码

自注意力本身不包含顺序信息。如果打乱输入 token 的位置，只看注意力计算，模型无法知道原始顺序。因此 Transformer 需要位置编码，把位置信息加入 token 表示。

位置编码可以是固定正弦余弦形式，也可以是可学习的位置向量。现代模型还会使用相对位置编码、旋转位置编码等方式改进长文本建模。

## 5. Encoder 与 Decoder

Transformer Encoder 适合理解输入序列，常用于分类、检索、embedding、序列标注等任务。Decoder 适合自回归生成，常用于语言模型和文本生成。Encoder-Decoder 结构适合翻译、摘要等输入到输出的序列转换任务。

BERT 类模型偏理解，GPT 类模型偏生成。CourseMind 的 embedding 检索更接近 Encoder 表示学习，而 DeepSeek 回答生成更接近 Decoder 语言模型能力。

## 6. 对 CourseMind 的意义

CourseMind 可以把 Transformer 讲成系统深度学习核心：

```text
用 Transformer embedding 把问题和 chunk 映射到向量空间
用 FAISS 在向量空间找语义相近片段
用 GraphRAG 扩展证据
用大语言模型基于证据生成回答
```

这样可以说明系统不是简单搜索，而是利用深度表示学习完成中文课程知识检索。
