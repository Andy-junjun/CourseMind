# Transformer 与注意力机制

## 自注意力机制（Self-Attention）
- Query、Key、Value 三个矩阵
- 通过计算 Query 和 Key 的相似度得到权重
- 用权重对 Value 加权求和

## 多头注意力（Multi-Head Attention）
- 多个注意力头并行计算
- 每个头关注不同位置的信息
- 最后拼接起来

## 位置编码（Positional Encoding）
- 补充序列的顺序信息
- 使用正弦余弦函数

## 应用
- BERT、GPT、机器翻译、文本生成