# 刘书洋数据集说明

## 当前可入库资料

这些文件会被 `python scripts/ingest.py` 自动读取：

```text
data/raw/liushuyang/Actor-Critic_刘书洋.pdf
data/raw/liushuyang/策略梯度_刘书洋.pdf
data/raw/liushuyang/模型参数_刘书洋.pdf
data/raw/liushuyang/实验结果分析方法_刘书洋.pdf
```

## 暂不能直接入库

```text
data/raw/liushuyang/模型训练方法_刘书洋.pptx
```

当前解析链路只支持 PDF、TXT、MD。PPTX 需要先转换为 PDF，或整理成 Markdown/TXT 后再进入主索引。

## 评测问题

```text
data/raw/liushuyang/questions.csv          当前可用于检索评测的问题
data/raw/liushuyang/questions_pending.csv  指向 PPTX 的待处理问题
data/raw/liushuyang/metadata.csv           文件来源、主题和提交状态
```
