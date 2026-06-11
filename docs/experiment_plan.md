# 实验计划

至少准备 20 个中文测试问题：

```text
8 个事实问答问题
5 个总结类问题
4 个自动出题任务
3 个资料外问题
```

对比方法：

```text
直接问 LLM
FAISS / RAG
FAISS + BM25
FAISS + BM25 + GraphRAG-lite
GraphRAG-lite + MiniRanker
```

评估指标：

```text
Recall@5
答案正确率
引用准确率
拒答准确率
平均响应时间
```

记录建议：

```text
每个问题保留 query、期望依据 chunk_id、系统答案、引用页码、是否答对、是否拒答正确、耗时。
实验结果只写真实跑出来的数据；如果是 mock 或 fallback，必须明确标注。
```

