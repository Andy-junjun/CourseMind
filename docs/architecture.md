# 系统架构

主流程如下：

```text
中文 PDF / 文本
  -> document_loader.load_pdf
  -> chunker.chunk_pages
  -> retriever.retrieve
  -> graph_store.expand_with_graph
  -> miniranker.rerank
  -> generator.answer_question / summarize_document / study_tools.generate_quiz
  -> Streamlit app.py
```

全系统统一主键：

```text
chunk_id
```

模块职责：

```text
文档解析：把 PDF、txt、md 转成 DocumentPage。
Chunk 切分：把页面文本切成稳定 Chunk，并抽取中文知识点。
检索模块：结合 dense_score 和 bm25_score 找候选片段。
GraphRAG-lite：根据同页、同知识点等关系扩展上下文。
MiniRanker：对候选证据重新排序，输出 ranker_score。
生成模块：基于证据生成答案、总结或题目，并保留引用来源。
Bandit：根据答题反馈推荐需要复习的知识点。
```

