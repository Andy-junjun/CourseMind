# Architecture

```text
PDF / text
  -> document_loader.load_pdf
  -> chunker.chunk_pages
  -> retriever.retrieve
  -> graph_store.expand_with_graph
  -> miniranker.rerank
  -> generator.answer_question / summarize_document / study_tools.generate_quiz
  -> Streamlit app.py
```

Shared identity key: `chunk_id`.

The vector store finds semantically similar chunks, GraphRAG-lite expands context, MiniRanker selects evidence, and the generator creates user-facing answers with citations.

