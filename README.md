# CourseMind

CourseMind 是一个面向中文课程资料的智能学习助手骨架，支持 PDF/文本解析、中文 Chunk 切分、FAISS 向量检索、BM25 关键词检索、GraphRAG-lite 扩展、MiniRanker 重排序、带引用问答、总结、出题和复习推荐。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/ingest.py
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

打开页面：

```text
http://127.0.0.1:8501/
```

## 数据导入

把课程资料放到 `data/raw/`，支持：

```text
.pdf
.txt
.md
.markdown
```

然后运行：

```bash
python scripts/ingest.py
```

导入后会生成：

```text
data/processed/chunks.jsonl
data/indexes/faiss.index
data/indexes/faiss.meta.json
```

这些文件是本地生成物，默认不提交到 Git。页面启动时会优先读取 `data/indexes/faiss.index`；如果索引不存在，才使用内置示例。

## 运行模式

```text
mock   : 默认模式，使用确定性的 fallback，适合集成和演示
hybrid : 优先使用真实模块，失败时 fallback
real   : 接入真实 embedding 模型、LLM API 等外部能力
```

设置方式：

```bash
set COURSEMIND_MODE=mock
```

## 轻量 Embedding

默认推荐先用内置轻量 embedding：

```text
COURSEMIND_MODE=real
EMBEDDING_PROVIDER=lite
EMBEDDING_DIM=384
LLM_PROVIDER=mock
```

`lite` provider 不依赖 Torch 或 ONNXRuntime，使用中文分词、字符 n-gram 和知识点 alias 生成归一化向量，再写入 FAISS。它比纯 mock hash 更适合本地中文资料检索，但仍属于轻量词法向量，不等同于 transformer 语义模型。

重新构建索引：

```bash
python scripts/ingest.py
```

如果机器支持 PyTorch 或 ONNXRuntime，也可以切换：

```text
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
```

或：

```text
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
```

## 当前检索链路

```text
data/raw 文档
-> PDF/文本解析
-> 中文 Chunk 切分与知识点抽取
-> embedding
-> FAISS IndexFlatIP 向量索引
-> BM25 关键词补充分数
-> GraphRAG-lite 扩展
-> MiniRanker 重排序
-> 答案保护与生成
```

## 测试

```bash
python -m pytest
```

当前测试覆盖文档解析、中文切分、FAISS 持久化索引、检索、GraphRAG-lite、MiniRanker、LLM mock 和答案保护。
