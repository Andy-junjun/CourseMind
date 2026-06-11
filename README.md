# CourseMind

CourseMind 是一个面向中文课程资料的智能学习助手项目骨架。第一版目标不是把所有模型都做满，而是先统一目录结构、数据结构、函数接口和演示链路，让组员可以在同一个骨架里并行补真实实现。

当前仓库默认可在 `mock` 模式运行，不需要 API Key、FAISS 索引或训练好的模型。

## 功能入口

```text
中文 PDF / 文本解析
中文 Chunk 切分
中文关键词检索 / BM25 fallback
向量检索接口
GraphRAG-lite 图扩展接口
MiniRanker 重排序接口
带引用的问答
文档总结
自动出题
答题反馈
Bandit 复习推荐
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

如果需要指定本地地址：

```bash
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

## 运行模式

```text
mock   : 默认模式，使用确定性的 fallback，适合集成和演示
hybrid : 有真实模块时优先使用真实模块，否则 fallback
real   : 预期接入真实 API、embedding 模型、索引和 reranker
```

设置模式：

```bash
set COURSEMIND_MODE=mock
```

## 中文检索说明

中文检索和英文检索的主要差异在于分词。当前骨架已做两层处理：

```text
1. 安装 jieba 时，使用 jieba 做中文分词。
2. 没有 jieba 时，退化为中文单字 + bigram fallback。
```

后续 B 组可以在 `src/bm25_store.py` 内替换为更正式的 BM25；也可以在 `src/embedder.py` 和 `src/vector_store.py` 内接入中文 embedding 模型，例如 bge、m3e 或其他课程允许使用的中文向量模型。不要改公共函数签名。

## 小组边界

```text
A 组：集成、Streamlit UI、README、可运行 demo
B 组：PDF 解析、中文 Chunk、embedding、FAISS/BM25、GraphRAG-lite、retrieval
C 组：LLM client、generator、MiniRanker、answer guard
D 组：Bandit 推荐、答题反馈、测试问题、PPT、演示素材
```

所有模块必须遵守 `src/schemas.py` 和 `docs/api_contract.md`。不要私自重命名共享字段或修改公共函数签名。

## 必须提交的交付物

每个成员至少提交一个仓库内可追踪交付物：代码、文档、测试问题、截图、实验结果、PPT 贡献记录或 README 更新。

不要提交 API Key、本地模型权重、私人课程资料或生成的索引文件。

