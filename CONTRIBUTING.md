# 贡献规范

本文档用于统一 CourseMind 小组协作方式。目标不是把流程写复杂，而是让每个人提交的代码、资料和实验记录都能被系统直接使用，也能在汇报时追溯来源。

## 基本规则

1. 不要提交 API Key、账号密码、私有模型权重或无法公开的资料。
2. 不要随意修改 `src/schemas.py` 的字段名；这些字段是解析、检索、出题和复习推荐共用的数据契约。
3. 每次提交前至少保证 `mock` 模式能运行，代码改动建议运行 `python -m pytest`。
4. 数据、实验记录、问题集、PPT 素材都要进仓库，不能只发在微信群里。
5. 新增本地生成物前先判断是否应该提交：`data/processed/`、`data/indexes/`、临时日志通常不提交。

## 资料提交规范

每位成员把资料放到独立目录：

```text
data/raw/<成员名或主题>/
```

建议结构：

```text
data/raw/<成员名或主题>/
  metadata.csv
  questions.csv
  资料1.md
  资料2.pdf
```

`metadata.csv` 建议字段：

```text
file_name,topic,source,contributor,reliability,notes
```

`questions.csv` 建议字段：

```text
question,answer,concept,source_file,page,difficulty
```

资料要求：

- 优先提交可解析文本：Markdown、txt、带文本层 PDF。
- PDF 如果主要是图片，先在 `metadata.csv` 说明，避免系统解析后没有有效 chunk。
- 文件名尽量包含主题和贡献人，例如 `模型训练_流程与常见问题_张三.md`。
- 不要只提交截图；截图中的信息很难进入检索和出题链路。

## 出题数据要求

当前页面可以基于 chunk 自动生成概念选择题，但人工整理的问题仍然很有价值。建议成员补充 `questions.csv`，用于后续评测和更高质量出题。

每道题至少包含：

- 问题
- 标准答案
- 对应知识点
- 来源文件或页码
- 难度

示例：

```csv
question,answer,concept,source_file,page,difficulty
卷积神经网络中权重共享的作用是什么？,减少参数量并增强局部特征提取能力,CNN,cha5-卷积神经网络.pdf,3,medium
```

## 代码贡献流程

1. 拉取最新代码。

```bash
git pull origin main
```

2. 安装依赖。

```bash
pip install -r requirements.txt
```

3. 修改代码或数据后运行测试。

```bash
python -m pytest
```

4. 如果改了 `data/raw/`，重新构建索引验证。

```bash
python scripts/ingest.py
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

5. 提交前检查状态。

```bash
git status
```

## 接口边界

常用数据结构在 `src/schemas.py`：

- `DocumentPage`：解析后的页
- `Chunk`：进入检索、图扩展和出题的最小知识片段
- `RetrievedChunk`：向量检索和 BM25 后的候选片段
- `RankedChunk`：MiniRanker 排序后的证据片段
- `QuizItem`：出题和 Bandit 反馈使用的题目结构

常用模块：

- `src/document_loader.py`：PDF / 文本解析
- `src/chunker.py`：chunk 切分与概念抽取
- `src/vector_store.py`：FAISS 索引构建、保存、加载
- `src/retriever.py`：dense + BM25 检索
- `src/graph_store.py`：GraphRAG-lite 扩展与解释
- `src/miniranker.py`：候选证据重排序
- `src/study_tools.py`：自动出题
- `src/bandit_recommender.py`：答题反馈和复习推荐

## 不要提交的内容

```text
.env
data/processed/chunks.jsonl
data/indexes/faiss.index
data/indexes/faiss.meta.json
data/processed/quiz_state.json
logs/
__pycache__/
.pytest_cache/
```

如果确实需要提交模型、索引或大型数据，先在群里说明原因，并确认文件大小和许可证。
