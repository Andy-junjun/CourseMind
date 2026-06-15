# CourseMind

CourseMind 是一个面向中文深度学习课程资料的智能学习助手。系统把 PDF、Markdown、文本资料解析成中文 chunk，建立 FAISS 向量索引，并在 Streamlit 页面中提供问答、引用证据、GraphRAG 可视化、概念理解练习和 Bandit 复习推荐。

当前项目适合课程汇报演示：既能展示真实检索链路，也能说明深度学习模型在中文 embedding、语义检索、重排序和生成式回答中的作用。

## 功能概览

- 资料解析：支持 `.pdf`、`.txt`、`.md`、`.markdown`，PDF 读取文本层。
- 中文 chunk：按页、章节和重叠窗口切分，自动抽取概念词。
- 语义检索：FAISS 向量库 + 中文 embedding，辅以 BM25 关键词分数。
- GraphRAG-lite：基于同页、同章节、相邻 chunk、共享概念扩展证据。
- MiniRanker：对候选证据进行重排序，优先保留更适合回答的问题片段。
- DeepSeek 回答：可选接入 DeepSeek API，用检索证据生成长文本回答。
- 概念练习：围绕知识库概念生成理解型选择题，而不是让用户判断证据片段。
- Bandit 推荐：根据错题率、练习次数和 UCB 探索分数推荐下一轮练习知识点。

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

把课程资料放到 `data/raw/` 下。建议每位成员建立自己的子目录，例如：

```text
data/raw/xujiaze/
data/raw/liushuyang/
data/raw/shiyan11-15/
```

每个数据目录建议包含：

```text
*.pdf / *.md / *.txt        课程资料正文
metadata.csv                数据来源、主题、贡献人、可信度说明
questions.csv               人工整理的问题、答案或题目
```

重新构建知识库：

```bash
python scripts/ingest.py
```

生成文件：

```text
data/processed/chunks.jsonl
data/indexes/faiss.index
data/indexes/faiss.meta.json
```

这些是本地生成物，默认不作为主要协作文件。页面启动时会优先读取 `data/indexes/faiss.index`；如果索引不存在，才使用内置示例。

## 运行模式

```text
mock   : 默认模式，使用确定性 fallback，适合集成测试
hybrid : 优先使用真实模块，失败时 fallback
real   : 使用真实 embedding、FAISS、MiniRanker、LLM API 等能力
```

示例 `.env`：

```text
COURSEMIND_MODE=real
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_PATH=models/embedding/finetuned
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API Key
BANDIT_STATE_PATH=data/processed/quiz_state.json
```

轻量模式可以使用：

```text
COURSEMIND_MODE=real
EMBEDDING_PROVIDER=lite
LLM_PROVIDER=mock
```

`lite` 不等同于 transformer 语义模型，适合低配置环境演示；正式汇报建议使用 `sentence_transformers` 或微调后的中文 embedding。

## DeepSeek 回答生成

检索、GraphRAG 和 MiniRanker 会先从本地 FAISS 索引中取证据；DeepSeek 只负责根据证据生成长文本回答。

```text
COURSEMIND_MODE=real
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API Key
LLM_MODEL=deepseek-v4-flash
LLM_API_BASE=https://api.deepseek.com/chat/completions
LLM_TIMEOUT=30
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1800
ANSWER_MAX_CHARS_PER_CHUNK=1200
```

修改 `.env` 后需要重启 Streamlit。

如果 API 网络不可用或 DeepSeek 临时失败，问答页不会中断检索流程；系统会显示提示，并先返回基于本地证据片段的临时回答，引用来源仍然保留。

## 练习与复习推荐

页面的“练习与复习”标签页会把出题和复习推荐放在同一个闭环中：

```text
知识库概念
-> 生成概念理解选择题或关键词遮罩完形题
-> 用户提交答案
-> 记录正确 / 错误反馈
-> Bandit 更新推荐知识点
-> 下一轮围绕推荐知识点出题
```

每道题包含：

- `question`：围绕概念理解的题干，或遮住关键词的完形填空题
- `options`：选项
- `answer`：正确解释或被遮住的关键词
- `explanation`：解释
- `concept`：对应知识点
- `source_chunk_id`：来源 chunk

用户提交答案后，系统会把正确/错误反馈写入 `BANDIT_STATE_PATH`。Bandit/UCB 会推荐优先复习的知识点：

```text
推荐分数 = 错题率 + 探索项
```

因此，错得多、练得少或长期没覆盖的知识点会更容易被推荐。

## 当前检索链路

```text
data/raw 课程资料
-> PDF / Markdown / 文本解析
-> 中文 chunk 切分与概念抽取
-> 中文 embedding
-> FAISS IndexFlatIP 向量检索
-> BM25 关键词补充分数
-> GraphRAG-lite 图扩展
-> MiniRanker 重排序
-> 答案保护与 DeepSeek / mock 生成
-> 概念理解练习与 Bandit 复习推荐
```

## PDF 图片说明

当前仓库已移除图片文字识别功能。PDF 解析只读取文件自带的文本层：

```text
可复制文字的 PDF     -> 可以解析
扫描版 / 图片型 PDF -> 无法直接识别图片文字
```

如果资料是扫描版 PDF，建议先手动整理成 Markdown、txt，或使用外部工具转成带文本层的 PDF 后再放入 `data/raw/`。

## 测试

```bash
python -m pytest
```

测试覆盖文档解析、中文切分、FAISS 持久化、检索、GraphRAG-lite、MiniRanker、LLM mock、概念练习题生成和 Bandit 复习推荐。


## 技术文档

- 系统架构与数据流：[docs/architecture.md](docs/architecture.md)
- 中文检索设计：[docs/chinese_retrieval.md](docs/chinese_retrieval.md)
- GraphRAG-lite 设计：[docs/graphrag_design.md](docs/graphrag_design.md)
- Embedding 微调实验报告：[docs/embedding_finetuning_report.md](docs/embedding_finetuning_report.md)

