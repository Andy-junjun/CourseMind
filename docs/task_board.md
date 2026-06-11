# 任务板

本文件是 3 天冲刺的任务分配基准。不要只在微信群里口头分工，所有任务都要能在仓库中追踪到交付物。

每个成员至少负责一个可验收交付物：代码、文档、测试问题、截图、实验记录、PPT 贡献说明或 README 更新。

## A1 项目初始化与运行环境

负责人：梁浩均
协作者：

输入：
- 项目骨架
- `.env.example`

输出：
- `README.md`
- `requirements.txt`
- 可验证的启动命令

涉及文件：
- `README.md`
- `requirements.txt`
- `.env.example`

验收标准：
- 新成员 clone 仓库后能运行 `streamlit run app.py`。
- README 说明 `mock`、`hybrid`、`real` 三种模式。
- README 明确说明 API Key 不能提交到仓库。

## A2 Streamlit 主界面

负责人：
协作者：

输入：
- `src/` 下各模块公开函数

输出：
- 完整可见的应用页面

涉及文件：
- `app.py`

验收标准：
- 页面包含智能问答、文档总结、自动出题、答题反馈、复习推荐、检索可视化和系统状态。
- 应用在 `mock` 模式下不依赖外部服务即可启动。
- 侧边栏能显示运行模式、页数和 chunk 数量。

## A3 端到端演示流程

负责人：
协作者：

输入：
- `demo/demo_questions.md`
- 各模块公开函数

输出：
- 一条稳定的现场演示路径

涉及文件：
- `app.py`
- `demo/demo_questions.md`

验收标准：
- 演示能覆盖上传/构建知识库、问答、引用、出题、故意答错和复习推荐。
- 没有上传 PDF 时，也能使用内置中文样例文本演示。
- 演示问题写入 `demo/demo_questions.md`。

## A4 集成守护与 Bug 修复

负责人：
协作者：

输入：
- 所有小组模块

输出：
- 稳定可运行的集成版本

涉及文件：
- `app.py`
- `src/config.py`
- `tests/`

验收标准：
- `python -m pytest` 通过。
- `python -m compileall src app.py` 通过。
- 不修改公共函数签名；如必须修改，先同步更新 `docs/api_contract.md` 并通知所有成员。

## B1 PDF / 文本解析

负责人：梁浩均
协作者：

输入：
- `data/raw/*.pdf`
- `data/raw/*.txt`
- `data/raw/*.md`

输出：
- `list[DocumentPage]`

涉及文件：
- `src/document_loader.py`

验收标准：
- 安装 PyMuPDF 后至少能解析 1 个中文 PDF。
- 能读取 `.txt` 或 `.md` 作为 fallback。
- 每页结果包含 `file_name`、`page`、`text`。

## B2 中文 Chunk 切分与知识点抽取

负责人：梁浩均
协作者：

输入：
- `list[DocumentPage]`

输出：
- `list[Chunk]`

涉及文件：
- `src/chunker.py`
- `tests/test_chunker.py`
- `tests/test_chinese_retrieval.py`

验收标准：
- 每个 chunk 包含 `chunk_id`、`file_name`、`page`、`text`、`chapter`、`concepts`。
- `chunk_id` 多次运行保持稳定。
- chunk_size 和 overlap 可配置。
- 能识别中文知识点，例如“检索增强生成”“重排序”“强化学习”。

## B3 中文 Embedding 与向量检索接口

负责人：梁浩均
协作者：

输入：
- 用户问题文本
- chunk 文本

输出：
- embedding 向量
- 向量相似度分数

涉及文件：
- `src/embedder.py`
- `src/vector_store.py`

验收标准：
- `mock` embedding 不需要下载模型即可运行。
- 后续可以接入中文 embedding 模型，例如 bge、bge-m3、m3e，不改变 `retrieve(...)` 接口。
- 检索结果中能看到数值型 dense_score。

## B4 中文 BM25 / 关键词检索

负责人：梁浩均
协作者：

输入：
- 用户问题
- `list[Chunk]`

输出：
- BM25 或关键词匹配分数

涉及文件：
- `src/bm25_store.py`
- `src/retriever.py`

验收标准：
- 安装 `jieba` 时使用中文分词。
- 没有 `jieba` 时能退化为中文单字 + bigram fallback。
- 每个 `RetrievedChunk` 包含 `bm25_score`。
- 同一个 query 的 Top-K 结果稳定可复现。

## B5 GraphRAG-lite 图扩展

负责人：梁浩均
协作者：

输入：
- seed `list[RetrievedChunk]`
- 全部 chunks

输出：
- 扩展后的 `list[RetrievedChunk]`

涉及文件：
- `src/graph_store.py`

验收标准：
- 同页 chunk 可以获得 graph_score。
- 共享中文知识点的 chunk 可以获得 graph_score。
- 扩展后的候选结果仍然使用统一的 `RetrievedChunk` 结构。

## C1 MiniRanker 重排序

负责人：梁浩均
协作者：

输入：
- 用户问题
- `list[RetrievedChunk]`

输出：
- `list[RankedChunk]`

涉及文件：
- `src/miniranker.py`

验收标准：
- 每个结果包含 `ranker_score`。
- UI 能展示重排序分数。
- 没有 `miniranker.pt` 时，`mock` 模式仍然可运行。

## C2 LLM 客户端与模式切换

负责人：
协作者：

输入：
- Prompt
- 环境变量

输出：
- 生成文本

涉及文件：
- `src/llm_client.py`
- `src/config.py`
- `.env.example`

验收标准：
- `mock` 模式返回稳定中文文本。
- `real` 模式的真实 API 接入位置明确。
- 缺少 API Key 时，`mock` 模式不能崩溃。

## C3 答案生成与原文引用

负责人：
协作者：

输入：
- 用户问题
- `list[RankedChunk]`

输出：
- 包含答案和引用的 dict

涉及文件：
- `src/generator.py`

验收标准：
- 答案包含 citation entries。
- 每条引用包含 `chunk_id`、`file_name`、`page`、score。
- 引用数据能被 `app.py` 渲染。

## C4 拒答机制与文档总结

负责人：
协作者：

输入：
- 用户问题
- 排序后的证据
- chunks

输出：
- 是否拒答
- 文档总结

涉及文件：
- `src/answer_guard.py`
- `src/generator.py`

验收标准：
- 证据不足的问题可以拒答。
- 文档总结在 `mock` 模式下可运行。
- 能用资料外问题演示拒答，例如“这份资料是否讨论 YOLOv10？”

## D1 中文自动出题

负责人：
协作者：

输入：
- `list[Chunk]`

输出：
- `list[QuizItem]`

涉及文件：
- `src/study_tools.py`

验收标准：
- 有足够 chunk 时至少生成 3 道题。
- 每道题包含 question、options、answer、explanation、concept、source_chunk_id。
- 题目能在 `app.py` 中展示并提交反馈。

## D2 Bandit 复习推荐

负责人：
协作者：

输入：
- 答题反馈
- 知识点 concept

输出：
- 推荐复习知识点
- 持久化状态文件

涉及文件：
- `src/bandit_recommender.py`
- `tests/test_bandit.py`

验收标准：
- `update_feedback(concept, correct)` 能记录答题次数。
- 答错会提高该知识点的复习优先级。
- 推荐结果包含 concept、score、scores、reason。

## D3 测试问题与实验记录

负责人：
协作者：

输入：
- 中文课程资料
- 演示问题

输出：
- 测试问题集
- 实验计划和结果

涉及文件：
- `demo/test_cases.csv`
- `docs/experiment_plan.md`

验收标准：
- 汇报前至少准备 20 个中文测试问题。
- 问题类型包含事实问答、总结、自动出题和资料外问题。
- 实验指标包含 Recall@5、答案正确率、引用准确率、拒答准确率和平均响应时间。

## D4 PPT、截图与成员贡献记录

负责人：
协作者：

输入：
- 应用截图
- 实验记录
- 成员工作记录

输出：
- PPT 素材
- 演示截图

涉及文件：
- `demo/screenshots/`
- `ppt/member_contributions.md`

验收标准：
- 每个成员都有贡献记录。
- 截图覆盖问答、引用、MiniRanker 分数、自动出题和 Bandit 推荐。
- PPT 只能写已经实现的功能；fallback 或 mock 功能必须明确标注。

