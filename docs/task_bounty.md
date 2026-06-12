# CourseMind 任务悬赏列表

> 用途：给组员认领任务。每个任务必须有仓库内可验收交付物，不接受只在微信群口头说明。
> 悬赏分不是钱，是工作量和汇报贡献权重参考。

## 总体建议

你的 6 个优化方向整体是合适的，但优先级需要调整：

1. 数据收集和评测集优先级最高。没有数据和标注，后面的 Transformer、fine-tuning、GraphRAG 都无法证明有效。
2. 本地 Transformer embedding 值得做，可以体现深度学习，但建议先部署现成中文 embedding 模型，再考虑微调。
3. fine-tuning 不建议做“从零训练 BERT”。推荐做 sentence-transformers 风格的 embedding 对比学习微调，或 MiniRanker/reranker 微调。
4. 图像数据当前代码不能可靠识别。现在只支持 PDF 文本层、TXT、MD；扫描版 PDF、图片、PPT 截图需要 OCR 或版面解析任务。
5. GraphRAG 可以优化，但要先定义图节点和边，不要只写“知识图谱”四个字。
6. 中文出题和 Bandit 推荐很适合分给组员，因为容易做出可见 demo 和可验收测试。

## 数据范围修正

知识库主内容应围绕深度学习课程，而不是围绕 CourseMind 的系统实现技术。

主数据范围：

```text
神经网络基础
反向传播
损失函数
优化器
CNN
RNN / LSTM / GRU
Transformer / 注意力机制
深度强化学习 / DRL
模型训练与调参
实验结果分析
```

不要把这些作为知识库主数据主题：

```text
RAG
FAISS
GraphRAG
MiniRanker
Bandit 推荐
OCR
Streamlit
```

这些属于系统实现技术，只放在架构文档、实验报告和 PPT 技术路线里。

## 数据格式建议

建议收集这些数据：

```text
data/raw/course_notes/*.pdf          深度学习课程讲义，优先文本型 PDF
data/raw/slides/*.pdf                深度学习课程 PPT 导出的 PDF
data/raw/notes/*.md                  神经网络/CNN/RNN/Transformer/DRL 笔记
data/raw/images/*.png|*.jpg          截图或扫描页，只有 OCR 任务完成后再纳入主流程
data/eval/retrieval_queries.csv      检索评测问题
data/eval/qa_gold.jsonl              问答标准答案和依据 chunk
data/eval/quiz_gold.jsonl            出题评测样本
```

`retrieval_queries.csv` 建议字段：

```csv
query,expected_file,expected_page,expected_concept,expected_keywords,query_type
LSTM 如何缓解长期依赖问题？,LSTM课程笔记.md,1,LSTM,"遗忘门;输入门;记忆单元",fact
Transformer 为什么需要位置编码？,Transformer课程笔记.md,1,Transformer,"位置信息;序列顺序;注意力机制",fact
```

`qa_gold.jsonl` 建议每行：

```json
{"query":"LSTM 如何缓解长期依赖问题？","answer_keywords":["遗忘门","输入门","记忆单元"],"evidence_pages":[1],"must_refuse":false}
```

## 任务

### X1 数据收集与清洗基准集

悬赏分：10

负责人：
协作者：

输入：
- 课程 PDF、PPT 导出 PDF、课程说明、课堂笔记、教材节选

输出：
- `data/raw/` 下的本地资料集合
- `data/eval/retrieval_queries.csv`
- `data/eval/qa_gold.jsonl`
- `docs/data_collection_report.md`

涉及文件：
- `data/raw/`
- `data/eval/`
- `docs/data_collection_report.md`

验收标准：
- 至少收集 20 份中文资料或 100 页以上内容
- 至少准备 50 条检索问题
- 每条问题必须标注期望文件、页码或知识点
- 明确哪些资料可公开提交，哪些只能本地使用

### X2 图像与扫描 PDF OCR 解析

悬赏分：12

负责人：
协作者：

输入：
- `data/raw/images/*.png`
- 扫描版 PDF

输出：
- OCR 后的 `DocumentPage`
- OCR 结果缓存
- OCR 解析说明

涉及文件：
- `src/document_loader.py`
- `src/ocr_loader.py`
- `tests/test_ocr_loader.py`
- `docs/image_ocr_pipeline.md`

验收标准：
- 能解析至少 3 张中文截图或扫描页
- OCR 输出包含 `file_name`、`page`、`text`
- 普通文本 PDF 仍走原 PyMuPDF 流程
- OCR 缺依赖时不能影响普通 PDF/TXT/MD 解析

### X3 本地中文 Transformer Embedding 部署

悬赏分：15

负责人：
协作者：

输入：
- `data/processed/chunks.jsonl`
- 中文 embedding 模型，例如 `BAAI/bge-small-zh-v1.5`、`moka-ai/m3e-small`

输出：
- 真正的 transformer embedding 向量
- 重新构建的 FAISS 索引
- 部署记录

涉及文件：
- `src/embedder.py`
- `scripts/ingest.py`
- `.env.example`
- `docs/embedding_deployment.md`

验收标准：
- `COURSEMIND_MODE=real`
- `EMBEDDING_PROVIDER=sentence_transformers` 或可替代真实模型 provider
- `python scripts/ingest.py` 能成功生成 FAISS 索引
- 索引维度与模型输出一致，例如 384 或 768
- 记录 CPU 推理耗时和内存占用

### X4 Embedding Fine-tuning 数据与训练脚本

悬赏分：18

负责人：
协作者：

输入：
- `data/eval/retrieval_queries.csv`
- 正样本：query 与正确 chunk
- 负样本：query 与错误 chunk

输出：
- 训练样本 JSONL
- fine-tuning 脚本
- 训练前后 Recall@5 对比

涉及文件：
- `data/training/embedding_pairs.jsonl`
- `scripts/build_training_pairs.py`
- `scripts/train_embedding.py`
- `docs/fine_tuning_report.md`

验收标准：
- 至少构造 200 条 query-positive-negative 样本
- 不做从零预训练 BERT，只做 embedding 或 reranker 微调
- 报告包含 baseline Recall@5 和 fine-tuned Recall@5
- 如果机器训练不了，也要提供可复现实验脚本和小样本 dry-run

### X5 GraphRAG 图结构优化

悬赏分：12

负责人：
协作者：

输入：
- chunks
- concepts
- page/chapter/file 元数据

输出：
- 更明确的文档图
- 图扩展可解释结果

涉及文件：
- `src/graph_store.py`
- `tests/test_graph_store.py`
- `docs/graphrag_design.md`

验收标准：
- 图节点至少包含 `chunk`、`concept`、`page`、`file`
- 图边至少包含 `same_page`、`same_chapter`、`adjacent`、`shared_concept`
- UI 或日志能展示为什么某个 chunk 被 GraphRAG 扩展出来
- 对比实验包含 “无 GraphRAG” 与 “有 GraphRAG” 的 Recall@5

### X6 中文出题质量升级

悬赏分：10

负责人：
协作者：

输入：
- `list[Chunk]`
- concepts
- 检索证据

输出：
- 更像中文考试题的选择题/判断题/简答题
- 每题带解析和来源

涉及文件：
- `src/study_tools.py`
- `tests/test_study_tools.py`
- `docs/quiz_generation.md`

验收标准：
- 至少支持选择题、判断题、简答题三类
- 每道题包含 `question`、`options`、`answer`、`explanation`、`concept`、`source_chunk_id`
- 干扰项不能全是固定模板
- 页面能展示题目并提交反馈

### X7 Bandit 复习推荐升级

悬赏分：10

负责人：
协作者：

输入：
- 用户答题记录
- concept 正误统计

输出：
- 可解释的复习推荐
- 学习状态持久化

涉及文件：
- `src/bandit_recommender.py`
- `tests/test_bandit.py`
- `docs/bandit_design.md`

验收标准：
- 支持按 concept 记录 `attempts`、`wrong`、`last_seen`
- 推荐理由能解释“错得多”还是“练得少”
- 至少实现 UCB 或 epsilon-greedy 中一种
- 页面能显示推荐知识点、分数和原因

### X8 中文编码与文档修复

悬赏分：6

负责人：
协作者：

输入：
- 当前仓库 Markdown 和中文源码字符串

输出：
- 正常 UTF-8 中文文档
- 无乱码 UI 文案

涉及文件：
- `README.md`
- `docs/*.md`
- `src/study_tools.py`
- `src/bandit_recommender.py`

验收标准：
- README 和 docs 打开后中文正常
- Streamlit 页面中文正常
- 出题、Bandit 推荐理由中文正常
- `python -m pytest` 通过

### X9 检索评测与实验报告

悬赏分：12

负责人：
协作者：

输入：
- `data/eval/retrieval_queries.csv`
- 不同检索配置

输出：
- 实验结果表
- 汇报图表

涉及文件：
- `scripts/evaluate_retrieval.py`
- `docs/experiment_report.md`
- `demo/results/`

验收标准：
- 至少比较 4 组：BM25、FAISS、FAISS+BM25、FAISS+BM25+GraphRAG+MiniRanker
- 输出 Recall@1、Recall@5、平均响应时间
- 报告中明确 mock、lite、transformer embedding 的区别
- PPT 可以直接引用实验表格或截图

## 推荐认领顺序

1. X1 数据收集与清洗基准集
2. X8 中文编码与文档修复
3. X3 本地中文 Transformer Embedding 部署
4. X9 检索评测与实验报告
5. X6 中文出题质量升级
6. X7 Bandit 复习推荐升级
7. X5 GraphRAG 图结构优化
8. X4 Embedding Fine-tuning 数据与训练脚本
9. X2 图像与扫描 PDF OCR 解析

## 给负责人的判断标准

如果组员参与积极性不高，优先自己完成 X1、X3、X9、X10。这四个最能支撑最终汇报：有数据、有模型、有实验、有演示。

如果时间不够，fine-tuning 可以降级为“构造训练数据 + dry-run 小样本训练 + 训练前后评测方案”，不要为了训练一个效果不稳定的模型牺牲主 demo。
