# CourseMind 任务悬赏列表

> 用途：给组员认领任务。每个任务必须有仓库内可验收交付物，不接受只在微信群口头说明。
> 悬赏分不是钱，是工作量和汇报贡献权重参考。

## 总体判断

关于 X3 和 X4，我重新判断后不建议完全合并。

原因是：

1. **概念上它们是一条链路**：X4 微调出来的 embedding 模型，最终必须交给 X3 的入库流程生成向量并重建 FAISS。
2. **工程上它们不应该变成一个大任务**：本地 Transformer embedding 部署是可控任务，fine-tuning 是高风险任务，受数据量、显卡、训练时间和效果波动影响。如果完全合并，组员容易没人敢领。
3. **验收上必须绑定接口**：X4 不能只交训练脚本和报告，必须产出一个能被 X3 加载的模型目录；X3 不能只支持现成模型，必须支持加载 X4 的微调模型。

所以任务设计改成：

```text
X3：本地中文 Transformer Embedding 部署与 FAISS 入库
X4：Embedding Fine-tuning 数据、训练脚本与模型交付

X4 的最终输出必须能作为 X3 的输入。
```

这比“完全拆开”更有闭环，也比“完全合并”更适合分工。

## 主线

当前项目主线：

```text
中文深度学习资料
-> 文档解析与 chunk
-> 本地 embedding / 微调 embedding
-> FAISS 检索
-> GraphRAG / MiniRanker 增强
-> 中文问答、出题和复习推荐
-> 检索评测与汇报实验
```

几个原则：

1. 数据收集和评测集优先级最高。没有数据和问题标注，后面的模型、微调、GraphRAG 都无法证明有效。
2. fine-tuning 不做“从零训练 BERT”。推荐做 sentence-transformers 风格的 embedding 对比学习微调，或者先做小样本 dry-run。
3. RAG、FAISS、GraphRAG、MiniRanker、Bandit、Streamlit 是系统实现技术，不是知识库的数据主题。
4. 中文出题和 Bandit 推荐适合分给组员，因为容易做出可见 demo 和可验收测试。

## 数据范围

知识库主内容围绕深度学习课程，不围绕 CourseMind 的系统实现技术。

主数据范围：

```text
神经网络基础
反向传播
损失函数
优化器
CNN
RNN / LSTM / GRU
Transformer / 注意力机制
自编码器
GAN
深度强化学习 / DRL
DQN
策略梯度
Actor-Critic
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
Streamlit
```

这些属于系统实现技术，只放在架构文档、实验报告和 PPT 技术路线里。

## 数据格式建议

建议收集这些数据：

```text
data/raw/                            全组共享的深度学习课程资料
data/raw/成员名/                     某个成员整理的一组资料
data/eval/retrieval_queries.csv      检索评测问题
data/eval/qa_gold.jsonl              问答标准答案和依据 chunk
data/eval/quiz_gold.jsonl            出题评测样本
data/training/embedding_pairs.jsonl  embedding 微调训练对
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

## Embedding 专项接口约定

X3 和 X4 之间必须遵守这个接口：

```text
X3 输入：
- EMBEDDING_PROVIDER=sentence_transformers
- EMBEDDING_MODEL_PATH=模型目录或 HuggingFace 模型名

X4 输出：
- models/embedding/finetuned/
- docs/embedding_finetuning_report.md
- data/training/embedding_pairs.jsonl
```

验收时必须能跑通：

```powershell
$env:EMBEDDING_PROVIDER="sentence_transformers"
$env:EMBEDDING_MODEL_PATH="models/embedding/finetuned"
python scripts/ingest.py
```

如果没有训练资源，X4 至少要提供 dry-run 小模型或小样本流程，保证接口和脚本可复现。

## 任务

### X1 数据收集与清洗基准集

悬赏分：10

负责人：
协作者：

输入：
- 深度学习课程 PDF
- PPT 导出的 PDF
- 课堂笔记
- 教材章节摘要
- 个人整理的概念笔记

输出：
- `data/raw/` 下的资料集合
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
- 明确哪些资料可以公开提交，哪些只能本地使用

### X3 本地中文 Transformer Embedding 部署与 FAISS 入库

悬赏分：15

负责人：
协作者：

输入：
- `data/processed/chunks.jsonl`
- 中文 embedding 基座模型，例如 `BAAI/bge-small-zh-v1.5`、`moka-ai/m3e-small`
- X4 产出的微调模型目录，例如 `models/embedding/finetuned/`

输出：
- 真实 transformer embedding 向量
- 使用基座模型重建的 FAISS 索引
- 使用 X4 微调模型重建的 FAISS 索引
- 部署记录

涉及文件：
- `src/embedder.py`
- `src/vector_store.py`
- `scripts/ingest.py`
- `.env.example`
- `docs/embedding_deployment.md`

验收标准：
- 支持 `EMBEDDING_PROVIDER=sentence_transformers`
- 支持 `EMBEDDING_MODEL_PATH=BAAI/bge-small-zh-v1.5`
- 支持 `EMBEDDING_MODEL_PATH=models/embedding/finetuned`
- `python scripts/ingest.py` 能成功生成 FAISS 索引
- 索引维度与模型输出一致，例如 384 或 768
- 记录 CPU 推理耗时和内存占用
- 明确 mock、lite、base transformer、fine-tuned transformer 的区别

### X4 Embedding Fine-tuning 数据、训练脚本与模型交付

悬赏分：18

负责人：
协作者：

输入：
- `data/eval/retrieval_queries.csv`
- 正样本：query 与正确 chunk
- 负样本：query 与错误 chunk
- X3 已验证可运行的中文 embedding 基座模型

输出：
- `data/training/embedding_pairs.jsonl`
- `scripts/build_training_pairs.py`
- `scripts/train_embedding.py`
- `models/embedding/finetuned/`
- `docs/embedding_finetuning_report.md`

涉及文件：
- `data/training/embedding_pairs.jsonl`
- `scripts/build_training_pairs.py`
- `scripts/train_embedding.py`
- `src/embedder.py`
- `docs/embedding_finetuning_report.md`

验收标准：
- 至少构造 200 条 `query-positive-negative` 训练样本
- 不做从零预训练 BERT，只做 embedding 模型微调或 dry-run
- 产出的 `models/embedding/finetuned/` 能被 X3 的 `EMBEDDING_MODEL_PATH` 加载
- 报告包含 baseline Recall@5 和 fine-tuned Recall@5
- 如果机器训练不了，也要提供可复现实验脚本和小样本 dry-run
- 不能只交“训练想法”，必须交可运行脚本、样本数据和评测结果

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
- 图扩展的可解释结果

涉及文件：
- `src/graph_store.py`
- `tests/test_graph_store.py`
- `docs/graphrag_design.md`

验收标准：
- 图节点至少包含 `chunk`、`concept`、`page`、`file`
- 图边至少包含 `same_page`、`same_chapter`、`adjacent`、`shared_concept`
- UI 或日志能展示为什么某个 chunk 被 GraphRAG 扩展出来
- 对比实验包含“无 GraphRAG”和“有 GraphRAG”的 Recall@5

### X6 中文出题质量升级

悬赏分：10

负责人：
协作者：

输入：
- `list[Chunk]`
- concepts
- 检索证据

输出：
- 更像中文考试题的选择题、判断题、简答题
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
- 报告中明确 mock、lite、transformer embedding、fine-tuned embedding 的区别
- PPT 可以直接引用实验表格或截图

## 推荐认领顺序

1. X1 数据收集与清洗基准集
2. X8 中文编码与文档修复
3. X3 本地中文 Transformer Embedding 部署与 FAISS 入库
4. X9 检索评测与实验报告
5. X4 Embedding Fine-tuning 数据、训练脚本与模型交付
6. X6 中文出题质量升级
7. X7 Bandit 复习推荐升级
8. X5 GraphRAG 图结构优化

## 给负责人的判断标准

如果组员参与积极性不高，优先自己完成 X1、X3、X9。这三个最能支撑最终汇报：有数据、有模型、有实验。

如果时间不够，X4 可以降级为：

```text
构造训练数据
+ dry-run 小样本训练
+ 证明微调模型能被 X3 加载
+ 写清楚训练前后评测方案
```

不要为了训练一个效果不稳定的模型牺牲主 demo。X4 的最低目标不是“训练出很强的模型”，而是把训练数据、训练脚本、模型输出目录、X3 加载接口、Recall 对比这条链路跑通。
