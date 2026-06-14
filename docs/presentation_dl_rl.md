# CourseMind 汇报技术文档：深度学习 + 强化学习

> 面向课程汇报的讲解稿。重点说明两条技术主线：
> **① 深度学习** —— 用 Transformer 句向量做中文语义检索（RAG），并对召回结果做学习排序（Learning-to-Rank）；
> **② 强化学习** —— 用多臂老虎机（Multi-Armed Bandit / UCB1）做自适应复习推荐。
>
> 文档同时给出每个模块的"为什么这么做"和可引用的论文/博客资料，方便答辩时回应追问。

---

## 0. 一句话定位

CourseMind 是一个面向**中文深度学习课程资料**的智能学习助手：把 PDF / Markdown / TXT 解析成中文 chunk，建立 FAISS 向量索引，提供「带引用的问答」「概念理解练习」「自适应复习推荐」三大功能。

它刻意避开了"简单图像分类"这类被点名的套路，把汇报落点放在两个有深度的技术点上：

| 主线 | 技术 | 在项目里的角色 |
|------|------|----------------|
| 深度学习 | Transformer 句向量（BGE-small-zh）+ FAISS 稠密检索 + MLP 重排序 | 让系统"听懂"中文问题并找到正确的课程片段 |
| 强化学习 | 多臂老虎机 UCB1 | 根据答题历史决定"下一题考哪个知识点"，平衡探索与利用 |

---

## 1. 系统全景：一条 RAG 检索链路

```text
data/raw 课程资料 (pdf / md / txt)
  └─> 文档解析（只读 PDF 文本层）
      └─> 中文 chunk 切分 + 概念抽取
          └─> 中文 Embedding（Transformer 句向量，DL 主线①）
              └─> FAISS IndexFlatIP 稠密检索
                  └─> BM25 关键词分数补充（混合检索）
                      └─> GraphRAG-lite 图扩展（同页/同章/相邻/共享概念）
                          └─> MiniRanker 重排序（MLP 学习排序，DL 主线①）
                              └─> Answer Guard 答案保护（够不够答？）
                                  └─> LLM 生成带引用的回答（DeepSeek / mock）

并行的学习闭环：
  知识库概念 → 生成选择题/完形题 → 用户作答 → 记录对错
            → Bandit/UCB 更新 → 推荐下一个知识点（RL 主线②）
```

**汇报话术**：RAG（Retrieval-Augmented Generation，检索增强生成）的核心思想是"不让大模型凭记忆瞎答，而是先检索证据、再基于证据生成"。我们这套链路从检索到生成每一步都可解释、可引用，深度学习用在"检索"和"排序"两端，强化学习用在"出题策略"上。

---

## 2. 深度学习主线①-A：Transformer 中文句向量（Embedding）

### 2.1 这一步在做什么

把一段中文文本（问题或课程片段）映射成一个**定长向量**，让"语义相近"的文本在向量空间里距离也近。这是整个语义检索的地基。

代码入口：[src/embedder.py](../src/embedder.py)

### 2.2 三档可切换的 Embedding 提供方

项目用环境变量 `EMBEDDING_PROVIDER` 在三档之间切换，兼顾"能演示"和"够先进"：

| 档位 | 实现 | 维度 | 用途 |
|------|------|------|------|
| `sentence_transformers` | **BAAI/bge-small-zh-v1.5**（真正的 Transformer 句向量模型） | 512 | 正式汇报推荐 |
| `lite` | 本地轻量 hash 加权词袋（无需 PyTorch） | 384 | 低配机器演示 |
| `mock` | 确定性 Blake2b 哈希向量 | 128 | 单元测试 |

**关键点**：真正体现深度学习的是 `sentence_transformers` 档位下的 **BGE 模型**。`lite`/`mock` 只是为了让没装 GPU/torch 的同学也能把链路跑通，汇报时要说清楚"演示用 lite，正式效果用 BGE"。

### 2.3 为什么选 BGE-small-zh-v1.5

- **专为中文优化**：BGE（BAAI General Embedding）是智源研究院的开源中文 embedding 模型，在中文检索基准 C-MTEB 上表现优秀。
- **small 版够轻**：512 维、CPU 可跑，适合课程项目的硬件条件。
- **句向量范式**：基于 Transformer encoder + 对比学习训练，输出 L2 归一化向量，天然适配余弦相似度检索。

### 2.4 向量归一化与相似度

所有 embedding 都做 **L2 归一化**（`v / ‖v‖`），这样：

```
余弦相似度 cos(a, b) = a·b / (‖a‖‖b‖) = a·b   （归一化后分母为 1）
```

也就是说，**归一化之后，内积 = 余弦相似度**。这正是下一节 FAISS 用 `IndexFlatIP`（内积索引）的前提。

**可引用资料（Embedding / Transformer）**：
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 —— Transformer 原始论文。
- Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019 —— 句向量范式奠基作。
- Xiao et al., *C-Pack / BGE: Packed Resources For General Chinese Embeddings*, 2023 —— BGE 中文 embedding。
- 博客：Sentence-Transformers 官方文档 <https://www.sbert.net/> ；BGE 模型卡 <https://huggingface.co/BAAI/bge-small-zh-v1.5>

---

## 3. 深度学习主线①-B：Embedding 微调（可选加分项）

如果答辩想展示"我们不仅会用，还会训练"，这部分是亮点。

代码：[scripts/train_embedding.py](../scripts/train_embedding.py)、[scripts/build_training_pairs.py](../scripts/build_training_pairs.py)
报告：[docs/embedding_finetuning_report.md](embedding_finetuning_report.md)

### 3.1 微调目标

通用 BGE 模型对"我们这门课的术语"不一定最优。我们用课程自己的 `questions.csv` 构造训练对，对 BGE 做领域微调，让"课程问题"和"对应资料片段"在向量空间里更靠近。

### 3.2 训练范式：Triplet Loss（三元组损失）

构造 `(query, positive, negative)` 三元组：

```
query    = 课程里的一个真实问题
positive = 该问题对应的正确资料片段（expected_file 匹配）
negative = 同领域但不相关的片段
```

Triplet Loss 让模型学会：

```
距离(query, positive) + margin  <  距离(query, negative)
```

即"正样本要比负样本更近至少一个间隔 margin"。这是度量学习（metric learning）的经典做法。

### 3.3 数据集划分与训练（严格区分 train / test）

代码：[scripts/split_dataset.py](../scripts/split_dataset.py) 划分、[scripts/build_training_pairs.py](../scripts/build_training_pairs.py) 构造训练对、[scripts/train_embedding.py](../scripts/train_embedding.py) 微调。

**关键方法论：先划分，再训练。** 我们把全部课程问题按 query 维度做 **80/20 不重叠划分**（固定随机种子=42），**只用 train 构造训练对、只用 test 评测**。这一步是为了避免数据泄漏（见 §3.4 的反转故事）。

| 项目 | 值 |
|------|-----|
| 全部 in-scope query | 197 个（含为补足覆盖、由资料内容自动生成的 74 题） |
| 训练集 / 测试集 | 158 / 39（80/20，种子固定可复现） |
| 训练三元组 | 1896 条（仅来自 158 个训练 query × 每题约 12 个负样本） |
| 基座模型 | BAAI/bge-small-zh-v1.5（512 维，L2 归一化） |
| 损失 / 配置 | TripletLoss，3 epochs，batch=8，warmup=80，约 32 分钟 |
| train_loss | 训练后约 4.48 |

> 说明：1896 = 158 个训练问题 × 每题约 12 个负样本。测试集的 39 个问题**完全没有参与训练**。

### 3.4 检索效果评测：一个值得讲的"反转" ⚠️

代码：[scripts/evaluate_retrieval.py](../scripts/evaluate_retrieval.py)（为每档 spawn 独立子进程，避开 embedder 的 `lru_cache` 串档问题）
评测集：`data/eval/test_queries.csv`（**39 个独立测试问题，未参与训练**）
指标：**Recall@K**（金标文件是否进 top-K）、**MRR**（首个命中的倒数排名均值）

| 档位 | Recall@1 | Recall@3 | Recall@5 | MRR |
|------|:--------:|:--------:|:--------:|:---:|
| lite（轻量词袋） | 0.6667 | 0.9231 | 0.9487 | 0.7885 |
| **bge-base（原始 BGE）** | **0.7436** | **0.9744** | **1.0000** | **0.8470** |
| bge-finetuned（小规模微调） | 0.5897 | 0.7949 | 0.8718 | 0.7030 |

**这张表怎么讲（这是汇报的高光点，不是败笔）**：

1. **先讲我们踩过的坑**：最初我们用全部问题既训练又评测，看到"微调后 Recall@5 从 0.78 冲到 1.00"，一度以为微调大获成功。
2. **再讲我们如何发现问题**：意识到训练集和测试集重叠 = **数据泄漏（data leakage）**，那个 1.00 其实是"模型在背过的题上考满分"，没有意义。
3. **严格划分后的真相**：在 39 个**没见过**的问题上，**原始 BGE 才是最好的（Recall@5=1.00、MRR=0.85）**，我们的小规模微调反而把它做差了（MRR 0.85→0.70）。
4. **给出有深度的结论**：对一个已经在中文检索上很强的基座（BGE-small-zh），**小数据 + TripletLoss + 弱负样本的微调会导致过拟合与表示退化，直接使用基座反而更好**。

**为什么微调会退化（技术归因，答辩可深入）**：
- **弱负样本**：我们的负样本是"同目录里 BM25 较高的其他片段"，其中很多是**同主题的正确内容**（如另一段也讲 CNN）。TripletLoss 把它们硬推开，破坏了 BGE 原本良好的语义聚类。
- **小数据过拟合**：158 个训练 query、3 epoch，模型拟合了训练分布的细节，损害泛化。
- **强基座难超越**：BGE-small-zh 已在大规模中文语料上对比学习过，小规模领域微调的边际收益为负。

**结论的价值**：这个"发现泄漏 → 修正方法论 → 得到反直觉但正确的结论"的过程，本身就是一次合格的实验科学实践，比"我微调涨点了"更有说服力。**汇报时主动讲这个反转**。

**可复现命令**：

```bash
# 1. 按 query 80/20 划分（固定种子，避免泄漏）
python scripts/split_dataset.py --test-ratio 0.2 --seed 42
# 2. 只用训练集构造训练对
python scripts/build_training_pairs.py --queries-file data/eval/train_queries.csv --target-count 2000
# 3. 微调
python scripts/train_embedding.py --epochs 3 --batch-size 8 --warmup-steps 80
# 4. 只在独立测试集上评测三档
python scripts/evaluate_retrieval.py --all
```

**下一步（若要让微调真正生效）**：换 MultipleNegativesRankingLoss（用 batch 内其他样本作负例，更适合检索）、清洗 hard negative（剔除同主题正确片段）、减到 1 epoch + 更小学习率、或扩充到数千条跨成员训练对。

**可引用资料（微调 / 度量学习 / 数据泄漏）**：
- Schroff et al., *FaceNet*, CVPR 2015 —— Triplet Loss 经典来源。
- Karpukhin et al., *Dense Passage Retrieval (DPR)*, EMNLP 2020 —— 对比学习训练检索 embedding；其中 in-batch negatives 的思想即 MultipleNegativesRankingLoss。
- Henderson et al., *Efficient Natural Language Response Suggestion*, 2017 —— in-batch negative 采样。
- 关于数据泄漏：Kaufman et al., *Leakage in Data Mining*, KDD 2011。
- 博客：Sentence-Transformers Training / Losses 文档 <https://www.sbert.net/docs/package_reference/losses.html>

---

## 4. 深度学习主线①-C：稠密检索 + 混合检索 + 图扩展

这一段是"找证据"的核心，把 DL 句向量真正用起来。

### 4.1 FAISS 稠密检索

代码：[src/vector_store.py](../src/vector_store.py)

- **索引类型**：`IndexFlatIP`（Flat = 精确暴力检索，IP = 内积）。
- 因为向量已 L2 归一化，**内积 = 余弦相似度**，检索即"找语义最近的 chunk"。
- 课程级数据量（百级 chunk）下 Flat 精确检索足够快，无需近似索引（IVF/HNSW），结果可复现、好解释。

新增的增量入库 `extend_index()`：上传新文件时**只对新内容做 embedding**并 `clone_index` 后追加，不重跑全库、不破坏原索引（这是本次修复的一部分）。

### 4.2 混合检索：稠密 + BM25

代码：[src/retriever.py](../src/retriever.py)、[src/bm25_store.py](../src/bm25_store.py)

纯语义检索有个弱点：遇到**精确术语 / 缩写 / 数字**（如"LSTM""ReLU"）可能不如关键词匹配稳。所以加一路 BM25 风格的关键词分数，按固定权重融合：

```text
hybrid_score = 0.6 * dense_score + 0.4 * bm25_score
```

- `dense_score`：FAISS 余弦相似度（语义）
- `bm25_score`：query 与 chunk 的词项重叠 / query 词数（召回导向，[0,1]）
- 中文分词用 jieba（不可用时退化为字符 + bigram）

**汇报话术**：这叫 hybrid retrieval（混合检索）。语义召回管"意思对不对"，关键词召回管"术语命中没命中"，两者互补，是当前 RAG 工程的主流做法。

### 4.3 GraphRAG-lite 图扩展

代码：[src/graph_store.py](../src/graph_store.py)

只靠相似度，有时会漏掉"语义上下文相邻"的片段。我们用文档结构构建轻量图，按四种关系给候选补一个 `graph_score`（取与种子 chunk 的最大关系分）：

| 关系 | 分值 | 含义 |
|------|------|------|
| 同页 | 0.30 | 同一文件同一页 |
| 同章节 | 0.20 | 同一 chapter |
| 相邻 chunk | 0.25 | 同文件内位置相邻 |
| 共享概念 | 0.40 + 0.10×额外 | 概念词重叠（用类 IDF 过滤掉太常见的词） |

这样能把"正确答案所在页的前后文"一并捞上来，提升答案完整度，又不需要部署真正的图数据库。

**可引用资料（检索 / RAG / 混合检索）**：
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 —— RAG 原始论文。
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009 —— BM25 经典综述。
- Johnson et al., *Billion-scale similarity search with GPUs (FAISS)*, 2017 —— FAISS。
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, 2024 —— GraphRAG。
- 博客：FAISS Wiki https://github.com/facebookresearch/faiss/wiki ；Pinecone "Hybrid Search" 指南 https://www.pinecone.io/learn/hybrid-search-intro/

---

## 5. 深度学习主线①-D：MiniRanker 学习排序（Learning-to-Rank）

代码：[src/miniranker.py](../src/miniranker.py)

### 5.1 为什么还要重排序

检索召回的是"差不多相关"的一批片段，但排在最前面的不一定是"最适合直接回答"的那一段。重排序（rerank）就是用一个更精细的小模型，对候选重新打分排序。这是工业检索系统（搜索、推荐、RAG）的标准第二阶段。

### 5.2 6 维特征 + 小型 MLP

每个候选片段抽取 6 个特征：

```text
[0] dense_score      稠密语义分
[1] bm25_score       关键词分
[2] graph_score      图扩展分
[3] same_page_bonus  与锚点同页(0/1)
[4] same_chapter     与锚点同章(0/1)
[5] length_norm      chunk 长度归一(目标400字)
```

真实模式下加载一个 PyTorch MLP 打分：

```text
nn.Linear(6, 32) → ReLU → Dropout(0.1)
→ nn.Linear(32, 16) → ReLU
→ nn.Linear(16, 1) → Sigmoid    # 输出 [0,1] 的相关性分
```

模型不可用时退化为固定权重的启发式打分（权重 `[0.42, 0.32, 0.14, 0.05, 0.04, 0.03]` 过 sigmoid），保证 mock/演示也能跑。

### 5.3 最终分数校准

把模型分和原始分数再融合一次，并乘一个内容质量因子：

```text
calibrated = (0.45*model + 0.35*dense + 0.15*bm25 + 0.05*graph) * 质量因子
```

**汇报话术**：这就是经典的"两阶段检索"——第一阶段（FAISS+BM25）追求**召回**，第二阶段（MiniRanker MLP）追求**精排**。我们的 reranker 虽小，但展示了 Learning-to-Rank 的完整思路：特征工程 + 神经网络打分 + 分数校准。

**可引用资料（重排序 / LTR）**：
- Nogueira & Cho, *Passage Re-ranking with BERT*, 2019 —— 神经重排序代表作。
- Burges et al., *Learning to Rank using Gradient Descent (RankNet)*, ICML 2005 —— LTR 奠基。
- 博客：Cohere / Pinecone 的 "Rerankers and Two-Stage Retrieval" 介绍。

### 5.4 Answer Guard：宁可不答，不要乱答

代码：[src/answer_guard.py](../src/answer_guard.py)

生成前先判断"证据够不够"。当最佳证据 `ranker_score < 0.52` 且关键词也不强、或问题命中越界词（如"股票""电影"）、或证据为空时，系统**拒答并说明原因**，而不是让大模型硬编。这是 RAG 系统对抗"幻觉"的重要一环，答辩时是个加分细节。

---

## 6. 强化学习主线②：多臂老虎机（UCB1）做自适应复习推荐

这是项目里**最能体现强化学习思想**的部分，建议作为汇报重头戏。

代码：[src/bandit_recommender.py](../src/bandit_recommender.py)

### 6.1 问题建模：把"该复习哪个知识点"变成多臂老虎机

把每一个**知识点（concept）当作老虎机的一条"臂"（arm）**：

- 每次给用户出一道某知识点的题 = "拉动"那条臂；
- 用户答对/答错 = 这条臂返回的"奖励"反馈；
- 目标：在有限的练习次数里，**既要多练用户的薄弱点（利用 exploitation），又要给练得少的知识点机会（探索 exploration）**。

这正是多臂老虎机（Multi-Armed Bandit, MAB）的经典 **探索 vs 利用（exploration-exploitation）** 困境。

### 6.2 核心算法：UCB1（Upper Confidence Bound）

每个知识点算一个 UCB 分数，每轮推荐分数最高的：

```text
对已练过的知识点 i：
    wrong_rate = wrong_i / attempts_i              # 利用项：错得越多越该练
    explore    = c * sqrt( ln(N+1) / attempts_i )  # 探索项：练得越少奖励越高
    UCB_i      = wrong_rate + explore

对从未练过的知识点：
    UCB_i = 0.5 + c * sqrt( ln(N+1) )              # 冷启动给高探索分，确保被试一次

其中：
    N = 所有知识点的总练习次数
    c = 1.414 ≈ √2                                 # 探索强度常数（UCB1 标准取值）
```

**逐项讲解**：
- **利用项 `wrong_rate`**：错题率高 → 说明是薄弱点 → 分数高 → 优先推荐。
- **探索项 `c·√(ln N / nᵢ)`**：某知识点练得少（`nᵢ` 小）→ 探索项大 → 也会被推荐，避免系统只盯着一个点反复考。随总次数 `N` 增长，探索项缓慢上升（`ln`），保证长期不会彻底冷落任何臂。
- **常数 `c = √2`**：来自 UCB1 论文的理论取值，平衡探索与利用。
- **冷启动**：没练过的知识点直接给一个很高的探索分，确保"每个臂至少被试一次"。

### 6.3 状态持久化

每个知识点维护：

```json
{
  "知识点名": {
    "attempts": 练习次数,
    "wrong": 错误次数,
    "last_seen": "ISO 时间戳",
    "source_chunk_ids": ["来源片段"]
  }
}
```

写入 `BANDIT_STATE_PATH`（默认 `data/processed/quiz_state.json`），所以推荐会**跨会话记忆**用户的答题历史。

### 6.4 本次修复的一致性保证（可讲的工程细节）

原来存在一个 bug：Bandit 可能推荐一个"无法出题"的知识点，导致页面静默 fallback 到无关题目，"推荐复习"和"实际题目"对不上。

修复：新增 `quizzable_concepts()` 作为"可出题知识点"的唯一真相来源，`recommend_concept(allowed_concepts=...)` 把推荐**限制在可出题集合内**，保证推荐的知识点和下一道题永远是同一个。这体现了"算法推荐要和可执行动作空间对齐"的工程意识。

**可引用资料（强化学习 / 老虎机）**：
- Auer, Cesa-Bianchi & Fischer, *Finite-time Analysis of the Multiarmed Bandit Problem*, Machine Learning 2002 —— **UCB1 原始论文**（必引）。
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), 2018 —— 第 2 章专讲多臂老虎机与 UCB，免费 PDF：http://incompleteideas.net/book/the-book-2nd.html
- Lattimore & Szepesvári, *Bandit Algorithms*, 2020 —— 老虎机理论权威教材，免费在线。
- 应用类比：教育领域的"间隔重复 / 自适应出题"（如 SuperMemo SM-2、Duolingo 的 Half-Life Regression），可作为"为什么用 bandit 做复习推荐"的应用佐证。

---

## 7. 出题引擎：把知识库变成题目

代码：[src/study_tools.py](../src/study_tools.py)

Bandit 决定"考哪个知识点"，出题引擎负责"把这个知识点变成一道题"。两种题型：

- **概念理解选择题**：用知识点对应的合规中文陈述句作为正确选项，干扰项来自其他概念。
- **关键词遮罩完形题**：遮住片段里的关键术语让用户填空。

工程细节（可作为"细节扎实"的佐证）：
- 概念可教性过滤 `is_teachable_concept`：排除系统噪声词（RAG/FAISS/Bandit 等）、长度限制 2–24 字、必须含中文或在英文白名单（CNN/RNN/LSTM/Transformer…）。
- 候选打分综合**频率、特异性、是否中文**，避免老考同一个词。
- 选项顺序用 **SHA256 确定性打乱**（按 chunk_id+concept 播种），既避免"答案永远是 A"的偏置，又保证可复现。

---

## 8. 生成层与运行模式

代码：[src/generator.py](../src/generator.py)、[src/config.py](../src/config.py)

- **生成**：检索到的证据拼成 prompt，交给 DeepSeek 生成"基于证据、带引用"的长回答；API 不可用时退回"本地证据片段摘要"，引用仍保留，问答链路不中断。
- **三种运行模式**（用 `COURSEMIND_MODE` 切换）：
  - `mock`：全本地确定性 fallback，给测试和无网络环境用；
  - `hybrid`：优先真实模块，失败自动 fallback；
  - `real`：全真实 embedding / FAISS / reranker / LLM。

**汇报话术**：这套"真实优先 + 优雅降级"的设计，让我们在任何机器上都能现场演示，不会因为没网或没 GPU 而翻车。

---

## 9. 现场演示脚本（建议 5 分钟）

1. **问答 Tab**：问一个课程概念（如"什么是注意力机制"），展示 → 检索到的证据片段 + 引用来源 + 生成回答。强调"答案有出处，不是瞎编"。
2. 故意问一个越界问题（如"推荐只股票"），展示 **Answer Guard 拒答**，说明抗幻觉设计。
3. **练习 Tab**：连续答错某个知识点 2–3 次，刷新后展示**推荐知识点跟着变**，讲 UCB 的"错得多→优先推荐"；再指出"推荐和题目永远一致"（本次修复点）。
4. **上传 Tab**：上传一个小 md/txt，展示"只对新文件 embedding 并入原库"，页数/chunk 数增加，原资料仍可检索。
5. （可选）展示 `models/embedding/finetuned/` 和训练日志，说明 embedding 微调链路已跑通。

---

## 10. 预设答辩问题与应答

| 可能的提问 | 建议应答 |
|------|------|
| 这跟直接调 ChatGPT 有什么区别？ | 我们是 RAG：先从课程资料检索证据再生成，答案可溯源、可引用，且不依赖大模型的记忆，资料更新即时生效。 |
| 为什么用 IndexFlatIP 不用近似索引？ | 课程数据量百级 chunk，精确检索已足够快且结果可复现；数据量上万再换 IVF/HNSW。 |
| 这里的强化学习是不是太简单？ | 多臂老虎机是强化学习中"无状态转移"的基础模型，UCB1 有严格的 regret 理论界（Auer 2002）。我们用它解决真实的探索-利用问题，并保证了推荐与可执行动作空间一致。 |
| 你们的 embedding 微调真有提升吗？ | 我们做了诚实的对照实验：在严格划分的独立测试集（39 题，未参与训练）上，**原始 BGE 反而最好（Recall@5=1.00、MRR=0.85），小规模微调因弱负样本+过拟合而退化（MRR 0.70）**。最初"同源评测"看到的 1.00 是数据泄漏假象，我们发现并纠正了它。结论：对强中文基座，直接用比小规模微调更好（详见 §3.4）。 |
| BM25 是标准实现吗？ | 我们用的是简化的词项重叠召回分（非带 k1/b 的标准 BM25），目的是给稠密检索补一路关键词信号；标准 BM25 是后续可替换项。 |

---

## 11. 学习资料汇总（按主题）

**深度学习 / Transformer / 句向量**
- Vaswani et al. 2017, *Attention Is All You Need* — Transformer。
- Reimers & Gurevych 2019, *Sentence-BERT* — 句向量。
- Xiao et al. 2023, *C-Pack/BGE* — 中文 embedding。
- 李宏毅《机器学习》课程（B 站）— 中文讲 Transformer/Attention，适合补基础。

**检索 / RAG / 重排序**
- Lewis et al. 2020, *RAG*。
- Karpukhin et al. 2020, *DPR*（稠密检索）。
- Nogueira & Cho 2019, *Passage Re-ranking with BERT*（重排序）。
- Edge et al. 2024, *GraphRAG*。

**强化学习 / 多臂老虎机**
- Auer et al. 2002, *Finite-time Analysis of MAB*（UCB1，必读）。
- Sutton & Barto 2018, *RL: An Introduction* 第 2 章（免费 PDF）。
- Lattimore & Szepesvári 2020, *Bandit Algorithms*（免费在线）。

**工具文档**
- Sentence-Transformers: https://www.sbert.net/
- FAISS Wiki: https://github.com/facebookresearch/faiss/wiki
- BGE 模型: https://huggingface.co/BAAI/bge-small-zh-v1.5

---

## 12. 一页总结（汇报收尾用）

> CourseMind 用**深度学习**解决"听懂中文问题、找对课程证据"——Transformer 句向量（BGE-small-zh）做语义编码，FAISS 做稠密检索，混合 BM25 与 GraphRAG-lite 补召回，再用小型 MLP 做学习排序精排，全程带引用、可拒答。我们还做了一次严谨的微调对照实验：构造 197 题数据集、80/20 划分 train/test，发现并纠正了"同源评测"的数据泄漏，在独立测试集上得出**原始 BGE 检索最强（Recall@5=1.00）、小规模微调反而因弱负样本与过拟合退化**这一反直觉但可信的结论。用**强化学习**解决"该复习什么"——把知识点建模为多臂老虎机，用 UCB1 在"练薄弱点"与"探索新点"之间自适应权衡。两条主线都不是玩具：检索链路是工业 RAG 的标准两阶段架构，老虎机有严格的理论保证，而我们对微调的评估体现了实验科学的严谨性。

> 更完整的全量架构、数据流和 API 契约见 [docs/technical_design.md](technical_design.md)。
