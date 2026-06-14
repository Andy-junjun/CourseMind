# GPU 服务器部署指南（RTX 3090 等）

面向带 GPU 的服务器（实测目标：RTX 3090 24G + 90G 内存 + 14 核）。这类机器可以
**全速跑 BGE（GPU）**、**现场重新微调（1-2 分钟）**，与早期 2GB 低配方案完全不同。

## 一键部署（推荐）

```bash
git clone <repo> && cd CourseMind
export DEEPSEEK_API_KEY=你的Key      # 可选，留空则用 mock 本地回答
bash scripts/deploy_gpu.sh
```

脚本依次完成：建虚拟环境 → 装依赖（基础 + GPU 版 torch）→ 校验 CUDA →
写 `.env`（BGE 基座 + DeepSeek）→ **用 GPU 重建 FAISS 索引** → 启动 Streamlit。

打开 `http://<服务器IP>:8501/`。

## 关键设计与取舍

| 项 | 选择 | 原因 |
|------|------|------|
| Embedding | BGE 基座（不指定 MODEL_PATH） | §3.4 实测基座在独立测试集检索最强（Recall@5=1.0） |
| 索引 | 服务器用 GPU 重建 | raw 资料随 git 携带；维度与线上 embedding 天然一致，不会错配 |
| LLM 生成 | DeepSeek API | 不占显存、效果好；联网即可 |
| torch | **CUDA 版**（requirements-transformer-gpu.txt） | CPU 版 torch 不会用 GPU，必须装 cu121 版才能提速 |

## CPU 版与 GPU 版依赖的区别

- `requirements-transformer.txt`：`torch==2.3.1+cpu` —— 本地开发/低配机用。
- `requirements-transformer-gpu.txt`：`torch==2.3.1`（cu121）—— GPU 服务器用，
  让 sentence-transformers 自动走显卡。服务器 CUDA 版本不同就把 `cu121` 改成对应版本。

校验 GPU 是否真的被用上：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: True NVIDIA GeForce RTX 3090
```

## 在服务器上重新微调（可选，GPU 1-2 分钟）

这台机器微调 BGE 比本地 CPU 快几十倍，适合**现场演示“我们会微调这条技术线”**：

```bash
# 1. 防泄漏划分 + 构造训练对（只用训练集）
python scripts/split_dataset.py --test-ratio 0.2 --seed 42
python scripts/build_training_pairs.py --queries-file data/eval/train_queries.csv --target-count 2000

# 2. GPU 微调（batch 可比 CPU 调大）
python scripts/train_embedding.py --epochs 3 --batch-size 16 --warmup-steps 80

# 3. 三档对比评测（lite / BGE 基座 / 微调）
python scripts/evaluate_retrieval.py --all

# 4. 若要让线上用微调模型：.env 设 EMBEDDING_MODEL_PATH=models/embedding/finetuned 后重建索引
python scripts/ingest.py
```

> 诚实提醒（§3.4 结论）：小规模微调在**独立测试集**上反而不如基座，根因是弱负样本 +
> 小数据过拟合，换 GPU 只是更快、不改变结论。建议把微调作为“可复现的实验过程 +
> 我们发现并分析了退化原因”来展示，主力检索仍用基座。改进方向：换
> MultipleNegativesRankingLoss、清洗 hard negative、扩大训练集。详见
> [presentation_dl_rl.md](presentation_dl_rl.md) §3.4。

## 后台常驻运行（演示期间不掉线）

脚本默认前台运行（Ctrl+C 即停）。想让它常驻：

```bash
# 用 nohup
nohup bash scripts/deploy_gpu.sh > logs/deploy.log 2>&1 &

# 或 tmux（推荐，可随时回看）
tmux new -s coursemind
bash scripts/deploy_gpu.sh
#  Ctrl+B 然后 D 脱离；tmux attach -t coursemind 回到会话
```

## 安全提醒

- `--server.address 0.0.0.0` 把服务暴露到公网且**无鉴权**。仅用于临时演示；
  演示后关闭，或用阿里云安全组只放行需要的来源 IP。
- DeepSeek Key 放 `.env`（已被 .gitignore 忽略），**不要**提交到仓库或写进代码。
- 微调模型权重（`models/embedding/finetuned/`，约 92MB）默认不入库，在服务器本地生成即可。

## 资源占用参考（见 technical_design.md §26）

GPU 服务器内存/显存都不是瓶颈，这里仅作对照：lite 检索核心 ~205MB、BGE 编码 ~440MB
（CPU 实测值）。3090 的 24G 显存还可额外承载 7B-14B 级开源 LLM，若日后想完全离线
（不调 DeepSeek API），可在此基础上扩展 `src/llm_client.py`。
