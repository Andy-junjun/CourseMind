#!/usr/bin/env bash
# CourseMind 一键部署脚本 —— 面向带 GPU 的服务器（如 RTX 3090 24G）。
#
# 用法：
#   1. git clone <repo> && cd CourseMind
#   2. 准备好 DeepSeek API Key（用于问答生成）
#   3. bash scripts/deploy_gpu.sh
#
# 脚本做的事：建虚拟环境 -> 装依赖（基础 + GPU 版 torch）-> 校验 CUDA
#            -> 写 .env（BGE 基座 + DeepSeek）-> 用 GPU 重建 FAISS 索引 -> 启动 Streamlit
#
# 设计选择：
#   - embedding 用 BGE 基座（sentence_transformers，不指定 MODEL_PATH）。§3.4 实测
#     基座在独立测试集上检索最强；想用微调权重见文末说明。
#   - 索引在服务器用 GPU 重建（raw 资料随 git 携带），不依赖本地预构建文件，
#     维度天然与线上 embedding 一致。
#   - 生成走 DeepSeek API，不占显存。
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "项目根目录: $PROJECT_ROOT"

# ---- 0. 读取 DeepSeek Key（环境变量优先，否则交互输入）----
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
if [ -z "$DEEPSEEK_API_KEY" ]; then
  read -rp "请输入 DeepSeek API Key（留空则用 mock 本地回答）: " DEEPSEEK_API_KEY
fi

# ---- 1. 虚拟环境 ----
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# ---- 2. 依赖：基础（无 torch）+ GPU 版 transformer ----
pip install -r requirements.txt
pip install -r requirements-transformer-gpu.txt

# ---- 3. 校验 GPU 可用 ----
echo "=== CUDA 自检 ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("警告: 未检测到 CUDA GPU，将退化为 CPU（仍可运行，但慢）。"
          "请检查驱动/CUDA 版本，或改用 requirements-transformer.txt（CPU 版）。")
PY

# ---- 4. 写 .env ----
if [ -n "$DEEPSEEK_API_KEY" ]; then
  LLM_PROVIDER=deepseek
else
  LLM_PROVIDER=mock
fi
cat > .env <<EOF
COURSEMIND_MODE=real
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
EMBEDDING_MODEL_PATH=
LLM_PROVIDER=${LLM_PROVIDER}
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
LLM_API_BASE=https://api.deepseek.com/chat/completions
LLM_MODEL=deepseek-v4-flash
BANDIT_STATE_PATH=data/processed/quiz_state.json
EOF
echo "已写入 .env（LLM_PROVIDER=${LLM_PROVIDER}）"

# ---- 5. 用 GPU 重建 FAISS 索引（raw 资料随仓库携带）----
echo "=== 重建向量索引（GPU 加速 BGE 编码）==="
python scripts/ingest.py

# ---- 6. 启动 ----
echo "=== 启动 Streamlit（0.0.0.0:8501）==="
echo "提示: 0.0.0.0 会暴露到公网且无鉴权，演示后请关闭或用安全组限制来源 IP。"
exec python -m streamlit run app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false

# ============================================================
# 想用“微调模型”而非基座，或想在服务器上重新微调（1-2 分钟，GPU）：
#   1. 划分数据并构造训练对（防泄漏）:
#        python scripts/split_dataset.py --test-ratio 0.2 --seed 42
#        python scripts/build_training_pairs.py --queries-file data/eval/train_queries.csv --target-count 2000
#   2. GPU 微调:
#        python scripts/train_embedding.py --epochs 3 --batch-size 16 --warmup-steps 80
#   3. 在 .env 里设 EMBEDDING_MODEL_PATH=models/embedding/finetuned 后重建索引:
#        python scripts/ingest.py
#   注意 §3.4：小规模微调在独立测试集上反而不如基座，建议作为“实验过程”展示。
# ============================================================
