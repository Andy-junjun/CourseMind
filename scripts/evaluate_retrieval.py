"""检索效果评估：对比 lite / 原始 BGE / 微调 BGE 三档 embedding 的 Recall@K 与 MRR。

为什么是子进程隔离：
    src/embedder.py 用 @lru_cache 缓存已加载的 SentenceTransformer，且缓存键不含
    模型路径。因此在同一个进程里先后评估 base BGE 和 finetuned BGE 会复用第一个加载
    的模型，结果失真。本脚本因此为每一档 spawn 一个独立子进程（--single 模式），主流程
    （--all 模式）只负责设置环境变量、收集各档结果并汇总成对比表。

用法：
    # 一次跑完三档并打印对比表（推荐）
    python scripts/evaluate_retrieval.py --all

    # 只评估当前环境变量指定的一档（供子进程调用，一般不手动用）
    COURSEMIND_MODE=real EMBEDDING_PROVIDER=lite \
        python scripts/evaluate_retrieval.py --single --label lite

评测指标：
    Recall@K : 金标文件(expected_file)是否出现在 top-K 检索结果中（命中率）
    MRR      : 首个命中金标文件的排名倒数的平均（Mean Reciprocal Rank）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_pages  # noqa: E402
from src.document_loader import load_pdf  # noqa: E402
from src.retriever import retrieve  # noqa: E402
from src.vector_store import build_index, load_chunks  # noqa: E402

DEFAULT_QUERIES = Path("data/eval/retrieval_queries-ddw.csv")
DEFAULT_CHUNKS = Path("data/processed/chunks.jsonl")
RESULT_DIR = Path("data/eval/_runs")
TOP_KS = (1, 3, 5)


def basename(path: str) -> str:
    """金标 expected_file 是裸文件名，chunk.file_name 带目录前缀，统一用 basename 比较。"""
    return Path(str(path).replace("\\", "/")).name.strip().lower()


def load_eval_queries(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            query = (row.get("query") or "").strip()
            expected = (row.get("expected_file") or "").strip()
            if query and expected:
                rows.append({"query": query, "expected_file": expected})
    return rows


def first_hit_rank(results, expected_file: str) -> int | None:
    """返回金标文件在检索结果中的 1-based 排名；未命中返回 None。"""
    target = basename(expected_file)
    for rank, item in enumerate(results, start=1):
        if basename(item.chunk.file_name) == target:
            return rank
    return None


def evaluate_single(label: str, queries_path: Path, chunks_path: Path) -> dict:
    """评估当前进程环境配置的那一档，返回指标字典。"""
    queries = load_eval_queries(queries_path)
    chunks = load_chunks(chunks_path)
    index = build_index(chunks)

    max_k = max(TOP_KS)
    hits = {k: 0 for k in TOP_KS}
    reciprocal_sum = 0.0
    per_query: list[dict] = []

    for q in queries:
        results = retrieve(q["query"], chunks, top_k=max_k, index=index)
        rank = first_hit_rank(results, q["expected_file"])
        for k in TOP_KS:
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal_sum += (1.0 / rank) if rank else 0.0
        per_query.append({"query": q["query"], "expected": q["expected_file"], "rank": rank})

    n = len(queries) or 1
    return {
        "label": label,
        "provider": os.getenv("EMBEDDING_PROVIDER", "?"),
        "model_path": os.getenv("EMBEDDING_MODEL_PATH", "") or os.getenv("EMBEDDING_MODEL_NAME", ""),
        "num_queries": len(queries),
        "recall": {f"@{k}": round(hits[k] / n, 4) for k in TOP_KS},
        "mrr": round(reciprocal_sum / n, 4),
        "per_query": per_query,
    }


# 三档配置：label -> 该子进程需要设置的环境变量。
# finetuned 档只有在微调模型目录存在时才会被评估。
TIERS = [
    ("lite", {"COURSEMIND_MODE": "real", "EMBEDDING_PROVIDER": "lite", "EMBEDDING_DIM": "384"}),
    (
        "bge-base",
        {
            "COURSEMIND_MODE": "real",
            "EMBEDDING_PROVIDER": "sentence_transformers",
            "EMBEDDING_MODEL_NAME": "BAAI/bge-small-zh-v1.5",
            "EMBEDDING_MODEL_PATH": "",
        },
    ),
    (
        "bge-finetuned",
        {
            "COURSEMIND_MODE": "real",
            "EMBEDDING_PROVIDER": "sentence_transformers",
            "EMBEDDING_MODEL_PATH": "models/embedding/finetuned",
        },
    ),
]


def run_all(queries_path: Path, chunks_path: Path) -> list[dict]:
    """为每一档 spawn 一个独立子进程评估（避开 embedder 的 lru_cache 串档问题）。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for label, env_overrides in TIERS:
        if label == "bge-finetuned" and not Path("models/embedding/finetuned").exists():
            print(f"[skip] {label}: models/embedding/finetuned 不存在，跳过该档")
            continue

        out_path = RESULT_DIR / f"{label}.json"
        child_env = os.environ.copy()
        child_env.update(env_overrides)
        cmd = [
            sys.executable,
            __file__,
            "--single",
            "--label",
            label,
            "--queries",
            str(queries_path),
            "--chunks",
            str(chunks_path),
            "--out",
            str(out_path),
        ]
        print(f"[run] {label}: provider={env_overrides.get('EMBEDDING_PROVIDER')} ...")
        proc = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[fail] {label} 评估失败：\n{proc.stderr.strip()[-800:]}")
            continue
        results.append(json.loads(out_path.read_text(encoding="utf-8")))
    return results


def print_comparison(results: list[dict]) -> None:
    if not results:
        print("没有可对比的结果。")
        return
    print("\n=== 检索效果对比（金标文件命中）===")
    n = results[0]["num_queries"]
    print(f"评测问题数：{n}\n")
    header = f"{'档位':<16}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        rc = r["recall"]
        print(
            f"{r['label']:<16}{rc['@1']:>10.4f}{rc['@3']:>10.4f}{rc['@5']:>10.4f}{r['mrr']:>10.4f}"
        )
    print("\n(明细见 data/eval/_runs/<档位>.json)")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估并对比三档 embedding 的检索效果")
    parser.add_argument("--all", action="store_true", help="跑完三档并打印对比表")
    parser.add_argument("--single", action="store_true", help="只评估当前环境配置的一档")
    parser.add_argument("--label", default="current", help="单档模式下的档位标签")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--out", type=Path, default=None, help="单档模式结果输出路径")
    args = parser.parse_args()

    if args.single:
        result = evaluate_single(args.label, args.queries, args.chunks)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("label", "recall", "mrr", "num_queries")}, ensure_ascii=False))
        return

    # 默认走 --all
    results = run_all(args.queries, args.chunks)
    print_comparison(results)
    summary_path = RESULT_DIR / "summary.json"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            [{k: r[k] for k in ("label", "provider", "num_queries", "recall", "mrr")} for r in results],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n汇总已写入 {summary_path}")


if __name__ == "__main__":
    main()
