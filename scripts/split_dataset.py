"""把所有检索 query 按 80/20 随机划分为训练集与测试集（固定随机种子，可复现）。

为什么要划分：
    若用全部 query 既训练 embedding 又评测，训练集与测试集重叠，会造成数据泄漏
    (data leakage)——评测分数偏乐观，不能反映模型在"没见过的问题"上的泛化能力。
    本脚本按 query 维度做不重叠划分，保证 train 与 test 没有同一个问题。

输入：
    data/raw/**/questions.csv  各成员人工整理的问题
    data/eval/retrieval_queries-*.csv  评测问题
    （跳过 query_type == out_of_scope 的越界问题）

输出：
    data/eval/train_queries.csv  用于 build_training_pairs.py 构造训练对
    data/eval/test_queries.csv   用于 evaluate_retrieval.py 评测

用法：
    python scripts/split_dataset.py --test-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = ["query", "expected_file", "expected_topic", "query_type", "answer_keywords", "source"]


def collect_queries() -> list[dict]:
    """汇总所有 in-scope query，附带来源标记。"""
    patterns = [
        str(PROJECT_ROOT / "data" / "raw" / "**" / "questions.csv"),
        str(PROJECT_ROOT / "data" / "eval" / "retrieval_queries-*.csv"),
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat, recursive=True)))

    rows: list[dict] = []
    seen: set[str] = set()
    for f in files:
        source = Path(f).relative_to(PROJECT_ROOT).as_posix()
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                query = (row.get("query") or "").strip()
                if not query:
                    continue
                if (row.get("query_type") or "").strip() == "out_of_scope":
                    continue
                if query in seen:  # 去重：同一问题只保留一次
                    continue
                seen.add(query)
                rows.append(
                    {
                        "query": query,
                        "expected_file": (row.get("expected_file") or "").strip(),
                        "expected_topic": (row.get("expected_topic") or "").strip(),
                        "query_type": (row.get("query_type") or "").strip(),
                        "answer_keywords": (row.get("answer_keywords") or "").strip(),
                        "source": source,
                    }
                )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="按 query 划分 train/test，避免数据泄漏")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", type=Path, default=PROJECT_ROOT / "data/eval/train_queries.csv")
    parser.add_argument("--test-out", type=Path, default=PROJECT_ROOT / "data/eval/test_queries.csv")
    args = parser.parse_args()

    rows = collect_queries()
    if not rows:
        raise SystemExit("没有收集到任何 query，检查 data/raw 与 data/eval 是否有 questions/retrieval_queries 文件。")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_test = max(1, round(len(rows) * args.test_ratio))
    test_rows = rows[:n_test]
    train_rows = rows[n_test:]

    write_csv(train_rows, args.train_out)
    write_csv(test_rows, args.test_out)

    print(f"总 query（去重、in-scope）: {len(rows)}")
    print(f"训练集: {len(train_rows)} -> {args.train_out.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"测试集: {len(test_rows)} -> {args.test_out.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"随机种子: {args.seed}（固定，可复现）")


if __name__ == "__main__":
    main()
