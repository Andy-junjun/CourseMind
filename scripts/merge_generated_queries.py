"""把 workflow 生成的 query 合并进各成员的 questions.csv（追加，去重，保留原有人工题）。

这是一次性的数据准备脚本。生成的 query 来自 data/training/_generated_queries.json
（由 generate-course-queries workflow 产出），按 source_file 的成员目录归属写入
data/raw/<member>/questions.csv。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED = PROJECT_ROOT / "data/training/_generated_queries.json"
FIELDS = ["query", "expected_file", "expected_topic", "query_type", "answer_keywords"]


def normalize_keywords(value: str) -> str:
    return value.replace("；", ";").strip()


def load_existing_queries(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            q = (row.get("query") or "").strip()
            if q:
                seen.add(q)
    return seen


def main() -> None:
    data = json.loads(GENERATED.read_text(encoding="utf-8"))
    queries = data["result"]["queries"]

    by_member: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        member = q["source_file"].split("/")[0]
        by_member[member].append(q)

    for member, rows in by_member.items():
        path = PROJECT_ROOT / "data" / "raw" / member / "questions.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_queries(path)
        new_rows = [r for r in rows if r["query"].strip() not in existing]

        is_new_file = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if is_new_file:
                writer.writeheader()
            for r in new_rows:
                writer.writerow(
                    {
                        "query": r["query"].strip(),
                        "expected_file": r["expected_file"].strip(),
                        "expected_topic": r["expected_topic"].strip(),
                        "query_type": r["query_type"].strip(),
                        "answer_keywords": normalize_keywords(r["answer_keywords"]),
                    }
                )
        print(f"{member}: 追加 {len(new_rows)} 题（生成 {len(rows)}，去重跳过 {len(rows) - len(new_rows)}）-> {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
