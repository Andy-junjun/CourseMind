from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.schemas import RankedChunk


def citation_rows(evidence: list[RankedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "文件": item.chunk.file_name,
            "页码": item.chunk.page,
            "chunk_id": item.chunk.chunk_id,
            "ranker": round(item.ranker_score, 3),
        }
        for index, item in enumerate(evidence)
    ]


def retrieval_rows(
    ranked: list[RankedChunk],
    seed_ids: set[str],
    graph_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "来源": "原始检索" if item.chunk.chunk_id in seed_ids else "GraphRAG扩展",
            "chunk_id": item.chunk.chunk_id,
            "文件": item.chunk.file_name,
            "页码": item.chunk.page,
            "dense": round(item.dense_score, 3),
            "bm25": round(item.bm25_score, 3),
            "graph": round(item.graph_score, 3),
            "ranker": round(item.ranker_score, 3),
            "命中依据": hit_reason(item, seed_ids, graph_rows),
            "片段预览": preview_text(item.chunk.text),
        }
        for index, item in enumerate(ranked)
    ]


def graph_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["扩展chunk"]), str(row["关系"]), normalize_reason(str(row["原因"])))
        current = grouped.setdefault(
            key,
            {
                "扩展chunk": row["扩展chunk"],
                "关系": row["关系"],
                "分数": row["分数"],
                "原因": normalize_reason(str(row["原因"])),
                "文件": row["文件"],
                "页码": row["页码"],
                "seed数量": 0,
                "seed示例": [],
            },
        )
        current["分数"] = max(float(current["分数"]), float(row["分数"]))
        current["seed数量"] += 1
        if len(current["seed示例"]) < 3:
            current["seed示例"].append(row["种子chunk"])

    return sorted(
        grouped.values(),
        key=lambda row: (str(row["扩展chunk"]), -float(row["分数"]), str(row["关系"])),
    )


def graph_reason(chunk_id: str, rows: list[dict[str, Any]], limit: int = 3) -> str:
    reasons_by_relation: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["扩展chunk"] != chunk_id:
            continue
        relation = str(row["关系"])
        reason = normalize_reason(str(row["原因"]))
        if reason not in reasons_by_relation[relation]:
            reasons_by_relation[relation].append(reason)

    reasons: list[str] = []
    for relation in ("same_page", "same_chapter", "adjacent", "shared_concept"):
        reasons.extend(reasons_by_relation.get(relation, []))
    return "；".join(reasons[:limit])


def hit_reason(
    item: RankedChunk,
    seed_ids: set[str],
    graph_rows: list[dict[str, Any]],
) -> str:
    if item.chunk.chunk_id in seed_ids:
        return "原始向量/BM25检索命中"
    return graph_reason(item.chunk.chunk_id, graph_rows) or "GraphRAG关系扩展"


def normalize_reason(reason: str) -> str:
    prefix = "共享知识点："
    if reason.startswith(prefix):
        concepts = sorted({part.strip() for part in reason[len(prefix) :].split("、") if part.strip()})
        return prefix + "、".join(concepts)
    return reason


def preview_text(text: str, max_chars: int = 120) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 1] + "…"
