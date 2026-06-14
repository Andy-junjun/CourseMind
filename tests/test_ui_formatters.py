from src.schemas import Chunk, RankedChunk
from src.ui_formatters import graph_reason, graph_summary_rows, retrieval_rows


def make_ranked(chunk_id: str, text: str = "正文内容") -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            file_name="course.md",
            page=1,
            text=text,
            concepts=["深度学习"],
        ),
        dense_score=0.7,
        bm25_score=0.4,
        graph_score=0.8,
        ranker_score=0.9,
    )


def test_graph_reason_deduplicates_repeated_shared_concepts():
    rows = [
        {
            "扩展chunk": "c2",
            "种子chunk": "s1",
            "关系": "shared_concept",
            "分数": 0.5,
            "原因": "共享知识点：深度学习",
            "文件": "course.md",
            "页码": 1,
        },
        {
            "扩展chunk": "c2",
            "种子chunk": "s2",
            "关系": "shared_concept",
            "分数": 0.5,
            "原因": "共享知识点：深度学习",
            "文件": "course.md",
            "页码": 1,
        },
        {
            "扩展chunk": "c2",
            "种子chunk": "s1",
            "关系": "same_page",
            "分数": 0.3,
            "原因": "与种子 chunk s1 位于同一页 p.1",
            "文件": "course.md",
            "页码": 1,
        },
    ]

    reason = graph_reason("c2", rows)

    assert reason.count("共享知识点：深度学习") == 1
    assert "位于同一页" in reason


def test_graph_summary_rows_aggregates_duplicate_reasons():
    rows = [
        {
            "扩展chunk": "c2",
            "种子chunk": "s1",
            "关系": "shared_concept",
            "分数": 0.5,
            "原因": "共享知识点：深度学习",
            "文件": "course.md",
            "页码": 1,
        },
        {
            "扩展chunk": "c2",
            "种子chunk": "s2",
            "关系": "shared_concept",
            "分数": 0.5,
            "原因": "共享知识点：深度学习",
            "文件": "course.md",
            "页码": 1,
        },
    ]

    summary = graph_summary_rows(rows)

    assert len(summary) == 1
    assert summary[0]["seed数量"] == 2
    assert summary[0]["seed示例"] == ["s1", "s2"]


def test_retrieval_rows_describe_seed_hits_without_duplicate_text_fields():
    rows = retrieval_rows([make_ranked("c1")], {"c1"}, [])

    assert rows[0]["命中依据"] == "原始向量/BM25检索命中"
    assert "片段预览" in rows[0]
