import src.keyword_extractor as keyword_extractor
from src.keyword_extractor import candidate_keywords, extract_keywords


def test_candidate_keywords_are_extracted_without_manual_vocabulary():
    keywords = candidate_keywords(
        "激活函数通过引入非线性，使多层神经网络能够拟合复杂函数。", max_candidates=8
    )

    assert keywords
    assert any("激活" in keyword or "非线性" in keyword for keyword in keywords)


def test_embedding_keyword_provider_ranks_candidates_by_chunk_similarity(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("KEYWORD_PROVIDER", "embedding")
    monkeypatch.setattr(
        keyword_extractor,
        "candidate_keywords",
        lambda text, max_candidates=32: ["激活函数", "神经网络"],
    )

    vectors = {
        "chunk": (1.0, 0.0),
        "激活函数": (0.95, 0.05),
        "神经网络": (0.2, 0.8),
    }
    monkeypatch.setattr(
        keyword_extractor,
        "embeddings_for",
        lambda texts: [vectors[text] for text in texts],
    )

    assert extract_keywords("chunk", top_k=2) == ["激活函数", "神经网络"]


def test_statistical_provider_is_used_when_embedding_provider_is_lite(monkeypatch):
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "lite")
    monkeypatch.setenv("KEYWORD_PROVIDER", "embedding")

    keywords = extract_keywords("Transformer 使用注意力机制计算 Query Key Value。", top_k=5)

    assert keywords
    assert any(keyword in keywords for keyword in ["Transformer", "Query", "Value"])
