import json
import sys
import types

from scripts.build_training_pairs import (
    build_pairs,
    find_query_files,
    infer_source_prefix,
    load_retrieval_queries,
    save_pairs,
)
from scripts.train_embedding import dry_run, load_pairs
from src.chunker import chunk_pages
from src.embedder import embed_text, get_embedding_model, get_embedding_model_path
from src.schemas import DocumentPage
from src.vector_store import save_chunks


def test_sentence_transformers_provider_prefers_model_path(monkeypatch):
    loaded = {}

    class FakeVector:
        def tolist(self):
            return [0.6, 0.8]

    class FakeModel:
        def __init__(self, model_path):
            loaded["model_path"] = model_path

        def encode(self, text, normalize_embeddings=True):
            loaded["text"] = text
            loaded["normalize"] = normalize_embeddings
            return FakeVector()

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setenv("COURSEMIND_MODE", "real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "models/embedding/finetuned")
    get_embedding_model.cache_clear()

    vector = embed_text("Transformer 注意力机制")

    assert get_embedding_model_path() == "models/embedding/finetuned"
    assert loaded == {
        "model_path": "models/embedding/finetuned",
        "text": "Transformer 注意力机制",
        "normalize": True,
    }
    assert vector == [0.6, 0.8]


def test_build_training_pairs_from_retrieval_queries(tmp_path):
    chunks = chunk_pages(
        [
            DocumentPage(
                file_name="cnn.md",
                page=1,
                text="CNN 通过卷积核提取局部特征，并使用权值共享减少参数。",
            ),
            DocumentPage(
                file_name="rnn.md",
                page=1,
                text="RNN 使用隐状态处理序列数据，适合时间序列建模。",
            ),
            DocumentPage(
                file_name="transformer.md",
                page=1,
                text="Transformer 使用 Query Key Value 计算注意力权重。",
            ),
        ],
        chunk_size=80,
        overlap=10,
    )
    chunks_path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, chunks_path)
    query_path = tmp_path / "retrieval_queries.csv"
    query_path.write_text(
        "query,expected_file,expected_topic,query_type,answer_keywords\n"
        "CNN 的卷积核有什么作用？,cnn.md,CNN,fact,卷积核;局部特征\n"
        "这个资料里有没有讲世界杯？,cnn.md,资料外,out_of_scope,\n",
        encoding="utf-8",
    )

    queries = load_retrieval_queries([query_path])
    pairs = build_pairs(queries, chunks_path=chunks_path, target_count=2, negatives_per_query=2)

    assert len(queries) == 1
    assert len(pairs) == 2
    assert all(pair.positive_file == "cnn.md" for pair in pairs)
    assert all(pair.negative_file != "cnn.md" for pair in pairs)


def test_find_query_files_recurses_nested_raw_dirs(tmp_path):
    eval_dir = tmp_path / "data" / "eval"
    raw_dir = tmp_path / "data" / "raw"
    nested = raw_dir / "public" / "deep_learning_core"
    nested.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (nested / "questions.csv").write_text(
        "query,expected_file,expected_topic,query_type,answer_keywords\n",
        encoding="utf-8",
    )
    (eval_dir / "retrieval_queries_demo.csv").write_text(
        "query,expected_file,expected_topic,query_type,answer_keywords\n",
        encoding="utf-8",
    )

    paths = find_query_files(eval_dir, raw_dir)

    assert eval_dir / "retrieval_queries_demo.csv" in paths
    assert nested / "questions.csv" in paths
    assert infer_source_prefix(nested / "questions.csv") == "public/deep_learning_core"


def test_training_pairs_save_and_dry_run(tmp_path):
    chunks = chunk_pages(
        [
            DocumentPage(file_name="a.md", page=1, text="LSTM 包含遗忘门和输入门。"),
            DocumentPage(file_name="b.md", page=1, text="CNN 包含卷积和池化。"),
        ]
    )
    chunks_path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, chunks_path)
    query_path = tmp_path / "retrieval_queries.csv"
    query_path.write_text(
        "query,expected_file,expected_topic,query_type,answer_keywords\n"
        "LSTM 有哪些门控？,a.md,LSTM,fact,遗忘门;输入门\n",
        encoding="utf-8",
    )
    pairs = build_pairs(
        load_retrieval_queries([query_path]),
        chunks_path=chunks_path,
        target_count=1,
        negatives_per_query=1,
    )
    pairs_path = tmp_path / "embedding_pairs.jsonl"
    output_dir = tmp_path / "finetuned"

    save_pairs(pairs, pairs_path)
    loaded_pairs = load_pairs(pairs_path)
    dry_run(loaded_pairs, output_dir, "BAAI/bge-small-zh-v1.5")

    manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "dry_run"
    assert manifest["pair_count"] == 1
