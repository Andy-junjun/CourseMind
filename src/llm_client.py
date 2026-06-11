from src.config import get_mode


def generate_text(prompt: str) -> str:
    if get_mode() == "real":
        return "Real LLM integration should be implemented in src/llm_client.py."
    return (
        "Mock answer: based on the retrieved course evidence, CourseMind combines "
        "RAG retrieval, citation display, MiniRanker reranking, GraphRAG-lite context "
        "expansion, quiz generation, and Bandit review recommendation."
    )

