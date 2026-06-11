from src.config import get_mode


def generate_text(prompt: str) -> str:
    if get_mode() == "real":
        return "真实 LLM 接入应在 src/llm_client.py 中实现。"
    return (
        "模拟回答：根据检索到的课程证据，CourseMind 结合了 RAG 检索、原文引用、"
        "MiniRanker 神经重排序、GraphRAG-lite 上下文扩展、自动出题和 Bandit "
        "复习推荐。真实模式下可以把这里替换为大语言模型 API 调用。"
    )
