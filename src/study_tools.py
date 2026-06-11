from src.schemas import Chunk, QuizItem


def generate_quiz(chunks: list[Chunk], num_questions: int = 3) -> list[QuizItem]:
    selected = chunks[:num_questions] or []
    items: list[QuizItem] = []
    for index, chunk in enumerate(selected, start=1):
        concept = chunk.concepts[0] if chunk.concepts else "通用知识点"
        answer = concept
        options = list(dict.fromkeys([answer, "无关知识点", "仅界面样式", "数据库分片"]))
        items.append(
            QuizItem(
                question=f"证据片段 {index} 最相关的知识点是什么？",
                options=options,
                answer=answer,
                explanation=f"该片段内容提到或被归类到知识点：{concept}。",
                concept=concept,
                source_chunk_id=chunk.chunk_id,
            )
        )
    return items
