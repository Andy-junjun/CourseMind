from src.schemas import Chunk, QuizItem


def generate_quiz(chunks: list[Chunk], num_questions: int = 3) -> list[QuizItem]:
    selected = chunks[:num_questions] or []
    items: list[QuizItem] = []
    for index, chunk in enumerate(selected, start=1):
        concept = chunk.concepts[0] if chunk.concepts else "General"
        answer = concept
        options = list(dict.fromkeys([answer, "Unrelated concept", "Only UI styling", "Database sharding"]))
        items.append(
            QuizItem(
                question=f"Which concept is most related to evidence chunk {index}?",
                options=options,
                answer=answer,
                explanation=f"The source chunk mentions or is categorized as {concept}.",
                concept=concept,
                source_chunk_id=chunk.chunk_id,
            )
        )
    return items

