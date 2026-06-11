from src.schemas import Chunk, DocumentPage, QuizItem


def test_shared_schemas_can_be_created():
    page = DocumentPage(file_name="course.pdf", page=1, text="Transformer RAG")
    chunk = Chunk(
        chunk_id="course_pdf_p001_c001",
        file_name=page.file_name,
        page=page.page,
        text=page.text,
        concepts=["Transformer"],
    )
    quiz = QuizItem(
        question="Which concept?",
        options=["Transformer", "Other"],
        answer="Transformer",
        explanation="Mentioned in the chunk.",
        concept="Transformer",
        source_chunk_id=chunk.chunk_id,
    )
    assert quiz.source_chunk_id == chunk.chunk_id

