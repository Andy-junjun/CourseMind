from src.chunker import chunk_pages
from src.schemas import DocumentPage


def test_chunker_outputs_stable_required_fields():
    pages = [DocumentPage(file_name="course.pdf", page=1, text="Transformer RAG " * 50)]
    chunks = chunk_pages(pages, chunk_size=80, overlap=10)
    assert chunks
    first = chunks[0]
    assert first.chunk_id == "course_pdf_p001_c001"
    assert first.file_name == "course.pdf"
    assert first.page == 1
    assert first.text
    assert "Transformer" in first.concepts

