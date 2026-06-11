import pytest

from src.document_loader import load_pdf


def test_load_utf8_text_file(tmp_path):
    path = tmp_path / "course.txt"
    path.write_text("中文课程资料\nRAG 检索增强生成", encoding="utf-8")

    pages = load_pdf(str(path))

    assert len(pages) == 1
    assert pages[0].file_name == "course.txt"
    assert pages[0].page == 1
    assert "检索增强生成" in pages[0].text


def test_load_gb18030_text_file(tmp_path):
    path = tmp_path / "course.md"
    path.write_bytes("中文编码测试".encode("gb18030"))

    pages = load_pdf(str(path))

    assert pages[0].text == "中文编码测试"


def test_text_file_form_feed_splits_pages(tmp_path):
    path = tmp_path / "course.txt"
    path.write_text("第一页\f第二页", encoding="utf-8")

    pages = load_pdf(str(path))

    assert [page.page for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["第一页", "第二页"]


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(str(tmp_path / "missing.pdf"))


def test_unsupported_extension_raises_value_error(tmp_path):
    path = tmp_path / "course.docx"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_pdf(str(path))


def test_load_pdf_when_pymupdf_is_available(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "course.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "CourseMind PDF parser")
    doc.save(str(path))
    doc.close()

    pages = load_pdf(str(path))

    assert len(pages) == 1
    assert pages[0].file_name == "course.pdf"
    assert pages[0].page == 1
    assert "CourseMind" in pages[0].text

