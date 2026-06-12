from pathlib import Path

from src.ocr_loader import build_cache_path, merge_text_and_ocr, should_ocr_page


def test_merge_text_and_ocr_appends_marker():
    merged = merge_text_and_ocr("文本层内容", "图中公式\nOCR 文字")

    assert "文本层内容" in merged
    assert "[OCR]" in merged
    assert "图中公式" in merged


def test_merge_text_and_ocr_avoids_duplicate_text():
    assert merge_text_and_ocr("已有 OCR 文字", "OCR 文字") == "已有 OCR 文字"


def test_should_ocr_page_defaults_to_image_pages_with_limited_text(monkeypatch):
    monkeypatch.delenv("OCR_MODE", raising=False)
    monkeypatch.delenv("OCR_TEXT_THRESHOLD", raising=False)

    assert should_ocr_page("少量文本", image_count=1) is True
    assert should_ocr_page("没有图片", image_count=0) is False
    assert should_ocr_page("长文本" * 1000, image_count=1) is False


def test_build_cache_path_is_stable(tmp_path):
    pdf = tmp_path / "course.pdf"
    pdf.write_bytes(b"fake")

    first = build_cache_path(
        file_path=pdf,
        page_number=1,
        provider="tesseract",
        lang="chi_sim+eng",
        cache_dir=Path("cache"),
    )
    second = build_cache_path(
        file_path=pdf,
        page_number=1,
        provider="tesseract",
        lang="chi_sim+eng",
        cache_dir=Path("cache"),
    )

    assert first == second
    assert first.name.startswith("course_p001_")
