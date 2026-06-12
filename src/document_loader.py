from pathlib import Path

from src.ocr_loader import merge_text_and_ocr, ocr_pdf_page
from src.schemas import DocumentPage


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


def load_pdf(file_path: str) -> list[DocumentPage]:
    """Load PDF or text-like course material into page records.

    The public name stays `load_pdf` because it is part of the first-sprint API
    contract, but the implementation also supports plain text and Markdown.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(path)

    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return _load_text_pages(path)

    raise ValueError(
        f"Unsupported document type: {suffix or '<no extension>'}. "
        "Supported types: .pdf, .txt, .md, .markdown"
    )


def _load_pdf_pages(path: Path) -> list[DocumentPage]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing requires PyMuPDF. Install it with `pip install pymupdf`."
        ) from exc

    try:
        with fitz.open(str(path)) as doc:
            pages = []
            for index, page in enumerate(doc):
                page_number = index + 1
                text_layer = page.get_text("text").strip()
                image_count = len(page.get_images(full=True))
                ocr_text = ocr_pdf_page(
                    page,
                    file_path=path,
                    page_number=page_number,
                    text_layer=text_layer,
                    image_count=image_count,
                )
                pages.append(
                    DocumentPage(
                        file_name=path.name,
                        page=page_number,
                        text=merge_text_and_ocr(text_layer, ocr_text),
                    )
                )
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF file: {path}") from exc

    if not pages:
        raise ValueError(f"PDF contains no pages: {path}")
    return pages


def _load_text_pages(path: Path) -> list[DocumentPage]:
    text = _read_text_with_fallback(path)
    parts = text.split("\f")
    pages = [
        DocumentPage(file_name=path.name, page=index + 1, text=part.strip())
        for index, part in enumerate(parts)
    ]
    return pages or [DocumentPage(file_name=path.name, page=1, text="")]


def _read_text_with_fallback(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding if last_error else "unknown",
        last_error.object if last_error else b"",
        last_error.start if last_error else 0,
        last_error.end if last_error else 0,
        f"Could not decode text file with encodings: {', '.join(TEXT_ENCODINGS)}",
    )
