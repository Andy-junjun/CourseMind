from pathlib import Path

from src.schemas import DocumentPage


def load_pdf(file_path: str) -> list[DocumentPage]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)
    if path.suffix.lower() == ".pdf":
        try:
            import fitz  # type: ignore

            doc = fitz.open(str(path))
            return [
                DocumentPage(file_name=path.name, page=i + 1, text=page.get_text().strip())
                for i, page in enumerate(doc)
            ]
        except ImportError:
            return [
                DocumentPage(
                    file_name=path.name,
                    page=1,
                    text="PDF parsing requires pymupdf in real mode. This fallback keeps the app runnable.",
                )
            ]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [DocumentPage(file_name=path.name, page=1, text=text)]

