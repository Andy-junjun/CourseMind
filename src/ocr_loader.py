import hashlib
import os
from pathlib import Path
from typing import Any


DEFAULT_OCR_CACHE_DIR = Path("data/processed/ocr_cache")
DEFAULT_OCR_LANG = "chi_sim+eng"
DEFAULT_OCR_DPI = 200


def get_ocr_provider() -> str:
    return os.getenv("OCR_PROVIDER", "none").strip().lower()


def get_ocr_language() -> str:
    return os.getenv("OCR_LANG", DEFAULT_OCR_LANG).strip()


def should_ocr_page(text: str, image_count: int) -> bool:
    mode = os.getenv("OCR_MODE", "auto").strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return False
    if mode in {"all", "always"}:
        return image_count > 0
    if mode in {"images", "image"}:
        return image_count > 0
    return image_count > 0 and len(text.strip()) < int(os.getenv("OCR_TEXT_THRESHOLD", "1200"))


def ocr_pdf_page(
    page: Any,
    *,
    file_path: Path,
    page_number: int,
    text_layer: str,
    image_count: int,
    cache_dir: Path = DEFAULT_OCR_CACHE_DIR,
) -> str:
    provider = get_ocr_provider()
    if provider in {"none", "off", ""}:
        return ""
    if not should_ocr_page(text_layer, image_count):
        return ""

    cache_path = build_cache_path(
        file_path=file_path,
        page_number=page_number,
        provider=provider,
        lang=get_ocr_language(),
        cache_dir=cache_dir,
    )
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    image_bytes = render_page_png(page)
    if provider == "tesseract":
        text = ocr_image_with_tesseract(image_bytes)
    else:
        raise ValueError(f"Unsupported OCR_PROVIDER: {provider}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def render_page_png(page: Any, dpi: int = DEFAULT_OCR_DPI) -> bytes:
    import fitz  # type: ignore

    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def ocr_image_with_tesseract(image_bytes: bytes) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image
        import io
    except ImportError as exc:
        raise RuntimeError(
            "OCR_PROVIDER=tesseract requires pytesseract and pillow. "
            "Install Python package `pytesseract` and the Tesseract executable with Chinese language data."
        ) from exc

    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image, lang=get_ocr_language()).strip()


def merge_text_and_ocr(text_layer: str, ocr_text: str) -> str:
    text_layer = text_layer.strip()
    ocr_text = normalize_ocr_text(ocr_text)
    if not ocr_text:
        return text_layer
    if not text_layer:
        return ocr_text
    if ocr_text in text_layer:
        return text_layer
    return f"{text_layer}\n\n[OCR]\n{ocr_text}"


def normalize_ocr_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def build_cache_path(
    *,
    file_path: Path,
    page_number: int,
    provider: str,
    lang: str,
    cache_dir: Path,
) -> Path:
    stat = file_path.stat()
    key = f"{file_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{page_number}:{provider}:{lang}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{file_path.stem}_p{page_number:03d}_{digest}.txt"
