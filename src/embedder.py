import hashlib


def embed_text(text: str, dim: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [digest[i % len(digest)] / 255.0 for i in range(dim)]
    total = sum(values) or 1.0
    return [value / total for value in values]

