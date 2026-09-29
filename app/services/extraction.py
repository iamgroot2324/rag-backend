"""Text extraction from uploaded files (.pdf and .txt)."""

import io

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".txt")


class UnsupportedFileError(ValueError):
    """Raised when the file extension is not one we can read."""


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from a .pdf or .txt upload."""
    name = filename.lower()  # make the extension check case-insensitive

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        # Some pages (e.g. scanned images) yield None, so fall back to an empty string.
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".txt"):
        # errors="ignore" drops undecodable bytes instead of failing the upload.
        return data.decode("utf-8", errors="ignore").strip()

    raise UnsupportedFileError("Only .pdf and .txt files are supported")
