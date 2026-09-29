"""Tests for text extraction from uploaded files."""

import pytest

from app.services.extraction import UnsupportedFileError, extract_text


def test_extract_txt() -> None:
    # Uppercase extension and surrounding whitespace should both be handled.
    assert extract_text("notes.TXT", b"  hello world \n") == "hello world"


def test_extract_txt_ignores_bad_bytes() -> None:
    # Invalid UTF-8 bytes are dropped instead of raising.
    assert "hello" in extract_text("a.txt", b"hello \xff\xfe")


def test_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileError):
        extract_text("image.png", b"data")
