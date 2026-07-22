import pytest

from vaillant_rag.loaders import SUPPORTED_EXTENSIONS, extract_text, is_supported


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert is_supported("notes.md")
    assert not is_supported("data.csv")


def test_extract_plain_text(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Hello, plain text.", encoding="utf-8")
    assert extract_text(path) == "Hello, plain text."


def test_extract_markdown(tmp_path):
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nSome content.", encoding="utf-8")
    assert "Some content." in extract_text(path)


def test_extract_html_strips_tags_and_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><script>var x = 1;</script><style>b{}</style></head>"
        "<body><h1>Header</h1><p>Body text.</p></body></html>",
        encoding="utf-8",
    )
    text = extract_text(path)
    assert "Header" in text
    assert "Body text." in text
    assert "var x" not in text


def test_extract_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF content here.")
    doc.save(path)
    doc.close()
    assert "PDF content here." in extract_text(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "missing.txt")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(path)
