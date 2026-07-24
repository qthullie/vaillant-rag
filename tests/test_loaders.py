import pytest

from vaillant_rag.loaders import (
    SUPPORTED_EXTENSIONS,
    extract_text,
    is_supported,
)


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert is_supported("notes.md")
    assert is_supported("report.DOCX")
    assert not is_supported("data.csv")
    assert not is_supported("legacy.doc")


def test_extract_plain_text(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Hello, plain text.", encoding="utf-8")
    assert extract_text(path) == "Hello, plain text."


def test_extract_md_input(tmp_path):
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


def test_extract_docx_headings_paragraphs_tables(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "report.docx"
    document = docx.Document()
    document.add_heading("Quarterly Report", level=1)
    document.add_paragraph("Revenue grew by 12 percent.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Region"
    table.rows[0].cells[1].text = "Total"
    document.save(str(path))

    text = extract_text(path)
    assert "# Quarterly Report" in text
    assert "Revenue grew by 12 percent." in text
    assert "| Region | Total |" in text


def test_extract_xlsx_sheets_as_tables(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "inventory.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Stock"
    sheet.append(["item", "qty"])
    sheet.append(["bolt", 42])
    sheet.append([None, None])  # empty row is dropped
    workbook.save(str(path))

    text = extract_text(path)
    assert "## Stock" in text
    assert "| item | qty |" in text
    assert "| bolt | 42 |" in text


def test_extract_html_strips_tags(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><body><h2>Chapter</h2><p>Body text.</p></body></html>",
        encoding="utf-8",
    )
    text = extract_text(path)
    assert "Chapter" in text
    assert "Body text." in text


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_text(tmp_path / "missing.txt")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(path)
