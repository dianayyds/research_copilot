from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re


SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}


@dataclass
class ParsedAssetFile:
    title: str
    asset_type: str
    content: str
    filename: str


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def decode_text_content(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Uploaded file must be UTF-8 or GB18030 encoded text")


def uploaded_asset_title(filename: str, title: str | None = None) -> str:
    fallback = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = Path(fallback).stem or fallback
    return (title or stem).strip()


def strip_front_matter(text: str) -> str:
    normalized = normalize_newlines(text)
    if not normalized.startswith("---\n"):
        return normalized
    closing = normalized.find("\n---\n", 4)
    if closing == -1:
        return normalized
    return normalized[closing + 5 :].lstrip()


def infer_asset_type(filename: str, provided: str | None = None) -> str:
    explicit = (provided or "").strip()
    if explicit:
        return explicit
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    return "note"


def markdown_document(text: str, *, title: str, filename: str) -> str:
    body = collapse_blank_lines(strip_front_matter(text))
    if not body:
        raise ValueError("Markdown file is empty")
    lines = body.splitlines()
    first_non_empty = next((line for line in lines if line.strip()), "")
    if first_non_empty.startswith("#"):
        remainder = body[len(first_non_empty) :].lstrip()
        parts = [first_non_empty.strip(), f"来源文件：{filename}"]
        if remainder:
            parts.append(remainder)
        return collapse_blank_lines("\n\n".join(parts))
    return collapse_blank_lines(f"# {title}\n\n来源文件：{filename}\n\n{body}")


def text_document(text: str, *, title: str, filename: str) -> str:
    body = collapse_blank_lines(normalize_newlines(text))
    if not body:
        raise ValueError("Text file is empty")
    return collapse_blank_lines(f"# {title}\n\n来源文件：{filename}\n\n{body}")


def normalize_pdf_text(text: str) -> str:
    lines = [line.strip() for line in normalize_newlines(text).splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(re.sub(r"\s+", " ", line))
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def pdf_document(raw: bytes, *, title: str, filename: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        extracted = normalize_pdf_text(page.extract_text() or "")
        if extracted:
            pages.append(f"## 第{index}页\n\n{extracted}")
    if not pages:
        raise ValueError("PDF 中没有可提取的文本")
    return collapse_blank_lines(f"# {title}\n\n来源文件：{filename}\n\n" + "\n\n".join(pages))


def parse_uploaded_asset(
    *,
    filename: str,
    raw: bytes,
    title: str | None = None,
    asset_type: str | None = None,
) -> ParsedAssetFile:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("Only .txt, .md, .markdown and .pdf files are supported")
    resolved_title = uploaded_asset_title(filename, title)
    resolved_type = infer_asset_type(filename, asset_type)
    if suffix == ".pdf":
        content = pdf_document(raw, title=resolved_title, filename=filename)
    else:
        decoded = decode_text_content(raw)
        content = (
            markdown_document(decoded, title=resolved_title, filename=filename)
            if suffix in MARKDOWN_EXTENSIONS
            else text_document(decoded, title=resolved_title, filename=filename)
        )
    return ParsedAssetFile(
        title=resolved_title,
        asset_type=resolved_type,
        content=content,
        filename=filename,
    )
