from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings
from app.vector_store import local_model_path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PAGE_MARKER_RE = re.compile(r"^<!--\s*page:(\d+)\s*-->$")
SENTENCE_RE = re.compile(r".+?(?:[。！？!?；;]+|(?<=[.!?])\s+|$)", re.S)


@dataclass(frozen=True)
class DocumentSection:
    heading: str
    body: str
    page_start: int = 0
    page_end: int = 0


@dataclass
class ChunkRecord:
    asset_id: str
    chunk_id: str
    title: str
    asset_type: str
    content: str
    source_path: str
    chunk_level: str = "child"
    parent_id: str = ""
    section_path: str = ""
    page_start: int = 0
    page_end: int = 0
    token_count: int = 0


@dataclass(frozen=True)
class TextSpan:
    text: str
    page_start: int = 0
    page_end: int = 0


@dataclass(frozen=True)
class ChunkSpan:
    text: str
    spans: tuple[TextSpan, ...]
    page_start: int = 0
    page_end: int = 0


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


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


def grobid_enabled(provider: str) -> bool:
    return provider == "grobid" or (provider == "auto" and bool(settings.grobid_api_base.strip()))


def pdf_parser_order() -> list[str]:
    provider = settings.pdf_parser_provider.strip().lower() or "auto"
    if provider == "auto":
        order = []
        if settings.grobid_api_base.strip():
            order.append("grobid")
        order.extend(["docling", "pypdf"])
        return order
    return [provider]


def tei_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def parse_grobid_tei(tei_xml: str, *, fallback_title: str, filename: str) -> str:
    root = ET.fromstring(tei_xml)
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    title = tei_text(root.find(".//tei:titleStmt/tei:title", ns)) or fallback_title
    parts = [f"# {title}", f"来源文件：{filename}"]

    abstract = [tei_text(p) for p in root.findall(".//tei:profileDesc//tei:abstract//tei:p", ns)]
    abstract = [item for item in abstract if item]
    if abstract:
        parts.extend(["## Abstract", "\n\n".join(abstract)])

    body = root.find(".//tei:text/tei:body", ns)
    if body is not None:
        for div in body.findall(".//tei:div", ns):
            heading = tei_text(div.find("./tei:head", ns))
            paragraphs = [tei_text(p) for p in div.findall("./tei:p", ns)]
            paragraphs = [item for item in paragraphs if item]
            if not paragraphs:
                continue
            if heading:
                parts.append(f"## {heading}")
            parts.append("\n\n".join(paragraphs))

    references = []
    for ref in root.findall(".//tei:listBibl//tei:biblStruct", ns):
        text = tei_text(ref)
        if text:
            references.append(text)
    if references:
        parts.extend(["## References", "\n\n".join(references)])

    return collapse_blank_lines("\n\n".join(parts))


def pdf_document_from_grobid(raw: bytes, *, title: str, filename: str) -> str:
    base = settings.grobid_api_base.strip().rstrip("/")
    if not base:
        raise ValueError("GROBID_API_BASE is required when PDF_PARSER_PROVIDER=grobid")
    response = httpx.post(
        f"{base}/api/processFulltextDocument",
        files={"input": (filename, raw, "application/pdf")},
        timeout=settings.grobid_timeout_seconds,
    )
    response.raise_for_status()
    return parse_grobid_tei(response.text, fallback_title=title, filename=filename)


def pdf_document_from_docling(raw: bytes, *, title: str, filename: str) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ModuleNotFoundError as exc:
        raise RuntimeError("docling is not installed") from exc

    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(raw)
        handle.flush()
        result = DocumentConverter().convert(handle.name)
    markdown = normalize_newlines(result.document.export_to_markdown())
    if not markdown:
        raise ValueError("Docling did not extract any text")
    if markdown.lstrip().startswith("#"):
        lines = markdown.splitlines()
        return collapse_blank_lines("\n".join([lines[0], "", f"来源文件：{filename}", "", *lines[1:]]))
    return collapse_blank_lines(f"# {title}\n\n来源文件：{filename}\n\n{markdown}")


def pdf_document_from_pypdf(raw: bytes, *, title: str, filename: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        extracted = normalize_pdf_text(page.extract_text() or "")
        if extracted:
            pages.append(f"<!-- page:{index} -->\n\n{extracted}")
    if not pages:
        raise ValueError("PDF 中没有可提取的文本")
    return collapse_blank_lines(f"# {title}\n\n来源文件：{filename}\n\n" + "\n\n".join(pages))


def pdf_document(raw: bytes, *, title: str, filename: str) -> str:
    errors: list[str] = []
    for provider in pdf_parser_order():
        try:
            if provider == "grobid":
                return pdf_document_from_grobid(raw, title=title, filename=filename)
            if provider == "docling":
                return pdf_document_from_docling(raw, title=title, filename=filename)
            if provider == "pypdf":
                return pdf_document_from_pypdf(raw, title=title, filename=filename)
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
            if settings.pdf_parser_provider.strip().lower() == provider:
                raise ValueError(f"PDF parser failed ({errors[-1]})") from exc
    raise ValueError("PDF 中没有可提取的文本" + (f"; parser errors: {' | '.join(errors)}" if errors else ""))


def slugify(text: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.strip().lower())
    return normalized.strip("-")[:72] or "chunk"


@lru_cache
def _local_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None
    path = local_model_path(settings.embedding_model, settings.embedding_model_dir)
    if not path.exists():
        return None
    try:
        return AutoTokenizer.from_pretrained(str(path), use_fast=True)
    except Exception:
        return None


def rough_tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_./-]+|[^\s]", text)


def token_count(text: str) -> int:
    tokenizer = _local_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(rough_tokens(text))


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in [item.strip() for item in normalize_newlines(text).split("\n\n") if item.strip()]:
        pieces = [item.strip() for item in SENTENCE_RE.findall(paragraph) if item.strip()]
        sentences.extend(pieces or [paragraph])
    return sentences


def split_sentence_spans(text: str) -> list[TextSpan]:
    spans: list[TextSpan] = []
    active_page = 0
    paragraph_lines: list[str] = []

    def flush() -> None:
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
        if not paragraph:
            return
        pieces = [item.strip() for item in SENTENCE_RE.findall(paragraph) if item.strip()] or [paragraph]
        spans.extend(TextSpan(piece, active_page, active_page) for piece in pieces)

    for raw_line in normalize_newlines(text).splitlines():
        stripped = raw_line.strip()
        page_match = PAGE_MARKER_RE.match(stripped)
        if page_match:
            flush()
            paragraph_lines = []
            active_page = int(page_match.group(1))
            continue
        if not stripped:
            flush()
            paragraph_lines = []
            continue
        paragraph_lines.append(raw_line)
    flush()
    return spans


def split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    tokens = rough_tokens(sentence)
    if len(tokens) <= max_tokens:
        return [sentence]
    chunks = []
    for start in range(0, len(tokens), max_tokens):
        chunks.append(" ".join(tokens[start : start + max_tokens]).strip())
    return [chunk for chunk in chunks if chunk]


def split_long_span(span: TextSpan, max_tokens: int) -> list[TextSpan]:
    pieces = split_long_sentence(span.text, max_tokens)
    return [TextSpan(piece, span.page_start, span.page_end) for piece in pieces]


def merge_sentences(sentences: list[str], max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    max_tokens = max(max_tokens, 32)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    expanded: list[str] = []
    for sentence in sentences:
        expanded.extend(split_long_sentence(sentence, max_tokens))

    for sentence in expanded:
        size = token_count(sentence)
        if current and current_tokens + size > max_tokens:
            chunks.append(" ".join(current).strip())
            if overlap_tokens:
                overlap: list[str] = []
                overlap_size = 0
                for item in reversed(current):
                    item_size = token_count(item)
                    if overlap and overlap_size + item_size > overlap_tokens:
                        break
                    overlap.insert(0, item)
                    overlap_size += item_size
                current = overlap
                current_tokens = overlap_size
            else:
                current = []
                current_tokens = 0
        current.append(sentence)
        current_tokens += size

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def merge_text_spans(spans: list[TextSpan], max_tokens: int, overlap_tokens: int = 0) -> list[ChunkSpan]:
    max_tokens = max(max_tokens, 32)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))
    expanded: list[TextSpan] = []
    for span in spans:
        expanded.extend(split_long_span(span, max_tokens))

    chunks: list[ChunkSpan] = []
    current: list[TextSpan] = []
    current_tokens = 0

    def make_chunk(items: list[TextSpan]) -> ChunkSpan:
        pages = [page for item in items for page in (item.page_start, item.page_end) if page]
        text = " ".join(item.text for item in items).strip()
        return ChunkSpan(
            text=text,
            spans=tuple(items),
            page_start=min(pages) if pages else 0,
            page_end=max(pages) if pages else 0,
        )

    for span in expanded:
        size = token_count(span.text)
        if current and current_tokens + size > max_tokens:
            chunks.append(make_chunk(current))
            if overlap_tokens:
                overlap: list[TextSpan] = []
                overlap_size = 0
                for item in reversed(current):
                    item_size = token_count(item.text)
                    if overlap and overlap_size + item_size > overlap_tokens:
                        break
                    overlap.insert(0, item)
                    overlap_size += item_size
                current = overlap
                current_tokens = overlap_size
            else:
                current = []
                current_tokens = 0
        current.append(span)
        current_tokens += size

    if current:
        chunks.append(make_chunk(current))
    return [chunk for chunk in chunks if chunk.text]


def markdown_sections(text: str) -> list[DocumentSection]:
    lines = normalize_newlines(text).splitlines()
    sections: list[DocumentSection] = []
    heading_stack: dict[int, str] = {}
    current_heading = "正文"
    current_lines: list[str] = []
    active_page = 0
    current_pages: list[int] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            pages = current_pages or ([active_page] if active_page else [])
            sections.append(
                DocumentSection(
                    heading=current_heading,
                    body=body,
                    page_start=min(pages) if pages else 0,
                    page_end=max(pages) if pages else 0,
                )
            )

    for raw_line in lines:
        stripped = raw_line.strip()
        page_match = PAGE_MARKER_RE.match(stripped)
        if page_match:
            active_page = int(page_match.group(1))
            current_lines.append(raw_line)
            continue
        match = HEADING_RE.match(stripped)
        if match:
            flush()
            current_lines = []
            current_pages = []
            level = len(match.group(1))
            heading_stack[level] = match.group(2).strip()
            for key in list(heading_stack):
                if key > level:
                    del heading_stack[key]
            current_heading = " / ".join(heading_stack[index] for index in sorted(heading_stack))
            continue
        current_lines.append(raw_line)
        if active_page and stripped:
            current_pages.append(active_page)
    flush()
    return sections or [DocumentSection("正文", text.strip())]


def numbered_chunk_id(asset_id: str, level: str, index: int) -> str:
    return f"{asset_id}-{level}-{index:05d}"


def build_asset_chunks(asset: Any) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    parent_index = 0
    child_index = 0
    for section in markdown_sections(str(asset.content)):
        section_slug = slugify(section.heading)
        source_path = f"/assets/{asset.id}#{section_slug}" if section.heading != "正文" else f"/assets/{asset.id}"
        parent_spans = merge_text_spans(
            split_sentence_spans(section.body),
            settings.asset_parent_chunk_tokens,
            overlap_tokens=0,
        )
        for parent_span in parent_spans:
            parent_index += 1
            parent_id = numbered_chunk_id(str(asset.id), "parent", parent_index)
            parent_title = str(asset.title) if section.heading == "正文" else f"{asset.title} · {section.heading}"
            parent = ChunkRecord(
                asset_id=str(asset.id),
                chunk_id=parent_id,
                title=parent_title,
                asset_type=str(asset.asset_type),
                content=parent_span.text,
                source_path=source_path,
                chunk_level="parent",
                parent_id="",
                section_path=section.heading,
                page_start=parent_span.page_start or section.page_start,
                page_end=parent_span.page_end or section.page_end,
                token_count=token_count(parent_span.text),
            )
            chunks.append(parent)
            child_spans = merge_text_spans(
                list(parent_span.spans),
                settings.asset_child_chunk_tokens,
                settings.asset_child_chunk_overlap_tokens,
            )
            for child_span in child_spans:
                child_index += 1
                chunks.append(
                    ChunkRecord(
                        asset_id=str(asset.id),
                        chunk_id=numbered_chunk_id(str(asset.id), "child", child_index),
                        title=parent_title,
                        asset_type=str(asset.asset_type),
                        content=child_span.text,
                        source_path=source_path,
                        chunk_level="child",
                        parent_id=parent_id,
                        section_path=section.heading,
                        page_start=child_span.page_start or parent.page_start,
                        page_end=child_span.page_end or parent.page_end,
                        token_count=token_count(child_span.text),
                    )
                )
    return chunks
