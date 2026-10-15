"""Notion block construction utilities with payload sanitization and length limits."""

from typing import Any

# Notion enforces a 2000-character maximum per rich_text item
MAX_TEXT_LENGTH = 1900
# Notion enforces a 100-block maximum per page creation or block append request
MAX_BLOCKS_PER_REQUEST = 100


def chunk_rich_text(text: str, link_url: str | None = None) -> list[dict[str, Any]]:
    """Splits string into chunks conforming to Notion's 2000 character rich text limit."""
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + MAX_TEXT_LENGTH]
        text_obj: dict[str, Any] = {"content": chunk}
        if link_url:
            text_obj["link"] = {"url": link_url}

        chunks.append({
            "type": "text",
            "text": text_obj,
        })
        start += MAX_TEXT_LENGTH

    return chunks


def create_heading_1(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_1",
        "heading_1": {
            "rich_text": chunk_rich_text(text),
        },
    }


def create_heading_2(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": chunk_rich_text(text),
        },
    }


def create_heading_3(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {
            "rich_text": chunk_rich_text(text),
        },
    }


def create_paragraph(text: str, link_url: str | None = None) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": chunk_rich_text(text, link_url=link_url),
        },
    }


def create_bullet_item(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {
            "rich_text": chunk_rich_text(text),
        },
    }


def create_numbered_item(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {
            "rich_text": chunk_rich_text(text),
        },
    }


def create_callout(text: str, emoji: str = "💡") -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": emoji},
            "rich_text": chunk_rich_text(text),
        },
    }


def create_code_block(code: str, language: str = "plain text") -> dict[str, Any]:
    # Normalize language string for Notion API
    lang = language.lower()
    valid_languages = {
        "python",
        "javascript",
        "typescript",
        "bash",
        "shell",
        "mermaid",
        "json",
        "yaml",
        "markdown",
        "sql",
        "html",
        "css",
        "c++",
        "rust",
        "go",
        "plain text",
    }
    if lang not in valid_languages:
        lang = "plain text"

    return {
        "object": "block",
        "type": "code",
        "code": {
            "language": lang,
            "rich_text": chunk_rich_text(code),
        },
    }


def create_divider() -> dict[str, Any]:
    return {
        "object": "block",
        "type": "divider",
        "divider": {},
    }


def chunk_blocks(blocks: list[dict[str, Any]], chunk_size: int = MAX_BLOCKS_PER_REQUEST) -> list[list[dict[str, Any]]]:
    """Batches blocks into groups satisfying Notion's 100-block request limit."""
    return [blocks[i : i + chunk_size] for i in range(0, len(blocks), chunk_size)]
