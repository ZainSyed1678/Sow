"""Comprehensive tests for Notion blocks, formatting, client, and safe publishing.

All external Notion API interactions are fully mocked.
"""

from unittest.mock import MagicMock

import pytest

from ai_ml_notion_researcher.models import (
    EducationalNote,
    WeeklyReviewNote,
)
from ai_ml_notion_researcher.notion.blocks import (
    MAX_TEXT_LENGTH,
    chunk_blocks,
    chunk_rich_text,
    create_paragraph,
    is_valid_notion_url,
)
from ai_ml_notion_researcher.notion.client import (
    NotionClient,
    NotionRateLimitError,
)
from ai_ml_notion_researcher.notion.formatter import (
    build_educational_note_blocks,
    build_page_properties,
    build_weekly_review_blocks,
)


def test_chunk_rich_text():
    short_text = "Hello Notion"
    chunks = chunk_rich_text(short_text)
    assert len(chunks) == 1
    assert chunks[0]["text"]["content"] == "Hello Notion"

    # Test splitting when text exceeds MAX_TEXT_LENGTH
    long_text = "A" * (MAX_TEXT_LENGTH * 2 + 50)
    chunks = chunk_rich_text(long_text)
    assert len(chunks) == 3
    assert len(chunks[0]["text"]["content"]) == MAX_TEXT_LENGTH
    assert len(chunks[1]["text"]["content"]) == MAX_TEXT_LENGTH
    assert len(chunks[2]["text"]["content"]) == 50


def test_chunk_blocks():
    blocks = [create_paragraph(f"Item {i}") for i in range(250)]
    batches = chunk_blocks(blocks, chunk_size=100)
    assert len(batches) == 3
    assert len(batches[0]) == 100
    assert len(batches[1]) == 100
    assert len(batches[2]) == 50


def test_build_educational_note_blocks(sample_educational_note: EducationalNote):
    blocks = build_educational_note_blocks(sample_educational_note)
    types = [b["type"] for b in blocks]

    # Verify key educational sections are present
    assert "callout" in types
    assert "heading_2" in types
    assert "numbered_list_item" in types
    assert "bulleted_list_item" in types
    assert "code" in types
    assert "divider" in types

    # Check that diagram block has mermaid language
    code_blocks = [b for b in blocks if b["type"] == "code"]
    assert any(cb["code"]["language"] in ("mermaid", "plain text") for cb in code_blocks)


def test_build_weekly_review_blocks():
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    review = WeeklyReviewNote(
        title="Weekly AI Review - W41",
        week_start=now,
        week_end=now,
        major_developments=["Major breakthrough 1"],
        concept_relationships=["Relationship 1"],
        key_lessons=["Lesson 1"],
        practical_exercises=["Build an agent"],
        source_links=["https://example.com"],
    )
    blocks = build_weekly_review_blocks(review)
    assert len(blocks) >= 5
    assert blocks[0]["type"] == "callout"


def test_build_page_properties_adaptive():
    schema = {
        "Title": {"type": "title"},
        "Canonical ID": {"type": "rich_text"},
        "Summary": {"type": "rich_text"},
        "Tags": {"type": "multi_select"},
        "URL": {"type": "url"},
        "Date": {"type": "date"},
    }

    props = build_page_properties(
        title="Sample Note",
        canonical_id="test-hash-123",
        summary="A short summary",
        tags=["Deep Learning", "LLM"],
        source_url="https://arxiv.org/abs/123",
        published_date="2026-10-15",
        schema=schema,
    )

    assert "Title" in props
    assert props["Title"]["title"][0]["text"]["content"] == "Sample Note"
    assert props["Canonical ID"]["rich_text"][0]["text"]["content"] == "test-hash-123"
    assert props["Summary"]["rich_text"][0]["text"]["content"] == "A short summary"
    assert props["URL"]["url"] == "https://arxiv.org/abs/123"
    assert props["Date"]["date"]["start"] == "2026-10-15"
    assert len(props["Tags"]["multi_select"]) == 2


def test_client_dry_run_mode(sample_educational_note: EducationalNote):
    mock_http = MagicMock()
    client = NotionClient(api_key="mock_secret_key", client=mock_http)

    result = client.publish_note(
        note=sample_educational_note,
        database_id="00000000000000000000000000000001",
        dry_run=True,
    )

    assert result["status"] == "dry_run_success"
    assert result["title"] == sample_educational_note.title
    assert result["blocks_count"] > 0
    # Crucial guarantee: no HTTP requests executed during dry-run
    mock_http.request.assert_not_called()


def test_client_publish_live_success(sample_educational_note: EducationalNote):
    mock_http = MagicMock()
    # 1. find_page_by_canonical_id -> no existing page
    # 2. get_database schema -> 200 OK
    # 3. create_page -> 200 OK
    mock_query_res = MagicMock(status_code=200)
    mock_query_res.json.return_value = {"results": []}

    mock_db_res = MagicMock(status_code=200)
    mock_db_res.json.return_value = {
        "title": [{"plain_text": "Concept Library"}],
        "properties": {"Name": {"type": "title"}},
    }

    mock_page_res = MagicMock(status_code=200)
    mock_page_res.json.return_value = {"id": "page_9999", "url": "https://notion.so/page_9999"}

    mock_http.request.side_effect = [mock_query_res, mock_db_res, mock_page_res]

    client = NotionClient(api_key="mock_key", client=mock_http)
    result = client.publish_note(
        note=sample_educational_note,
        database_id="mock_db_id",
        dry_run=False,
    )

    assert result["status"] == "published"
    assert result["page_id"] == "page_9999"
    assert mock_http.request.call_count == 3


def test_client_duplicate_skip(sample_educational_note: EducationalNote):
    mock_http = MagicMock()
    # Mock finding existing page with matching Canonical ID
    mock_query_res = MagicMock(status_code=200)
    mock_query_res.json.return_value = {
        "results": [{"id": "existing_page_123"}]
    }
    mock_http.request.return_value = mock_query_res

    client = NotionClient(api_key="mock_key", client=mock_http)
    result = client.publish_note(
        note=sample_educational_note,
        database_id="mock_db_id",
        dry_run=False,
    )

    assert result["status"] == "skipped_duplicate"
    assert result["page_id"] == "existing_page_123"
    # Only the query request was executed, page creation was never called
    assert mock_http.request.call_count == 1


def test_client_rate_limiting_retry():
    mock_http = MagicMock()
    mock_429 = MagicMock(status_code=429, text="Rate limited")
    mock_429.headers = {"Retry-After": "0.001"}

    mock_200 = MagicMock(status_code=200)
    mock_200.json.return_value = {"title": [{"plain_text": "Test DB"}], "properties": {}}

    mock_http.request.side_effect = [mock_429, mock_200]

    client = NotionClient(api_key="mock_key", client=mock_http, backoff_factor=0.01)
    is_valid, msg, props = client.validate_database("test_db")

    assert is_valid is True
    assert "Test DB" in msg
    assert mock_http.request.call_count == 2


def test_client_rate_limiting_exhaustion():
    mock_http = MagicMock()
    mock_429 = MagicMock(status_code=429, text="Rate limited")
    mock_429.headers = {}
    mock_http.request.return_value = mock_429

    client = NotionClient(api_key="mock_key", client=mock_http, max_retries=2, backoff_factor=0.001)
    with pytest.raises(NotionRateLimitError):
        client.get_database("test_db")


def test_client_not_found():
    mock_http = MagicMock()
    mock_404 = MagicMock(status_code=404, text="Not Found")
    mock_http.request.return_value = mock_404

    client = NotionClient(api_key="mock_key", client=mock_http)
    is_valid, msg, _ = client.validate_database("missing_db")

    assert is_valid is False
    assert "not found" in msg.lower()


def test_client_permission_denied():
    mock_http = MagicMock()
    mock_403 = MagicMock(status_code=403, text="Unauthorized integration")
    mock_http.request.return_value = mock_403

    client = NotionClient(api_key="mock_key", client=mock_http)
    is_valid, msg, _ = client.validate_database("private_db")

    assert is_valid is False
    assert "permission denied" in msg.lower()


def test_is_valid_notion_url():
    assert is_valid_notion_url("https://en.wikipedia.org/wiki/Cloud_computing") is True
    assert is_valid_notion_url("http://arxiv.org/abs/2103.00020") is True
    assert is_valid_notion_url("https://custom-entry.local/test") is True
    # Invalid schemes or malformed URIs that Notion rejects
    assert is_valid_notion_url("topic://concept_library/cloud-computing") is False
    assert is_valid_notion_url("file:///etc/passwd") is False
    assert is_valid_notion_url("cloud-computing") is False
    assert is_valid_notion_url("https://") is False
    assert is_valid_notion_url("") is False
    assert is_valid_notion_url(None) is False


def test_chunk_rich_text_url_sanitization():
    # Valid URL retains link
    valid_chunks = chunk_rich_text("Click here", link_url="https://example.com")
    assert len(valid_chunks) == 1
    assert valid_chunks[0]["text"]["link"] == {"url": "https://example.com"}

    # Invalid pseudo URL strips link so Notion does not 400
    invalid_chunks = chunk_rich_text("Topic link", link_url="topic://concept_library/test")
    assert len(invalid_chunks) == 1
    assert "link" not in invalid_chunks[0]["text"]


def test_build_educational_note_blocks_pseudo_url(sample_educational_note: EducationalNote):
    sample_educational_note.source_links = ["topic://concept_library/cloud-computing"]
    blocks = build_educational_note_blocks(sample_educational_note)

    # Find the reference block
    ref_blocks = [b for b in blocks if "cloud-computing" in str(b)]
    assert len(ref_blocks) == 1
    rich_text = ref_blocks[0]["paragraph"]["rich_text"]
    # Ensure no invalid link object was attached
    assert all("link" not in item["text"] for item in rich_text)
    assert "Reference: topic://concept_library/cloud-computing" in rich_text[0]["text"]["content"]


def test_build_page_properties_url_sanitization():
    # Invalid pseudo-url is filtered out of Notion page properties
    props_invalid = build_page_properties(
        title="Test",
        canonical_id="123",
        source_url="topic://concept_library/test",
    )
    assert "URL" not in props_invalid

    # Valid url is preserved
    props_valid = build_page_properties(
        title="Test",
        canonical_id="123",
        source_url="https://en.wikipedia.org/wiki/Test",
    )
    assert props_valid["URL"] == {"url": "https://en.wikipedia.org/wiki/Test"}

