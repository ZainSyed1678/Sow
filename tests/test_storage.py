"""Unit tests for SQLite storage and deduplication."""

from pathlib import Path

from ai_ml_notion_researcher.models import RawItem, SourceType
from ai_ml_notion_researcher.storage.sqlite_store import SQLiteStore


def test_sqlite_store_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_researcher.db"
    store = SQLiteStore(db_path=db_file)

    assert store.get_total_processed_count() == 0
    assert not store.is_processed("item_1")

    # Mark item 1 as processed
    store.mark_processed(
        item_id="item_1",
        source_type="arxiv",
        canonical_url="https://arxiv.org/abs/2301.00001",
        title="Attention Paper",
        target_database="research_papers",
    )

    assert store.is_processed("item_1")
    assert store.get_total_processed_count() == 1


def test_sqlite_filter_unprocessed(tmp_path: Path):
    db_file = tmp_path / "test_filter.db"
    store = SQLiteStore(db_path=db_file)

    store.mark_processed(
        item_id="item_already_seen",
        source_type="rss",
        canonical_url="https://example.com/post1",
        title="Seen Post",
        target_database="daily_briefings",
    )

    item1 = RawItem(
        id="item_already_seen",
        source_type=SourceType.RSS,
        title="Seen Post",
        url="https://example.com/post1",
    )
    item2 = RawItem(
        id="item_new",
        source_type=SourceType.RSS,
        title="Brand New Post",
        url="https://example.com/post2",
    )

    filtered = store.filter_unprocessed([item1, item2])
    assert len(filtered) == 1
    assert filtered[0].id == "item_new"
