"""End-to-end unit tests for ResearchPipeline and dry-run guarantees."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_ml_notion_researcher.collectors.manager import CollectorManager
from ai_ml_notion_researcher.config import AppConfig
from ai_ml_notion_researcher.llm.generator import NoteGenerator
from ai_ml_notion_researcher.models import (
    ClassifiedItem,
    EducationalNote,
    NotionDatabaseType,
    RawItem,
    SourceType,
    TradeOffs,
    WeeklyReviewNote,
)
from ai_ml_notion_researcher.notion.client import NotionClient
from ai_ml_notion_researcher.pipeline import ResearchPipeline
from ai_ml_notion_researcher.storage.sqlite_store import SQLiteStore


def _make_dummy_classified(item_id: str, title: str) -> ClassifiedItem:
    raw = RawItem(
        id=item_id,
        source_type=SourceType.ARXIV,
        title=title,
        url=f"https://arxiv.org/abs/{item_id}",
        raw_text="A paper on transformers.",
    )
    return ClassifiedItem(
        item=raw,
        target_database=NotionDatabaseType.RESEARCH_PAPERS,
        relevance_score=0.90,
        topics=["Deep Learning"],
        summary="A paper on transformers.",
    )


def _make_dummy_note(canonical_id: str, title: str) -> EducationalNote:
    return EducationalNote(
        title=title,
        summary="Executive summary",
        target_database=NotionDatabaseType.RESEARCH_PAPERS,
        canonical_id=canonical_id,
        layman_explanation="Simple explanation",
        technical_explanation="Technical mechanics",
        step_by_step_mechanism=["Step 1", "Step 2"],
        real_world_use_cases=["Use case 1"],
        technologies_and_interactions=["Tech interaction"],
        trade_offs=TradeOffs(advantages=["Adv 1"], limitations=["Lim 1"], trade_offs=["TO 1"]),
        practical_exercise="Exercise 1",
        interview_takeaways=["Takeaway 1"],
        source_links=[f"https://arxiv.org/abs/{canonical_id}"],
    )


def test_pipeline_dry_run_success(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    mock_mgr = MagicMock(spec=CollectorManager)
    mock_mgr.collect_and_prioritize.return_value = [
        _make_dummy_classified("item1", "Paper 1"),
        _make_dummy_classified("item2", "Paper 2"),
    ]

    mock_gen = MagicMock(spec=NoteGenerator)
    mock_gen.generate_note.side_effect = [
        _make_dummy_note("item1", "Paper 1"),
        _make_dummy_note("item2", "Paper 2"),
    ]

    mock_notion = MagicMock(spec=NotionClient)
    mock_notion.publish_note.return_value = {
        "status": "dry_run_success",
        "canonical_id": "test",
    }

    app_cfg = AppConfig()
    pipeline = ResearchPipeline(
        config=app_cfg,
        store=store,
        collector_manager=mock_mgr,
        note_generator=mock_gen,
        notion_client=mock_notion,
    )

    stats = pipeline.run(dry_run=True)

    assert stats.status == "completed"
    assert stats.dry_run is True
    assert stats.items_scored == 2
    assert stats.notes_generated == 2
    assert stats.notes_published == 2
    assert len(stats.errors) == 0

    # Both items should be recorded in SQLite store
    assert store.is_processed("item1")
    assert store.is_processed("item2")


def test_pipeline_empty_collection(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    mock_mgr = MagicMock(spec=CollectorManager)
    mock_mgr.collect_and_prioritize.return_value = []

    pipeline = ResearchPipeline(
        config=AppConfig(),
        store=store,
        collector_manager=mock_mgr,
        note_generator=MagicMock(),
        notion_client=MagicMock(),
    )

    stats = pipeline.run(dry_run=True)
    assert stats.status == "completed"
    assert stats.items_scored == 0
    assert stats.notes_generated == 0


def test_pipeline_error_resilience(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    mock_mgr = MagicMock(spec=CollectorManager)
    mock_mgr.collect_and_prioritize.return_value = [
        _make_dummy_classified("item1", "Failing Paper"),
        _make_dummy_classified("item2", "Working Paper"),
    ]

    mock_gen = MagicMock(spec=NoteGenerator)
    # Item 1 raises generation error, Item 2 succeeds
    mock_gen.generate_note.side_effect = [
        RuntimeError("LLM hallucinated invalid schema"),
        _make_dummy_note("item2", "Working Paper"),
    ]

    mock_notion = MagicMock(spec=NotionClient)
    mock_notion.publish_note.return_value = {"status": "dry_run_success"}

    pipeline = ResearchPipeline(
        config=AppConfig(),
        store=store,
        collector_manager=mock_mgr,
        note_generator=mock_gen,
        notion_client=mock_notion,
    )

    stats = pipeline.run(dry_run=True)

    assert stats.status == "completed_with_errors"
    assert stats.notes_generated == 1
    assert stats.notes_published == 1
    assert len(stats.errors) == 1
    assert "LLM hallucinated invalid schema" in stats.errors[0]

    # Item 2 succeeded despite item 1 failure
    assert store.is_processed("item2")
    assert not store.is_processed("item1")


def test_pipeline_weekly_review(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    mock_gen = MagicMock(spec=NoteGenerator)
    mock_gen.generate_weekly_review.return_value = MagicMock(spec=WeeklyReviewNote)

    mock_notion = MagicMock(spec=NotionClient)
    mock_notion.publish_weekly_review.return_value = {"status": "dry_run_success"}

    pipeline = ResearchPipeline(
        config=AppConfig(),
        store=store,
        collector_manager=MagicMock(),
        note_generator=mock_gen,
        notion_client=mock_notion,
    )

    dummy_notes = [_make_dummy_note("id1", "Note 1"), _make_dummy_note("id2", "Note 2")]
    stats = pipeline.run_weekly_review(notes=dummy_notes, dry_run=True)

    assert stats.status == "completed"
    assert stats.notes_generated == 1
    assert stats.notes_published == 1
    mock_notion.publish_weekly_review.assert_called_once()


def test_pipeline_rate_limit_pacing(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    mock_mgr = MagicMock(spec=CollectorManager)
    mock_mgr.collect_and_prioritize.return_value = [
        _make_dummy_classified("item1", "Paper 1"),
        _make_dummy_classified("item2", "Paper 2"),
    ]

    mock_gen = MagicMock(spec=NoteGenerator)
    mock_gen.generate_note.side_effect = [
        _make_dummy_note("item1", "Paper 1"),
        _make_dummy_note("item2", "Paper 2"),
    ]

    mock_notion = MagicMock(spec=NotionClient)
    mock_notion.publish_note.return_value = {"status": "dry_run_success"}

    app_cfg = AppConfig()
    app_cfg.llm.request_delay_seconds = 0.05

    pipeline = ResearchPipeline(
        config=app_cfg,
        store=store,
        collector_manager=mock_mgr,
        note_generator=mock_gen,
        notion_client=mock_notion,
    )

    with patch("time.sleep") as mock_sleep:
        stats = pipeline.run(dry_run=True)
        assert stats.status == "completed"
        # Verify time.sleep was called with 0.05 for pacing between item 1 and item 2
        mock_sleep.assert_any_call(0.05)


def test_pipeline_custom_input_item(tmp_path: Path):
    store = SQLiteStore(db_path=tmp_path / "test.db")
    mock_gen = MagicMock(spec=NoteGenerator)
    mock_gen.generate_note.return_value = _make_dummy_note("custom-1", "Custom Robotics Paper")

    mock_notion = MagicMock(spec=NotionClient)
    mock_notion.publish_note.return_value = {"status": "dry_run_success"}

    pipeline = ResearchPipeline(
        config=AppConfig(),
        store=store,
        note_generator=mock_gen,
        notion_client=mock_notion,
    )

    stats = pipeline.run_custom_item(
        title="Custom Robotics Paper",
        url="https://arxiv.org/abs/2401.55555",
        text="A custom study on quadruped robot locomotion.",
        target_database=NotionDatabaseType.RESEARCH_PAPERS,
        dry_run=True,
    )
    assert stats.status == "completed"
    assert stats.items_scored == 1
    assert stats.notes_generated == 1
    assert stats.notes_published == 1
    assert store.is_processed("custom-1")
