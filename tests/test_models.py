"""Tests for core domain data models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_ml_notion_researcher.models import (
    ClassifiedItem,
    EducationalNote,
    ExecutionStats,
    NotionDatabaseType,
    RawItem,
    SourceType,
    WeeklyReviewNote,
)


def test_raw_item_creation(sample_raw_item: RawItem):
    assert sample_raw_item.source_type == SourceType.ARXIV
    assert "arxiv.org" in sample_raw_item.url
    assert len(sample_raw_item.id) == 16
    assert sample_raw_item.id == RawItem.make_id("arxiv", sample_raw_item.url)


def test_raw_item_make_id_deterministic():
    id1 = RawItem.make_id("rss", "https://example.com/post/1/")
    id2 = RawItem.make_id("rss", "https://example.com/post/1")
    assert id1 == id2


def test_classified_item_validation(sample_raw_item: RawItem):
    classified = ClassifiedItem(
        item=sample_raw_item,
        target_database=NotionDatabaseType.RESEARCH_PAPERS,
        relevance_score=0.92,
        topics=["attention", "transformers"],
        summary="A valuable retrospective on attention.",
        reasoning="Presents core mathematical intuition.",
    )
    assert classified.relevance_score == 0.92
    assert classified.target_database == NotionDatabaseType.RESEARCH_PAPERS


def test_classified_item_invalid_score(sample_raw_item: RawItem):
    with pytest.raises(ValidationError):
        ClassifiedItem(
            item=sample_raw_item,
            target_database=NotionDatabaseType.RESEARCH_PAPERS,
            relevance_score=1.5,  # Exceeds max 1.0
            summary="Invalid",
        )


def test_educational_note_structure(sample_educational_note: EducationalNote):
    assert sample_educational_note.title.startswith("Transformer")
    assert len(sample_educational_note.step_by_step_mechanism) == 4
    assert len(sample_educational_note.real_world_use_cases) == 3
    assert len(sample_educational_note.interview_takeaways) == 3
    assert sample_educational_note.trade_offs.advantages
    assert sample_educational_note.trade_offs.limitations
    assert sample_educational_note.trade_offs.trade_offs


def test_educational_note_validation():
    with pytest.raises(ValidationError):
        # Missing required fields
        EducationalNote(
            title="Incomplete Note",
            summary="Too short",
            target_database=NotionDatabaseType.CONCEPT_LIBRARY,
            canonical_id="test",
            layman_explanation="",
            technical_explanation="",
            step_by_step_mechanism=[],
            real_world_use_cases=[],
            technologies_and_interactions=[],
            trade_offs={"advantages": [], "limitations": [], "trade_offs": []},  # type: ignore
            practical_exercise="",
            interview_takeaways=[],
            source_links=[],
        )


def test_weekly_review_note():
    now = datetime.now(UTC)
    review = WeeklyReviewNote(
        title="AI/ML Weekly Review: Week 41, 2026",
        week_start=now,
        week_end=now,
        major_developments=["Llama 3.3 release", "vLLM speculative decoding advancements"],
        concept_relationships=["Speculative decoding connects draft models to KV-cache re-use"],
        key_lessons=["Memory bandwidth is the primary bottleneck in generative inference"],
        practical_exercises=["Benchmark speculative decoding speedup vs temperature"],
        referenced_note_titles=["Transformer Architecture"],
        source_links=["https://arxiv.org"],
    )
    assert review.target_database == NotionDatabaseType.WEEKLY_REVIEWS
    assert len(review.major_developments) == 2


def test_execution_stats():
    stats = ExecutionStats(run_id="run-2026-10-15-01")
    assert stats.status == "running"
    assert stats.dry_run is True
    stats.items_collected = 12
    stats.items_scored = 10
    stats.notes_generated = 4
    stats.status = "completed"
    assert stats.notes_generated == 4
