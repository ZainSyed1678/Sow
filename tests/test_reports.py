"""Unit tests for reports summary formatting."""

from ai_ml_notion_researcher.models import ExecutionStats
from ai_ml_notion_researcher.reports.summary import format_markdown_summary


def test_format_markdown_summary():
    stats = ExecutionStats(
        run_id="run-test-123",
        items_scored=5,
        notes_generated=3,
        notes_published=3,
        errors=["Minor warning"],
    )
    report = format_markdown_summary(stats)

    assert "run-test-123" in report
    assert "Items Scored**: 5" in report
    assert "Notes Generated**: 3" in report
    assert "Minor warning" in report
