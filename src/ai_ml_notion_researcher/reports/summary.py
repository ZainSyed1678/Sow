"""Report generation package for execution summaries."""

from ai_ml_notion_researcher.models import ExecutionStats


def format_markdown_summary(stats: ExecutionStats) -> str:
    """Formats ExecutionStats as a readable Markdown report."""
    duration = "N/A"
    if stats.completed_at:
        duration = f"{(stats.completed_at - stats.started_at).total_seconds():.1f}s"

    lines = [
        f"## Research Execution Summary: `{stats.run_id}`",
        f"- **Status**: `{stats.status}`",
        f"- **Dry Run**: `{stats.dry_run}`",
        f"- **Duration**: {duration}",
        f"- **Items Scored**: {stats.items_scored}",
        f"- **Notes Generated**: {stats.notes_generated}",
        f"- **Notes Published**: {stats.notes_published}",
    ]

    if stats.errors:
        lines.append("\n### Errors Encountered:")
        for err in stats.errors:
            lines.append(f"- {err}")

    return "\n".join(lines)
