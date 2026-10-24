"""Main entrypoint for AI/ML Notion Researcher."""

import argparse
import re
import sys
import urllib.parse
from pathlib import Path

from ai_ml_notion_researcher.config import load_config
from ai_ml_notion_researcher.logging_utils import get_logger, setup_logging
from ai_ml_notion_researcher.models import NotionDatabaseType
from ai_ml_notion_researcher.pipeline import ResearchPipeline

logger = get_logger("main")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="AI/ML Notion Researcher")
    parser.add_argument("--dry-run", action="store_true", help="Simulate pipeline without writing to Notion")
    parser.add_argument("--live", action="store_true", help="Publish directly to Notion (requires valid .env)")
    parser.add_argument("--weekly", action="store_true", help="Run weekly review synthesis instead of daily research")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (default: ai_ml)")
    parser.add_argument("--min-score", type=float, default=None, help="Minimum relevance score threshold (0.0 to 1.0)")
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Directly generate a concept reference note for a topic (no research paper required)",
    )
    parser.add_argument("--custom-url", type=str, default=None, help="Directly generate a note for a custom URL")
    parser.add_argument("--custom-title", type=str, default=None, help="Title for the custom input item")
    parser.add_argument("--custom-text", type=str, default="", help="Optional abstract or text for custom input")
    parser.add_argument(
        "--custom-db",
        type=str,
        default=None,
        choices=[db.value for db in NotionDatabaseType],
        help="Target database for custom input",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--json-logs", action="store_true", help="Output logs in JSON format")

    args = parser.parse_args()

    setup_logging(log_level=args.log_level, json_format=args.json_logs)
    logger.info("Initializing AI/ML Notion Researcher...")

    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(project_root=project_root)

    if args.topic:
        config.topic = args.topic

    if args.min_score is not None:
        config.pipeline.min_relevance_score = args.min_score

    # Determine dry-run status:
    # 1. If --dry-run is passed -> True
    # 2. If --live is passed -> False
    # 3. Otherwise -> defaults to config.pipeline.dry_run
    if args.dry_run:
        dry_run = True
    elif args.live:
        dry_run = False
    else:
        dry_run = config.pipeline.dry_run

    config.pipeline.dry_run = dry_run

    # If attempting live run, verify required credentials
    if not dry_run:
        if not config.notion.is_configured_for_publishing():
            logger.error(
                "Cannot execute live publication: Notion API key or Database IDs are missing. "
                "Please configure .env or run with --dry-run."
            )
            return 1
        if not config.llm.api_key or not config.llm.api_key.get_secret_value().strip():
            logger.error(
                "Cannot execute: GROQ_API_KEY is missing from environment. "
                "Please configure .env."
            )
            return 1

    pipeline = ResearchPipeline(config=config)

    if args.concept:
        concept_title = args.concept.strip()
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", concept_title.lower()).strip("-")
        wiki_title = urllib.parse.quote(concept_title.replace(" ", "_"))
        url = args.custom_url or f"https://en.wikipedia.org/wiki/{wiki_title}"
        target_db = NotionDatabaseType(args.custom_db) if args.custom_db else NotionDatabaseType.CONCEPT_LIBRARY
        logger.info(f"Generating concept reference guide for topic '{concept_title}'...")
        stats = pipeline.run_custom_item(
            title=concept_title,
            url=url,
            text=args.custom_text or "",
            target_database=target_db,
            dry_run=dry_run,
        )
    elif args.custom_url or args.custom_title:
        custom_url = (args.custom_url or "").strip()
        if custom_url and not (custom_url.startswith("http://") or custom_url.startswith("https://")):
            custom_url = f"https://{custom_url}"
        title = args.custom_title or custom_url or "Custom Topic"
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")
        url = custom_url or f"https://custom-entry.local/{slug}"
        logger.info(f"Processing direct custom input note for '{title}'...")
        target_db = NotionDatabaseType(args.custom_db) if args.custom_db else None
        stats = pipeline.run_custom_item(
            title=title,
            url=url,
            text=args.custom_text,
            target_database=target_db,
            dry_run=dry_run,
        )
    elif args.weekly:
        logger.info("Starting Weekly Review synthesis...")
        stats = pipeline.run_weekly_review(dry_run=dry_run)
    else:
        logger.info(
            f"Starting Daily Research run (topic: {config.topic}, "
            f"min_relevance_score: {config.pipeline.min_relevance_score:.2f})..."
        )
        stats = pipeline.run(dry_run=dry_run)

    print("\n" + "=" * 60)
    print("Execution Run Summary:")
    print(f"Run ID:            {stats.run_id}")
    print(f"Status:            {stats.status}")
    print(f"Dry Run:           {stats.dry_run}")
    print(f"Items Scored:      {stats.items_scored}")
    print(f"Notes Generated:   {stats.notes_generated}")
    print(f"Notes Published:   {stats.notes_published}")
    print(f"Errors Encountered:{len(stats.errors)}")
    print("=" * 60 + "\n")

    return 0 if stats.status in ("completed", "completed_no_content") else 1


if __name__ == "__main__":
    sys.exit(main())
