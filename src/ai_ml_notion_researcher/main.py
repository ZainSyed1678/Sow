"""Main entrypoint for AI/ML Notion Researcher."""

import argparse
import sys
from pathlib import Path

from ai_ml_notion_researcher.config import load_config
from ai_ml_notion_researcher.logging_utils import get_logger, setup_logging

logger = get_logger("main")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="AI/ML Notion Researcher")
    parser.add_argument("--dry-run", action="store_true", help="Run without publishing to Notion")
    parser.add_argument("--topic", type=str, default=None, help="Research topic (default: ai_ml)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--json-logs", action="store_true", help="Output logs in JSON format")

    args = parser.parse_args()

    setup_logging(log_level=args.log_level, json_format=args.json_logs)
    logger.info("Initializing AI/ML Notion Researcher...")

    project_root = Path(__file__).resolve().parent.parent.parent
    config = load_config(project_root=project_root)

    if args.dry_run:
        config.pipeline.dry_run = True
    if args.topic:
        config.topic = args.topic

    logger.info(
        "Configuration loaded successfully",
        extra={
            "topic": config.topic,
            "dry_run": config.pipeline.dry_run,
            "llm_provider": config.llm.provider,
            "llm_model": config.llm.model,
            "notion_configured": config.notion.is_configured_for_publishing(),
        },
    )

    if not config.notion.is_configured_for_publishing():
        if config.pipeline.dry_run:
            logger.warning(
                "Notion integration credentials are not configured, but running in DRY RUN mode."
            )
        else:
            logger.error(
                "Cannot publish: Notion API key or Database IDs are missing. "
                "Please configure .env or pass --dry-run."
            )
            return 1

    logger.info("Phase 1 initialization complete. Foundation ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
