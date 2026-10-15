"""Complete autonomous research and publishing pipeline with dry-run and error isolation."""

import uuid
from datetime import UTC, datetime, timedelta

from ai_ml_notion_researcher.collectors.manager import CollectorManager
from ai_ml_notion_researcher.config import AppConfig, load_config
from ai_ml_notion_researcher.llm.generator import NoteGenerator
from ai_ml_notion_researcher.llm.groq_client import GroqClient
from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import (
    EducationalNote,
    ExecutionStats,
    NotionDatabaseType,
)
from ai_ml_notion_researcher.notion.client import NotionClient
from ai_ml_notion_researcher.storage.sqlite_store import SQLiteStore

logger = get_logger("pipeline")


class ResearchPipeline:
    """Orchestrates research collection, relevance scoring, LLM note generation, and Notion publishing."""

    def __init__(
        self,
        config: AppConfig | None = None,
        store: SQLiteStore | None = None,
        collector_manager: CollectorManager | None = None,
        note_generator: NoteGenerator | None = None,
        notion_client: NotionClient | None = None,
    ):
        self.config = config or load_config()
        self.store = store or SQLiteStore(db_path=self.config.storage.sqlite_path)
        self.collector_manager = collector_manager or CollectorManager(
            config=self.config,
            store=self.store,
        )

        if note_generator is not None:
            self.note_generator = note_generator
        else:
            llm_client = GroqClient(config=self.config.llm)
            self.note_generator = NoteGenerator(client=llm_client)

        if notion_client is not None:
            self.notion_client = notion_client
        else:
            api_key = self.config.notion.api_key.get_secret_value() if self.config.notion.api_key else ""
            self.notion_client = NotionClient(
                api_key=api_key,
                api_version=self.config.notion.api_version,
                base_url=self.config.notion.base_url,
                timeout_seconds=self.config.pipeline.request_timeout_seconds,
            )

    def run(self, dry_run: bool | None = None) -> ExecutionStats:
        """Executes the daily research pipeline end-to-end with bounded error containment."""
        effective_dry_run = self.config.pipeline.dry_run if dry_run is None else dry_run
        run_id = f"daily-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        stats = ExecutionStats(
            run_id=run_id,
            started_at=datetime.now(UTC),
            dry_run=effective_dry_run,
        )

        logger.info(
            f"=== Starting Research Pipeline [{run_id}] | Dry-Run: {effective_dry_run} ==="
        )

        # 1. Collection & Scoring
        try:
            prioritized_items = self.collector_manager.collect_and_prioritize()
            stats.items_scored = len(prioritized_items)
        except Exception as exc:
            err_msg = f"Collector manager failed: {exc}"
            logger.error(err_msg, exc_info=True)
            stats.errors.append(err_msg)
            stats.status = "failed"
            stats.completed_at = datetime.now(UTC)
            return stats

        if not prioritized_items:
            logger.info("No new valuable items found meeting the relevance threshold.")
            stats.status = "completed"
            stats.completed_at = datetime.now(UTC)
            return stats

        generated_notes: list[EducationalNote] = []

        # 2. LLM Note Synthesis (Error isolated per item)
        for classified in prioritized_items:
            try:
                note = self.note_generator.generate_note(classified)
                generated_notes.append(note)
                stats.notes_generated += 1
            except Exception as exc:
                err_msg = f"Failed to generate note for '{classified.item.title}': {exc}"
                logger.error(err_msg, exc_info=True)
                stats.errors.append(err_msg)
                continue

        # 3. Notion Publication (Error isolated per note)
        for note in generated_notes:
            db_id = self.config.notion.database_ids.get(note.target_database, "")
            if not db_id and not effective_dry_run:
                err_msg = f"Missing database ID for target '{note.target_database.value}'"
                logger.error(err_msg)
                stats.errors.append(err_msg)
                continue

            try:
                pub_res = self.notion_client.publish_note(
                    note=note,
                    database_id=db_id,
                    dry_run=effective_dry_run,
                )

                if pub_res.get("status") in ("published", "dry_run_success", "skipped_duplicate"):
                    stats.notes_published += 1
                    # Record in local SQLite storage so subsequent runs don't re-process
                    self.store.mark_processed(
                        item_id=note.canonical_id,
                        source_type=note.target_database.value,
                        canonical_url=note.source_links[0] if note.source_links else "",
                        title=note.title,
                        target_database=note.target_database.value,
                        status=pub_res.get("status", "unknown"),
                    )
            except Exception as exc:
                err_msg = f"Failed to publish note '{note.title}': {exc}"
                logger.error(err_msg, exc_info=True)
                stats.errors.append(err_msg)
                continue

        stats.completed_at = datetime.now(UTC)
        stats.status = "completed" if not stats.errors else "completed_with_errors"

        duration = (stats.completed_at - stats.started_at).total_seconds()
        logger.info(
            f"=== Pipeline Finished [{run_id}] in {duration:.1f}s | "
            f"Scored: {stats.items_scored}, Generated: {stats.notes_generated}, "
            f"Published: {stats.notes_published}, Errors: {len(stats.errors)} ==="
        )
        return stats

    def run_weekly_review(
        self,
        notes: list[EducationalNote] | None = None,
        dry_run: bool | None = None,
    ) -> ExecutionStats:
        """Synthesizes recent developments into a Weekly Review note published to Notion."""
        effective_dry_run = self.config.pipeline.dry_run if dry_run is None else dry_run
        run_id = f"weekly-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

        stats = ExecutionStats(
            run_id=run_id,
            started_at=datetime.now(UTC),
            dry_run=effective_dry_run,
        )

        now = datetime.now(UTC)
        week_start = now - timedelta(days=7)

        # If notes not provided in-memory, generate a synthesis from current prioritized research
        if not notes:
            prioritized = self.collector_manager.collect_and_prioritize()
            notes = []
            for item in prioritized[:4]:
                try:
                    notes.append(self.note_generator.generate_note(item))
                except Exception as exc:
                    logger.warning(f"Could not generate note during weekly review: {exc}")

        if not notes:
            logger.info("Insufficient material to produce weekly review.")
            stats.status = "completed_no_content"
            stats.completed_at = datetime.now(UTC)
            return stats

        try:
            weekly_note = self.note_generator.generate_weekly_review(
                notes=notes,
                week_start=week_start,
                week_end=now,
            )
            stats.notes_generated = 1

            db_id = self.config.notion.database_ids.get(NotionDatabaseType.WEEKLY_REVIEWS, "")
            res = self.notion_client.publish_weekly_review(
                review=weekly_note,
                database_id=db_id,
                dry_run=effective_dry_run,
            )
            if res.get("status") in ("published", "dry_run_success"):
                stats.notes_published = 1

            stats.status = "completed"
        except Exception as exc:
            err_msg = f"Failed to synthesize weekly review: {exc}"
            logger.error(err_msg, exc_info=True)
            stats.errors.append(err_msg)
            stats.status = "failed"

        stats.completed_at = datetime.now(UTC)
        return stats
