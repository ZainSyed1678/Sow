"""Collector orchestrator managing multi-source collection, deduplication, and scoring."""

from ai_ml_notion_researcher.collectors.arxiv import ArxivCollector
from ai_ml_notion_researcher.collectors.base import BaseCollector
from ai_ml_notion_researcher.collectors.github import GithubCollector
from ai_ml_notion_researcher.collectors.rss import RSSCollector
from ai_ml_notion_researcher.config import AppConfig
from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import ClassifiedItem, RawItem
from ai_ml_notion_researcher.processing.scorer import score_and_classify
from ai_ml_notion_researcher.storage.sqlite_store import SQLiteStore

logger = get_logger("collectors.manager")


class CollectorManager:
    """Coordinates research collectors with error containment and dual-layer deduplication."""

    def __init__(
        self,
        config: AppConfig,
        store: SQLiteStore | None = None,
        collectors: list[BaseCollector] | None = None,
    ):
        self.config = config
        self.store = store or SQLiteStore(db_path=config.storage.sqlite_path)

        if collectors is not None:
            self.collectors = collectors
        else:
            self.collectors = []
            if config.sources.arxiv.enabled:
                self.collectors.append(
                    ArxivCollector(
                        config=config.sources.arxiv,
                        timeout_seconds=config.pipeline.request_timeout_seconds,
                        user_agent=config.pipeline.user_agent,
                    )
                )
            if config.sources.rss_feeds.enabled:
                self.collectors.append(
                    RSSCollector(
                        config=config.sources.rss_feeds,
                        timeout_seconds=config.pipeline.request_timeout_seconds,
                        user_agent=config.pipeline.user_agent,
                    )
                )
            if config.sources.github.enabled:
                self.collectors.append(
                    GithubCollector(
                        config=config.sources.github,
                        timeout_seconds=config.pipeline.request_timeout_seconds,
                        user_agent=config.pipeline.user_agent,
                    )
                )

    def run_collection(self) -> list[RawItem]:
        """Runs all collectors, catching failures per source and deduplicating in-memory."""
        raw_items: list[RawItem] = []
        seen_ids: set[str] = set()

        for collector in self.collectors:
            items = collector.safe_collect()
            for item in items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    raw_items.append(item)

        logger.info(f"Total raw items collected across all sources: {len(raw_items)}")
        return raw_items

    def collect_and_prioritize(self) -> list[ClassifiedItem]:
        """Runs the complete collection, local deduplication, and relevance scoring workflow."""
        raw_items = self.run_collection()

        # Layer 1 Deduplication: SQLite local storage filter
        unprocessed_items = self.store.filter_unprocessed(raw_items)
        logger.info(
            f"After SQLite deduplication: {len(unprocessed_items)} new items to score "
            f"(filtered {len(raw_items) - len(unprocessed_items)} previously processed)"
        )

        classified: list[ClassifiedItem] = []
        min_score = self.config.pipeline.min_relevance_score

        for item in unprocessed_items[: self.config.pipeline.max_items_to_score]:
            classified_item = score_and_classify(item, min_score=min_score)
            if classified_item:
                classified.append(classified_item)

        # Sort by relevance score descending
        classified.sort(key=lambda ci: ci.relevance_score, reverse=True)

        # Limit to configured maximum items to generate
        top_items = classified[: self.config.pipeline.max_items_to_generate]
        logger.info(
            f"Prioritized top {len(top_items)} items meeting relevance threshold >= {min_score}"
        )
        return top_items
