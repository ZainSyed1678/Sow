"""Abstract base class for research collectors."""

from abc import ABC, abstractmethod

import httpx

from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import RawItem

logger = get_logger("collectors.base")


class BaseCollector(ABC):
    """Abstract collector enforcing safe execution and error isolation."""

    def __init__(
        self,
        name: str,
        timeout_seconds: int = 30,
        user_agent: str = "AIMLNotionResearcher/0.1.0",
        client: httpx.Client | None = None,
    ):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self._client = client

    def get_http_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )

    def safe_collect(self) -> list[RawItem]:
        """Runs collection with bounded error containment.

        Guarantees that a collector failure will not abort the pipeline.
        """
        try:
            logger.info(f"Starting collection from {self.name}...")
            items = self.collect()
            logger.info(f"Successfully collected {len(items)} items from {self.name}")
            return items
        except Exception as exc:
            logger.error(f"Collector '{self.name}' failed with error: {exc}", exc_info=True)
            return []

    @abstractmethod
    def collect(self) -> list[RawItem]:
        """Subclasses must implement actual item extraction."""
        raise NotImplementedError
