"""Notion API client with rate-limiting, bounded backoff, idempotency, and safe publishing."""

from __future__ import annotations

import time
from typing import Any

import httpx

from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import EducationalNote, WeeklyReviewNote
from ai_ml_notion_researcher.notion.blocks import chunk_blocks
from ai_ml_notion_researcher.notion.formatter import (
    build_educational_note_blocks,
    build_page_properties,
    build_weekly_review_blocks,
)

logger = get_logger("notion.client")


class NotionAPIError(Exception):
    """Base exception for Notion API interactions."""
    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class NotionRateLimitError(NotionAPIError):
    """Raised when Notion API rate limit (HTTP 429) persists after retries."""


class NotionPermissionError(NotionAPIError):
    """Raised when integration does not have access to database or page."""


class NotionNotFoundError(NotionAPIError):
    """Raised when requested database or page does not exist."""


class NotionClient:
    """Production-ready Notion API client adhering to safe publishing rules."""

    def __init__(
        self,
        api_key: str,
        api_version: str = "2022-06-28",
        base_url: str = "https://api.notion.com/v1",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._schema_cache: dict[str, dict[str, Any]] = {}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.api_version,
            "Content-Type": "application/json",
        }

        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout_seconds,
        )

    def __enter__(self) -> NotionClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Closes the underlying HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Executes an HTTP request with bounded retries for 429 and transient 5xx errors."""
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        attempt = 0

        while attempt <= self.max_retries:
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 429:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise NotionRateLimitError(
                            "Notion API rate limit exceeded after maximum retries.",
                            status_code=429,
                            response_body=response.text,
                        )
                    # Respect Retry-After header if provided
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (self.backoff_factor ** attempt)
                    logger.warning(
                        "Rate limited by Notion API (429). Retrying...",
                        extra={"attempt": attempt, "sleep_time": sleep_time},
                    )
                    time.sleep(sleep_time)
                    continue

                if 500 <= response.status_code < 600:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise NotionAPIError(
                            f"Notion server error ({response.status_code}) after retries: {response.text}",
                            status_code=response.status_code,
                        )
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)
                    continue

                if response.status_code == 404:
                    raise NotionNotFoundError(
                        f"Resource not found on Notion: {response.text}",
                        status_code=404,
                    )

                if response.status_code in (401, 403):
                    raise NotionPermissionError(
                        f"Notion permission denied ({response.status_code}): {response.text}",
                        status_code=response.status_code,
                    )

                # Any other 4xx errors
                raise NotionAPIError(
                    f"Notion API error ({response.status_code}): {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            except httpx.RequestError as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise NotionAPIError(f"HTTP transport failed after {attempt} attempts: {exc}") from exc
                time.sleep(self.backoff_factor ** attempt)

        raise NotionAPIError("Unexpected exit from request retry loop.")

    def get_database(self, database_id: str) -> dict[str, Any]:
        """Retrieves metadata and schema definition for a Notion database."""
        db_id = database_id.replace("-", "")
        return self._request("GET", f"databases/{db_id}")

    def validate_database(self, database_id: str) -> tuple[bool, str, dict[str, Any]]:
        """Safely verifies database existence and extracts its property schema.

        Returns (is_valid, status_message, properties_dict).
        """
        try:
            data = self.get_database(database_id)
            title_objs = data.get("title", [])
            title_name = "".join([t.get("plain_text", "") for t in title_objs]) or "Untitled Database"
            properties = data.get("properties", {})
            self._schema_cache[database_id] = properties
            return True, f"Connected to '{title_name}' ({len(properties)} properties)", properties
        except NotionNotFoundError:
            return False, "Database not found. Check that the ID is correct.", {}
        except NotionPermissionError:
            return False, "Permission denied. Ensure the Notion integration is shared with this database.", {}
        except NotionAPIError as err:
            return False, f"API error: {err}", {}

    def find_page_by_canonical_id(self, database_id: str, canonical_id: str) -> dict[str, Any] | None:
        """Queries the database to check if a page with the given Canonical ID already exists."""
        db_id = database_id.replace("-", "")
        query_payload = {
            "filter": {
                "property": "Canonical ID",
                "rich_text": {
                    "equals": canonical_id,
                },
            },
            "page_size": 1,
        }
        try:
            res = self._request("POST", f"databases/{db_id}/query", json_data=query_payload)
            results = res.get("results", [])
            return results[0] if results else None
        except NotionAPIError:
            # If the database does not have a "Canonical ID" property configured,
            # query without filter and return None to allow safe fallback.
            return None

    def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
        children: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Creates a page in the specified database, handling block chunking if >100 blocks."""
        db_id = database_id.replace("-", "")
        block_batches = chunk_blocks(children, chunk_size=100)
        initial_blocks = block_batches[0] if block_batches else []

        payload = {
            "parent": {"database_id": db_id},
            "properties": properties,
            "children": initial_blocks,
        }

        created = self._request("POST", "pages", json_data=payload)
        page_id = created.get("id")

        # Append remaining blocks if total children exceeded 100
        if len(block_batches) > 1 and page_id:
            for extra_batch in block_batches[1:]:
                self._request("PATCH", f"blocks/{page_id}/children", json_data={"children": extra_batch})

        return created

    def publish_note(
        self,
        note: EducationalNote,
        database_id: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Publishes an EducationalNote to Notion idempotently.

        # ponytail: If dry_run is True, validates blocks without calling Notion write endpoints.
        """
        blocks = build_educational_note_blocks(note)

        if dry_run:
            logger.info(
                f"[DRY-RUN] Validated note '{note.title}' ({len(blocks)} blocks) for database {database_id}"
            )
            return {
                "status": "dry_run_success",
                "canonical_id": note.canonical_id,
                "title": note.title,
                "blocks_count": len(blocks),
                "target_database": str(note.target_database),
            }

        # 1. Deduplication check
        existing = self.find_page_by_canonical_id(database_id, note.canonical_id)
        if existing:
            logger.info(
                f"Skipping duplicate note '{note.title}' (Canonical ID: {note.canonical_id})",
                extra={"page_id": existing.get("id")},
            )
            return {
                "status": "skipped_duplicate",
                "canonical_id": note.canonical_id,
                "page_id": existing.get("id"),
            }

        # 2. Retrieve schema to dynamically adapt property names
        schema = self._schema_cache.get(database_id)
        if schema is None:
            _, _, schema = self.validate_database(database_id)

        source_url = note.source_links[0] if note.source_links else None
        date_str = note.created_at.strftime("%Y-%m-%d")

        properties = build_page_properties(
            title=note.title,
            canonical_id=note.canonical_id,
            summary=note.summary,
            tags=note.tags,
            source_url=source_url,
            published_date=date_str,
            schema=schema,
        )

        res = self.create_page(database_id, properties=properties, children=blocks)
        logger.info(f"Published note '{note.title}' to Notion database {database_id}")
        return {
            "status": "published",
            "canonical_id": note.canonical_id,
            "page_id": res.get("id"),
            "url": res.get("url"),
        }

    def publish_weekly_review(
        self,
        review: WeeklyReviewNote,
        database_id: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Publishes a WeeklyReviewNote to Notion."""
        blocks = build_weekly_review_blocks(review)

        if dry_run:
            logger.info(f"[DRY-RUN] Validated weekly review '{review.title}' ({len(blocks)} blocks)")
            return {
                "status": "dry_run_success",
                "title": review.title,
                "blocks_count": len(blocks),
            }

        schema = self._schema_cache.get(database_id)
        if schema is None:
            _, _, schema = self.validate_database(database_id)

        date_str = review.week_start.strftime("%Y-%m-%d")
        properties = build_page_properties(
            title=review.title,
            canonical_id=f"weekly-{review.week_start.strftime('%Y-%W')}",
            summary="; ".join(review.major_developments[:2]),
            tags=["Weekly Review"],
            published_date=date_str,
            schema=schema,
        )

        res = self.create_page(database_id, properties=properties, children=blocks)
        return {
            "status": "published",
            "page_id": res.get("id"),
            "url": res.get("url"),
        }
