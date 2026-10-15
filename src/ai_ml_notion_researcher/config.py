"""Configuration loader and schema validation for AI/ML Notion Researcher.

Combines YAML configuration files with environment variables via Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

from ai_ml_notion_researcher.models import NotionDatabaseType


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""
    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    api_key: SecretStr | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_seconds: int = 45
    max_retries: int = 3
    request_delay_seconds: float = 0.0


class NotionConfig(BaseModel):
    """Configuration for Notion integration."""
    api_key: SecretStr | None = None
    api_version: str = "2022-06-28"
    base_url: str = "https://api.notion.com/v1"
    rate_limit_requests_per_second: int = 3
    database_ids: dict[NotionDatabaseType, str] = Field(default_factory=dict)

    def is_configured_for_publishing(self) -> bool:
        """Checks if all required database IDs and API key are provided."""
        if not self.api_key or not self.api_key.get_secret_value().strip():
            return False
        for db_type in NotionDatabaseType:
            if not self.database_ids.get(db_type):
                return False
        return True


class PipelineConfig(BaseModel):
    """Execution limits and runtime behavior."""
    dry_run: bool = True
    max_items_to_score: int = 50
    max_items_to_generate: int = 6
    min_relevance_score: float = 0.65
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    user_agent: str = "AIMLNotionResearcher/0.1.0"


class StorageConfig(BaseModel):
    """Local execution state storage."""
    sqlite_path: str = "local_state/researcher.db"


class ArxivSourceConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL"])
    max_results_per_category: int = 15
    sort_by: str = "submittedDate"
    sort_order: str = "descending"


class RSSFeedItemConfig(BaseModel):
    name: str
    url: str
    category: NotionDatabaseType = NotionDatabaseType.DAILY_BRIEFINGS

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, v: Any) -> NotionDatabaseType:
        if isinstance(v, NotionDatabaseType):
            return v
        return NotionDatabaseType(str(v))


class RSSSourceConfig(BaseModel):
    enabled: bool = True
    feeds: list[RSSFeedItemConfig] = Field(default_factory=list)


class GithubSourceConfig(BaseModel):
    enabled: bool = True
    topics: list[str] = Field(default_factory=list)
    tracked_repos: list[str] = Field(default_factory=list)
    min_stars_daily: int = 20
    api_token: SecretStr | None = None


class SourcesConfig(BaseModel):
    topic: str = "ai_ml"
    arxiv: ArxivSourceConfig = Field(default_factory=ArxivSourceConfig)
    rss_feeds: RSSSourceConfig = Field(default_factory=RSSSourceConfig)
    github: GithubSourceConfig = Field(default_factory=GithubSourceConfig)


class AppConfig(BaseModel):
    """Root application configuration."""
    topic: str = "ai_ml"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    notion: NotionConfig = Field(default_factory=NotionConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    sources: SourcesConfig = Field(default_factory=SourcesConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Safely loads a YAML file if it exists."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _clean_notion_id(raw_id: str | None) -> str:
    """Extracts and normalizes Notion database ID by removing query params and dashes."""
    if not raw_id:
        return ""
    val = raw_id.split("?")[0].replace("-", "").strip()
    if "/" in val:
        val = val.rstrip("/").split("/")[-1]
    return val


def load_config(
    project_root: Path | None = None,
    settings_file: str = "config/settings.yaml",
    sources_file: str = "config/sources.yaml",
    load_env: bool = True,
) -> AppConfig:
    """Loads configuration by merging YAML files and environment variables.

    Environment variables take precedence over YAML configuration values.
    """
    if project_root is None:
        # Default to finding project root relative to this file
        project_root = Path(__file__).resolve().parent.parent.parent

    # Load .env file if enabled and present
    if load_env:
        env_path = project_root / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

    settings_data = _load_yaml(project_root / settings_file)
    sources_data = _load_yaml(project_root / sources_file)

    # 1. Pipeline & App settings
    app_meta = settings_data.get("app", {})
    topic = os.getenv("TOPIC", app_meta.get("topic", "ai_ml"))

    pipe_data = settings_data.get("pipeline", {})
    dry_run_env = os.getenv("DRY_RUN")
    dry_run = dry_run_env.lower() in ("true", "1", "yes") if dry_run_env else pipe_data.get("dry_run", True)

    pipeline_cfg = PipelineConfig(
        dry_run=dry_run,
        max_items_to_score=int(pipe_data.get("max_items_to_score", 50)),
        max_items_to_generate=int(pipe_data.get("max_items_to_generate", 6)),
        min_relevance_score=float(pipe_data.get("min_relevance_score", 0.65)),
        request_timeout_seconds=int(pipe_data.get("request_timeout_seconds", 30)),
        retry_attempts=int(pipe_data.get("retry_attempts", 3)),
        retry_backoff_factor=float(pipe_data.get("retry_backoff_factor", 2.0)),
        user_agent=pipe_data.get("user_agent", "AIMLNotionResearcher/0.1.0"),
    )

    # 2. LLM Config
    llm_data = settings_data.get("llm", {})
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", llm_data.get("default_model", "openai/gpt-oss-120b"))
    groq_delay = os.getenv("GROQ_REQUEST_DELAY_SECONDS")
    request_delay = float(groq_delay) if groq_delay else float(llm_data.get("request_delay_seconds", 5.0))

    llm_cfg = LLMConfig(
        provider=llm_data.get("provider", "groq"),
        model=groq_model,
        api_key=SecretStr(groq_api_key) if groq_api_key else None,
        temperature=float(llm_data.get("temperature", 0.2)),
        max_tokens=int(llm_data.get("max_tokens", 4096)),
        base_url=llm_data.get("base_url", "https://api.groq.com/openai/v1"),
        timeout_seconds=int(llm_data.get("timeout_seconds", 45)),
        max_retries=int(llm_data.get("max_retries", 3)),
        request_delay_seconds=request_delay,
    )

    # 3. Notion Config
    notion_data = settings_data.get("notion", {})
    notion_api_key = os.getenv("NOTION_API_KEY")

    database_ids: dict[NotionDatabaseType, str] = {
        NotionDatabaseType.DAILY_BRIEFINGS: _clean_notion_id(os.getenv("NOTION_DATABASE_ID_DAILY_BRIEFINGS", "")),
        NotionDatabaseType.CONCEPT_LIBRARY: _clean_notion_id(os.getenv("NOTION_DATABASE_ID_CONCEPT_LIBRARY", "")),
        NotionDatabaseType.RESEARCH_PAPERS: _clean_notion_id(os.getenv("NOTION_DATABASE_ID_RESEARCH_PAPERS", "")),
        NotionDatabaseType.TOOLS_FRAMEWORKS: _clean_notion_id(os.getenv("NOTION_DATABASE_ID_TOOLS_FRAMEWORKS", "")),
        NotionDatabaseType.WEEKLY_REVIEWS: _clean_notion_id(os.getenv("NOTION_DATABASE_ID_WEEKLY_REVIEWS", "")),
    }

    notion_cfg = NotionConfig(
        api_key=SecretStr(notion_api_key) if notion_api_key else None,
        api_version=notion_data.get("api_version", "2022-06-28"),
        base_url=notion_data.get("base_url", "https://api.notion.com/v1"),
        rate_limit_requests_per_second=int(notion_data.get("rate_limit_requests_per_second", 3)),
        database_ids=database_ids,
    )

    # 4. Storage Config
    storage_data = settings_data.get("storage", {})
    sqlite_path = os.getenv("SQLITE_DB_PATH", storage_data.get("sqlite_path", "local_state/researcher.db"))
    storage_cfg = StorageConfig(sqlite_path=sqlite_path)

    # 5. Sources Config
    github_token = os.getenv("GITHUB_TOKEN")
    github_raw = sources_data.get("github", {})
    github_cfg = GithubSourceConfig(
        enabled=github_raw.get("enabled", True),
        topics=github_raw.get("topics", []),
        tracked_repos=github_raw.get("tracked_repos", []),
        min_stars_daily=int(github_raw.get("min_stars_daily", 20)),
        api_token=SecretStr(github_token) if github_token else None,
    )

    arxiv_raw = sources_data.get("arxiv", {})
    arxiv_cfg = ArxivSourceConfig(
        enabled=arxiv_raw.get("enabled", True),
        categories=arxiv_raw.get("categories", ["cs.AI", "cs.LG", "cs.CL"]),
        max_results_per_category=int(arxiv_raw.get("max_results_per_category", 15)),
        sort_by=arxiv_raw.get("sort_by", "submittedDate"),
        sort_order=arxiv_raw.get("sort_order", "descending"),
    )

    rss_raw = sources_data.get("rss_feeds", {})
    rss_feeds_list = [
        RSSFeedItemConfig(
            name=f["name"],
            url=f["url"],
            category=f.get("category", NotionDatabaseType.DAILY_BRIEFINGS),
        )
        for f in rss_raw.get("feeds", [])
    ]
    rss_cfg = RSSSourceConfig(
        enabled=rss_raw.get("enabled", True),
        feeds=rss_feeds_list,
    )

    sources_cfg = SourcesConfig(
        topic=sources_data.get("topic", "ai_ml"),
        arxiv=arxiv_cfg,
        rss_feeds=rss_cfg,
        github=github_cfg,
    )

    return AppConfig(
        topic=topic,
        llm=llm_cfg,
        notion=notion_cfg,
        pipeline=pipeline_cfg,
        storage=storage_cfg,
        sources=sources_cfg,
    )
