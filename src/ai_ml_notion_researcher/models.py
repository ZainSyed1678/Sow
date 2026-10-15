"""Core data models and schemas for AI/ML Notion Researcher.

Validated using Pydantic v2 to ensure strict data integrity.
"""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceType(StrEnum):
    """Supported source collectors."""
    ARXIV = "arxiv"
    RSS = "rss"
    GITHUB = "github"


class NotionDatabaseType(StrEnum):
    """The 5 target Notion databases."""
    DAILY_BRIEFINGS = "daily_briefings"
    CONCEPT_LIBRARY = "concept_library"
    RESEARCH_PAPERS = "research_papers"
    TOOLS_FRAMEWORKS = "tools_frameworks"
    WEEKLY_REVIEWS = "weekly_reviews"


class RawItem(BaseModel):
    """Normalized raw publication or development collected from a research source."""
    id: str = Field(description="Deterministic unique ID based on URL or source ID")
    source_type: SourceType
    title: str = Field(min_length=1)
    url: str = Field(min_length=5)
    published_at: datetime | None = None
    raw_text: str = Field(default="", description="Snippet, abstract, or full extracted text")
    authors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def generate_id_if_empty(cls, v: Any, info: Any) -> str:
        if v:
            return str(v)
        # Fallback will be handled if not provided
        return ""

    @classmethod
    def make_id(cls, source_type: str, url: str) -> str:
        """Helper to create a deterministic hash ID from source and canonical URL."""
        normalized_url = url.strip().lower().rstrip("/")
        content = f"{source_type}:{normalized_url}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class ClassifiedItem(BaseModel):
    """Item filtered by relevance and assigned to a target Notion database."""
    item: RawItem
    target_database: NotionDatabaseType
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    topics: list[str] = Field(default_factory=list)
    summary: str = Field(description="Short summary of why this item is valuable")
    reasoning: str = Field(default="", description="Classification reasoning")


class TradeOffs(BaseModel):
    """Detailed advantages, limitations, and architectural trade-offs."""
    advantages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)


class EducationalNote(BaseModel):
    """Complete educational note meeting the 11 quality standards."""
    title: str = Field(min_length=1)
    summary: str = Field(min_length=10)
    target_database: NotionDatabaseType
    canonical_id: str = Field(description="Unique fingerprint for deduplication in Notion")

    # The 11 core educational components
    layman_explanation: str = Field(description="Concept explained in simple, intuitive layman's terms")
    technical_explanation: str = Field(description="Deep dive into underlying technical mechanisms")
    step_by_step_mechanism: list[str] = Field(description="Numbered step-by-step flow of how it works")
    real_world_use_cases: list[str] = Field(description="Concrete real-world applications")
    technologies_and_interactions: list[str] = Field(description="How tools, models, and frameworks interact")
    architecture_diagram: str | None = Field(default=None, description="Mermaid or ASCII architecture diagram")
    trade_offs: TradeOffs = Field(description="Advantages, limitations, and trade-offs")
    practical_exercise: str = Field(description="Hands-on coding or implementation project exercise")
    interview_takeaways: list[str] = Field(description="Key concepts and questions for technical interviews")
    source_links: list[str] = Field(description="Direct links to original papers, repos, or articles")

    # Meta
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WeeklyReviewNote(BaseModel):
    """Weekly synthesized review connecting developments across the week."""
    title: str = Field(min_length=1)
    week_start: datetime
    week_end: datetime
    major_developments: list[str] = Field(description="Summary of the week's major breakthroughs")
    concept_relationships: list[str] = Field(description="How concepts introduced this week connect")
    key_lessons: list[str] = Field(description="High-level engineering takeaways and principles")
    practical_exercises: list[str] = Field(description="Synthesized weekly mini-projects to build")
    referenced_note_titles: list[str] = Field(default_factory=list)
    source_links: list[str] = Field(default_factory=list)
    target_database: NotionDatabaseType = NotionDatabaseType.WEEKLY_REVIEWS
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionStats(BaseModel):
    """Runtime tracking metrics for a pipeline execution."""
    run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: str = "running"
    items_collected: int = 0
    items_deduplicated: int = 0
    items_scored: int = 0
    notes_generated: int = 0
    notes_published: int = 0
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = True
