"""High-level synthesis engine generating EducationalNotes and WeeklyReviews."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from ai_ml_notion_researcher.llm.base import BaseLLMClient
from ai_ml_notion_researcher.llm.prompts import (
    EDUCATIONAL_NOTE_SYSTEM_PROMPT,
    WEEKLY_REVIEW_SYSTEM_PROMPT,
    build_note_generation_prompt,
    build_weekly_review_prompt,
)
from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import (
    ClassifiedItem,
    EducationalNote,
    TradeOffs,
    WeeklyReviewNote,
)

logger = get_logger("llm.generator")


class _LLMNotePayload(BaseModel):
    """Intermediate model for LLM JSON parsing before attaching system metadata."""
    title: str
    summary: str
    layman_explanation: str
    technical_explanation: str
    step_by_step_mechanism: list[str]
    real_world_use_cases: list[str]
    technologies_and_interactions: list[str]
    architecture_diagram: str | None = None
    trade_offs: TradeOffs
    practical_exercise: str
    interview_takeaways: list[str]
    tags: list[str] = Field(default_factory=list)


class _LLMWeeklyReviewPayload(BaseModel):
    """Intermediate model for LLM weekly review JSON parsing."""
    title: str
    major_developments: list[str]
    concept_relationships: list[str]
    key_lessons: list[str]
    practical_exercises: list[str]


class NoteGenerator:
    """Orchestrates LLM synthesis into strictly validated educational notes."""

    def __init__(self, client: BaseLLMClient):
        self.client = client

    def generate_note(self, classified_item: ClassifiedItem) -> EducationalNote:
        """Synthesizes a ClassifiedItem into an EducationalNote."""
        raw_item = classified_item.item
        logger.info(f"Synthesizing educational note for '{raw_item.title}'...")

        prompt = build_note_generation_prompt(
            title=raw_item.title,
            source_type=raw_item.source_type.value,
            source_url=raw_item.url,
            raw_content=raw_item.raw_text,
            authors=raw_item.authors,
        )

        payload = self.client.generate_structured(
            prompt=prompt,
            system_prompt=EDUCATIONAL_NOTE_SYSTEM_PROMPT,
            response_model=_LLMNotePayload,
        )

        # Merge tags from classification and generation
        combined_tags = list(dict.fromkeys(payload.tags + classified_item.topics))

        note = EducationalNote(
            title=payload.title,
            summary=payload.summary,
            target_database=classified_item.target_database,
            canonical_id=raw_item.id,
            layman_explanation=payload.layman_explanation,
            technical_explanation=payload.technical_explanation,
            step_by_step_mechanism=payload.step_by_step_mechanism,
            real_world_use_cases=payload.real_world_use_cases,
            technologies_and_interactions=payload.technologies_and_interactions,
            architecture_diagram=payload.architecture_diagram,
            trade_offs=payload.trade_offs,
            practical_exercise=payload.practical_exercise,
            interview_takeaways=payload.interview_takeaways,
            source_links=[raw_item.url],
            tags=combined_tags,
            created_at=datetime.now(UTC),
        )

        logger.info(f"Successfully generated educational note: '{note.title}'")
        return note

    def generate_weekly_review(
        self,
        notes: list[EducationalNote],
        week_start: datetime,
        week_end: datetime,
    ) -> WeeklyReviewNote:
        """Synthesizes multiple educational notes into a WeeklyReviewNote."""
        logger.info(f"Generating weekly review from {len(notes)} notes...")

        summaries: list[dict[str, Any]] = [
            {
                "title": n.title,
                "summary": n.summary,
                "takeaways": "; ".join(n.interview_takeaways[:2]),
            }
            for n in notes
        ]

        prompt = build_weekly_review_prompt(summaries)
        payload = self.client.generate_structured(
            prompt=prompt,
            system_prompt=WEEKLY_REVIEW_SYSTEM_PROMPT,
            response_model=_LLMWeeklyReviewPayload,
        )

        all_source_links: list[str] = []
        for n in notes:
            all_source_links.extend(n.source_links)

        return WeeklyReviewNote(
            title=payload.title,
            week_start=week_start,
            week_end=week_end,
            major_developments=payload.major_developments,
            concept_relationships=payload.concept_relationships,
            key_lessons=payload.key_lessons,
            practical_exercises=payload.practical_exercises,
            referenced_note_titles=[n.title for n in notes],
            source_links=list(dict.fromkeys(all_source_links)),
            created_at=datetime.now(UTC),
        )
