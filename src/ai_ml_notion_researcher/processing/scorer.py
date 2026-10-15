"""Relevance scoring, noise filtering, and database classification for AI/ML content."""

import re

from ai_ml_notion_researcher.models import (
    ClassifiedItem,
    NotionDatabaseType,
    RawItem,
    SourceType,
)

# High-value AI/ML technical indicators
CORE_TOPIC_KEYWORDS = {
    "transformer",
    "attention",
    "llm",
    "large language model",
    "reasoning",
    "diffusion",
    "quantization",
    "fine-tuning",
    "finetuning",
    "lora",
    "benchmark",
    "rlhf",
    "dpo",
    "vllm",
    "rag",
    "retrieval-augmented",
    "agent",
    "autonomous agent",
    "embedding",
    "inference",
    "architecture",
    "multimodal",
    "distillation",
    "kv cache",
    "flashattention",
    "speculative decoding",
    "mixture of experts",
    "moe",
    "scaling laws",
    "interpretability",
    "mechanistic",
}

# Technical depth markers
DEPTH_MARKERS = {
    "benchmark",
    "evaluation",
    "ablation",
    "throughput",
    "latency",
    "memory",
    "gradient",
    "loss",
    "weights",
    "algorithm",
    "optimization",
    "perplexity",
    "accuracy",
    "state-of-the-art",
    "sota",
}

# Generic noise or clickbait filters
LOW_VALUE_MARKERS = {
    "crypto",
    "nft",
    "bitcoin",
    "stock price",
    "venture capital funding round",
    "celebrity",
    "sponsored",
}


def compute_relevance_score(
    title: str,
    text: str,
    source_type: SourceType,
    topic: str | None = None,
    custom_keywords: list[str] | set[str] | None = None,
) -> float:
    """Computes a normalized relevance score between 0.0 and 1.0 based on technical signal."""
    combined = f"{title} {text}".lower()

    # Immediate rejection if clickbait/noise signals dominate
    noise_count = sum(1 for m in LOW_VALUE_MARKERS if m in combined)
    if noise_count >= 2:
        return 0.1

    # Base score by source authority
    score = 0.50 if source_type in (SourceType.ARXIV, SourceType.CUSTOM) else 0.30

    # Build active keyword set from core keywords + custom topic keywords
    active_keywords = set(CORE_TOPIC_KEYWORDS)
    if custom_keywords:
        for ck in custom_keywords:
            if ck and ck.strip():
                active_keywords.add(ck.strip().lower())

    topic_matched = False
    if topic and topic.strip() and topic.strip().lower() != "ai_ml":
        clean_topic = topic.strip().lower().replace("_", " ").replace("-", " ")
        active_keywords.add(clean_topic)
        topic_parts = [p for p in clean_topic.split() if len(p) > 2]
        for part in topic_parts:
            active_keywords.add(part)
            if part.endswith("ics"):
                active_keywords.add(part[:-3])  # robotics -> robot
                active_keywords.add(part[:-1])  # robotics -> robotic
            elif part.endswith("ing"):
                active_keywords.add(part[:-3])  # reasoning -> reason
            elif part.endswith("s"):
                active_keywords.add(part[:-1])

        if any(kw in combined for kw in active_keywords if kw not in CORE_TOPIC_KEYWORDS):
            topic_matched = True

    # Match active topic keywords
    keyword_matches = sum(1 for kw in active_keywords if re.search(rf"\b{re.escape(kw)}\b", combined))
    # Add up to 0.35 based on keyword density
    score += min(0.35, keyword_matches * 0.07)

    # Match depth markers
    depth_matches = sum(1 for dm in DEPTH_MARKERS if re.search(rf"\b{re.escape(dm)}\b", combined))
    score += min(0.15, depth_matches * 0.03)

    # Title match bonus (topic present in title is high signal)
    title_matches = sum(1 for kw in active_keywords if re.search(rf"\b{re.escape(kw)}\b", title.lower()))
    if title_matches > 0:
        score += 0.15

    # Bonus if user-specified custom topic was directly matched
    if topic_matched:
        score += 0.15

    return min(1.0, max(0.0, round(score, 2)))


def classify_database(item: RawItem, relevance_score: float) -> NotionDatabaseType:
    """Determines the appropriate Notion database destination."""
    combined = f"{item.title} {item.raw_text}".lower()

    # Explicit metadata destination takes priority
    if "target_database" in item.metadata:
        try:
            return NotionDatabaseType(item.metadata["target_database"])
        except ValueError:
            pass

    if item.source_type == SourceType.ARXIV:
        return NotionDatabaseType.RESEARCH_PAPERS

    if item.source_type == SourceType.GITHUB:
        return NotionDatabaseType.TOOLS_FRAMEWORKS

    # RSS feed or custom entry classification heuristics
    if any(k in combined for k in ["release", "library", "sdk", "github.com", "framework", "repo", "v0.", "v1.", "tool"]):
        return NotionDatabaseType.TOOLS_FRAMEWORKS

    if any(k in combined for k in ["how it works", "deep dive", "architecture", "under the hood", "guide", "concept", "tutorial", "explained"]):
        return NotionDatabaseType.CONCEPT_LIBRARY

    if any(k in combined for k in ["paper", "research", "arxiv", "propose", "methodology", "study", "dataset"]):
        return NotionDatabaseType.RESEARCH_PAPERS

    return NotionDatabaseType.DAILY_BRIEFINGS


def extract_matched_topics(
    text: str,
    topic: str | None = None,
    custom_keywords: list[str] | None = None,
) -> list[str]:
    """Extracts detected topic tags from text."""
    lower_text = text.lower()
    active_keywords = set(CORE_TOPIC_KEYWORDS)
    if custom_keywords:
        for ck in custom_keywords:
            if ck and ck.strip():
                active_keywords.add(ck.strip().lower())
    if topic and topic.strip() and topic.strip().lower() != "ai_ml":
        clean_topic = topic.strip().lower().replace("_", " ").replace("-", " ")
        active_keywords.add(clean_topic)

    matched = [
        kw.title()
        for kw in active_keywords
        if re.search(rf"\b{re.escape(kw)}\b", lower_text)
    ]
    # Ensure custom topic is tagged if provided
    if topic and topic.strip() and topic.strip().lower() != "ai_ml":
        topic_title = topic.replace("_", " ").replace("-", " ").title()
        if topic_title not in matched:
            matched.insert(0, topic_title)

    return matched[:5]


def score_and_classify(
    item: RawItem,
    min_score: float = 0.40,
    topic: str | None = None,
    custom_keywords: list[str] | None = None,
) -> ClassifiedItem | None:
    """Scores an item and assigns it to a target database if it meets the quality threshold."""
    score = compute_relevance_score(
        title=item.title,
        text=item.raw_text,
        source_type=item.source_type,
        topic=topic,
        custom_keywords=custom_keywords,
    )
    if score < min_score:
        return None

    target_db = classify_database(item, score)
    topics = extract_matched_topics(
        text=f"{item.title} {item.raw_text}",
        topic=topic,
        custom_keywords=custom_keywords,
    )
    summary = item.raw_text[:250].strip() or item.title

    reasoning = (
        f"Scored {score:.2f} (threshold >= {min_score:.2f}) due to keyword alignment and technical depth from {item.source_type.value}."
    )

    return ClassifiedItem(
        item=item,
        target_database=target_db,
        relevance_score=score,
        topics=topics,
        summary=summary,
        reasoning=reasoning,
    )
