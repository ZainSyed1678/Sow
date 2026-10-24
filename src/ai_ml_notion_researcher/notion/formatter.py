"""Formatters for translating EducationalNotes and WeeklyReviews into structured Notion blocks."""

from typing import Any

from ai_ml_notion_researcher.models import EducationalNote, WeeklyReviewNote
from ai_ml_notion_researcher.notion.blocks import (
    chunk_rich_text,
    create_bullet_item,
    create_callout,
    create_code_block,
    create_divider,
    create_heading_2,
    create_heading_3,
    create_numbered_item,
    create_paragraph,
    is_valid_notion_url,
)


def build_educational_note_blocks(note: EducationalNote) -> list[dict[str, Any]]:
    """Transforms an EducationalNote into rich, structured Notion blocks."""
    blocks: list[dict[str, Any]] = []

    # 1. Summary Callout
    blocks.append(create_callout(note.summary, emoji="📌"))
    blocks.append(create_divider())

    # 2. Layman's Explanation
    blocks.append(create_heading_2("1. Layman's Explanation"))
    blocks.append(create_callout(note.layman_explanation, emoji="🧑‍🏫"))

    # 3. Technical Explanation
    blocks.append(create_heading_2("2. Technical Mechanism & Theory"))
    blocks.append(create_paragraph(note.technical_explanation))

    # 4. Step-by-Step Flow
    blocks.append(create_heading_2("3. Step-by-Step Architecture Flow"))
    for step in note.step_by_step_mechanism:
        blocks.append(create_numbered_item(step))

    # Optional Diagram
    if note.architecture_diagram and note.architecture_diagram.strip():
        diag = note.architecture_diagram.strip()
        lang = "mermaid" if ("graph " in diag or "flowchart " in diag or "sequenceDiagram" in diag) else "plain text"
        blocks.append(create_code_block(diag, language=lang))

    # 5. Real-World Use Cases
    blocks.append(create_heading_2("4. Real-World Applications"))
    for uc in note.real_world_use_cases:
        blocks.append(create_bullet_item(uc))

    # 6. Technologies and Interactions
    blocks.append(create_heading_2("5. Ecosystem & How Technologies Interact"))
    for tech in note.technologies_and_interactions:
        blocks.append(create_bullet_item(tech))

    # 7. Trade-Offs & Limitations
    blocks.append(create_heading_2("6. Advantages, Limitations & Trade-Offs"))
    if note.trade_offs.advantages:
        blocks.append(create_heading_3("Key Advantages"))
        for adv in note.trade_offs.advantages:
            blocks.append(create_bullet_item(adv))

    if note.trade_offs.limitations:
        blocks.append(create_heading_3("Known Limitations"))
        for lim in note.trade_offs.limitations:
            blocks.append(create_bullet_item(lim))

    if note.trade_offs.trade_offs:
        blocks.append(create_heading_3("Architectural Trade-Offs"))
        for to in note.trade_offs.trade_offs:
            blocks.append(create_bullet_item(to))

    # 8. Practical Implementation Exercise
    blocks.append(create_heading_2("7. Hands-on Implementation Exercise"))
    blocks.append(create_callout(note.practical_exercise, emoji="💻"))

    # 9. Interview Takeaways
    blocks.append(create_heading_2("8. Technical Interview Preparation"))
    for takeaway in note.interview_takeaways:
        blocks.append(create_bullet_item(takeaway))

    # 10. Sources and Links
    if note.source_links:
        blocks.append(create_divider())
        blocks.append(create_heading_2("Original Sources & Research"))
        for link in note.source_links:
            clean_link = link.strip()
            if is_valid_notion_url(clean_link):
                blocks.append(create_paragraph(clean_link, link_url=clean_link))
            elif clean_link:
                blocks.append(create_paragraph(f"Reference: {clean_link}"))

    return blocks


def build_weekly_review_blocks(review: WeeklyReviewNote) -> list[dict[str, Any]]:
    """Transforms a WeeklyReviewNote into Notion blocks."""
    blocks: list[dict[str, Any]] = []

    date_str = f"Week: {review.week_start.strftime('%Y-%m-%d')} to {review.week_end.strftime('%Y-%m-%d')}"
    blocks.append(create_callout(f"AI/ML Synthesized Review - {date_str}", emoji="🧠"))
    blocks.append(create_divider())

    blocks.append(create_heading_2("1. Major Developments & Breakthroughs"))
    for dev in review.major_developments:
        blocks.append(create_bullet_item(dev))

    blocks.append(create_heading_2("2. Connecting Concepts & Intersections"))
    for rel in review.concept_relationships:
        blocks.append(create_bullet_item(rel))

    blocks.append(create_heading_2("3. Core Engineering Lessons & Principles"))
    for lesson in review.key_lessons:
        blocks.append(create_bullet_item(lesson))

    blocks.append(create_heading_2("4. Synthesized Weekly Mini-Projects to Build"))
    for exercise in review.practical_exercises:
        blocks.append(create_bullet_item(exercise))

    if review.source_links:
        blocks.append(create_divider())
        blocks.append(create_heading_2("References & Source Papers"))
        for link in review.source_links:
            clean_link = link.strip()
            if is_valid_notion_url(clean_link):
                blocks.append(create_paragraph(clean_link, link_url=clean_link))
            elif clean_link:
                blocks.append(create_paragraph(f"Reference: {clean_link}"))

    return blocks


def build_page_properties(
    title: str,
    canonical_id: str,
    summary: str = "",
    tags: list[str] | None = None,
    source_url: str | None = None,
    published_date: str | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dynamically builds Notion page property payload adapting to database schema."""
    tags = tags or []
    props: dict[str, Any] = {}

    title_prop_name = "Name"
    if schema:
        # Detect exact title property name configured in database
        for prop_name, prop_def in schema.items():
            if prop_def.get("type") == "title":
                title_prop_name = prop_name
                break

    props[title_prop_name] = {
        "title": chunk_rich_text(title[:500]),
    }

    if not schema:
        # Default standard properties when schema is not inspected
        if canonical_id:
            props["Canonical ID"] = {"rich_text": chunk_rich_text(canonical_id)}
        if summary:
            props["Summary"] = {"rich_text": chunk_rich_text(summary[:1800])}
        if source_url and is_valid_notion_url(source_url):
            props["URL"] = {"url": source_url.strip()}
        if published_date:
            props["Date"] = {"date": {"start": published_date}}
        if tags:
            props["Tags"] = {"multi_select": [{"name": t[:99].replace(",", "")} for t in tags[:10]]}
        return props

    # Adaptive schema matching: match property names ignoring case
    schema_lower = {k.lower(): (k, v.get("type")) for k, v in schema.items()}

    def find_prop(name: str) -> tuple[str, str] | None:
        return schema_lower.get(name.lower())

    # Canonical ID / ID
    for candidate in ["Canonical ID", "ID", "canonical_id", "Hash"]:
        matched = find_prop(candidate)
        if matched and matched[1] == "rich_text":
            props[matched[0]] = {"rich_text": chunk_rich_text(canonical_id)}
            break

    # Summary
    matched = find_prop("Summary") or find_prop("Description")
    if matched and matched[1] == "rich_text" and summary:
        props[matched[0]] = {"rich_text": chunk_rich_text(summary[:1800])}

    # URL / Source
    matched = find_prop("URL") or find_prop("Source") or find_prop("Link")
    if matched and matched[1] == "url" and source_url:
        if is_valid_notion_url(source_url):
            props[matched[0]] = {"url": source_url.strip()}

    # Date / Published Date
    matched = find_prop("Date") or find_prop("Published") or find_prop("Created")
    if matched and matched[1] == "date" and published_date:
        props[matched[0]] = {"date": {"start": published_date}}

    # Tags / Topics
    matched = find_prop("Tags") or find_prop("Topics") or find_prop("Category")
    if matched and matched[1] == "multi_select" and tags:
        clean_tags = [{"name": t[:99].replace(",", "")} for t in tags[:10]]
        props[matched[0]] = {"multi_select": clean_tags}
    return props


# Alias for backward compatibility
build_database_properties = build_page_properties
