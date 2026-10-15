"""Text sanitization and prompt injection defense for untrusted research sources."""

import re
from html import unescape
from html.parser import HTMLParser

# Patterns commonly used in prompt injection attempts
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"(system\s+prompt|system\s+message)\s*:", re.IGNORECASE),
    re.compile(r"<system>.*?</system>", re.IGNORECASE | re.DOTALL),
    re.compile(r"```(system|instructions?).*?```", re.IGNORECASE | re.DOTALL),
    re.compile(r"you\s+are\s+now\s+(in\s+developer\s+mode|dan|unfiltered)", re.IGNORECASE),
    re.compile(r"bypass\s+all\s+(rules|filters|safety)", re.IGNORECASE),
]


class _MLStripper(HTMLParser):
    """Minimal HTML tag stripper using Python standard library."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts: list[str] = []

    def handle_data(self, d: str) -> None:
        self.text_parts.append(d)

    def get_data(self) -> str:
        return "".join(self.text_parts)


def strip_html_tags(html_text: str) -> str:
    """Strips HTML markup and unescapes HTML entities using standard library."""
    if not html_text:
        return ""
    stripper = _MLStripper()
    stripper.feed(html_text)
    clean_text = stripper.get_data()
    return unescape(clean_text)


def sanitize_untrusted_text(text: str, max_chars: int = 5000) -> str:
    """Sanitizes untrusted raw content from web feeds, repos, and research papers.

    - Strips HTML tags and unescapes entities.
    - Defangs known prompt injection vectors.
    - Strips non-printable control characters.
    - Truncates to max_chars to prevent context stuffing.
    """
    if not text:
        return ""

    # 1. Strip HTML tags
    cleaned = strip_html_tags(text)

    # 2. Defang prompt injection attempts
    for pattern in INJECTION_PATTERNS:
        cleaned = pattern.sub("[FILTERED_POTENTIAL_INJECTION]", cleaned)

    # 3. Strip non-printable control characters except newline and tab
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # 4. Normalize repetitive whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    # 5. Length bounding
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "... [TRUNCATED]"

    return cleaned
