"""arXiv API research collector for computer science AI/ML papers."""

import xml.etree.ElementTree as ET
from datetime import datetime

from ai_ml_notion_researcher.collectors.base import BaseCollector
from ai_ml_notion_researcher.config import ArxivSourceConfig
from ai_ml_notion_researcher.models import RawItem, SourceType
from ai_ml_notion_researcher.processing.sanitizer import sanitize_untrusted_text

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivCollector(BaseCollector):
    """Fetches and normalizes pre-print research papers from the arXiv API."""

    def __init__(
        self,
        config: ArxivSourceConfig,
        timeout_seconds: int = 30,
        user_agent: str = "AIMLNotionResearcher/0.1.0",
        client=None,
    ):
        super().__init__(
            name="arXiv",
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
            client=client,
        )
        self.config = config
        self.base_url = "https://export.arxiv.org/api/query"

    def collect(self) -> list[RawItem]:
        if not self.config.enabled or not self.config.categories:
            return []

        # Construct category query (e.g., "cat:cs.AI OR cat:cs.LG OR cat:cs.CL")
        cat_query = "+OR+".join([f"cat:{cat}" for cat in self.config.categories])
        url = (
            f"{self.base_url}?search_query={cat_query}"
            f"&sortBy={self.config.sort_by}"
            f"&sortOrder={self.config.sort_order}"
            f"&max_results={self.config.max_results_per_category}"
        )

        with self.get_http_client() as http:
            response = http.get(url)
            response.raise_for_status()
            return self._parse_feed(response.text)

    def _parse_feed(self, xml_text: str) -> list[RawItem]:
        """Parses arXiv Atom XML response into RawItem instances."""
        items: list[RawItem] = []
        root = ET.fromstring(xml_text)

        for entry in root.findall("atom:entry", ATOM_NS):
            id_elem = entry.find("atom:id", ATOM_NS)
            title_elem = entry.find("atom:title", ATOM_NS)
            summary_elem = entry.find("atom:summary", ATOM_NS)
            published_elem = entry.find("atom:published", ATOM_NS)

            raw_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
            # Canonicalize arXiv URL to standard abs URL
            canonical_url = raw_url.replace("http://", "https://")

            raw_title = title_elem.text if title_elem is not None and title_elem.text else "Untitled"
            # Normalize whitespace/newlines in title
            title = " ".join(raw_title.split())

            raw_summary = summary_elem.text if summary_elem is not None and summary_elem.text else ""
            abstract = sanitize_untrusted_text(raw_summary)

            published_at = None
            if published_elem is not None and published_elem.text:
                try:
                    published_at = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None

            # Extract authors
            authors: list[str] = []
            for author_elem in entry.findall("atom:author/atom:name", ATOM_NS):
                if author_elem.text:
                    authors.append(author_elem.text.strip())

            # Generate deterministic item ID
            item_id = RawItem.make_id(SourceType.ARXIV.value, canonical_url)

            items.append(
                RawItem(
                    id=item_id,
                    source_type=SourceType.ARXIV,
                    title=title,
                    url=canonical_url,
                    published_at=published_at,
                    raw_text=abstract,
                    authors=authors,
                    metadata={"source": "arxiv"},
                )
            )

        return items
