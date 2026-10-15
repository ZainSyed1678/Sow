"""RSS and Atom feed collector for industry labs, engineering blogs, and research news."""

import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

from ai_ml_notion_researcher.collectors.base import BaseCollector
from ai_ml_notion_researcher.config import RSSSourceConfig
from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import RawItem, SourceType
from ai_ml_notion_researcher.processing.sanitizer import sanitize_untrusted_text

logger = get_logger("collectors.rss")


def _find_child(element: ET.Element, tag_name: str) -> ET.Element | None:
    """Finds first direct child matching tag_name, ignoring XML namespaces."""
    for child in element:
        if child.tag == tag_name or child.tag.endswith(f"}}{tag_name}"):
            return child
    return None


class RSSCollector(BaseCollector):
    """Fetches articles from AI lab feeds (HuggingFace, OpenAI, DeepMind, BAIR, etc.)."""

    def __init__(
        self,
        config: RSSSourceConfig,
        timeout_seconds: int = 30,
        user_agent: str = "AIMLNotionResearcher/0.1.0",
        client=None,
    ):
        super().__init__(
            name="RSSFeeds",
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
            client=client,
        )
        self.config = config

    def collect(self) -> list[RawItem]:
        if not self.config.enabled or not self.config.feeds:
            return []

        all_items: list[RawItem] = []
        with self.get_http_client() as http:
            for feed in self.config.feeds:
                try:
                    logger.debug(f"Fetching RSS feed: {feed.name} ({feed.url})")
                    resp = http.get(feed.url)
                    resp.raise_for_status()
                    feed_items = self._parse_feed_xml(resp.text, feed.name)
                    all_items.extend(feed_items)
                except Exception as exc:
                    logger.warning(f"Failed to fetch RSS feed '{feed.name}': {exc}")
                    continue

        return all_items

    def _parse_feed_xml(self, xml_text: str, feed_name: str) -> list[RawItem]:
        """Parses both RSS 2.0 and Atom feeds into RawItem models."""
        items: list[RawItem] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as err:
            logger.warning(f"Malformed XML in feed '{feed_name}': {err}")
            return []

        # Case 1: RSS 2.0 (channel -> item)
        channel = _find_child(root, "channel")
        if channel is not None:
            for item_elem in channel:
                if item_elem.tag == "item" or item_elem.tag.endswith("}item"):
                    raw_item = self._parse_rss_item(item_elem, feed_name)
                    if raw_item:
                        items.append(raw_item)
            return items

        # Case 2: Atom (<feed> -> <entry>)
        for entry_elem in root:
            if entry_elem.tag == "entry" or entry_elem.tag.endswith("}entry"):
                raw_item = self._parse_atom_entry(entry_elem, feed_name)
                if raw_item:
                    items.append(raw_item)

        return items

    def _parse_rss_item(self, item_elem: ET.Element, feed_name: str) -> RawItem | None:
        title_elem = _find_child(item_elem, "title")
        link_elem = _find_child(item_elem, "link")
        desc_elem = _find_child(item_elem, "description")
        pub_elem = _find_child(item_elem, "pubDate")

        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
        if not title or not link:
            return None

        content = desc_elem.text if desc_elem is not None and desc_elem.text else ""
        clean_text = sanitize_untrusted_text(content)

        published_at = None
        if pub_elem is not None and pub_elem.text:
            try:
                published_at = parsedate_to_datetime(pub_elem.text)
            except Exception:
                published_at = None

        return RawItem(
            id=RawItem.make_id(SourceType.RSS.value, link),
            source_type=SourceType.RSS,
            title=title,
            url=link,
            published_at=published_at,
            raw_text=clean_text,
            metadata={"feed_name": feed_name},
        )

    def _parse_atom_entry(self, entry_elem: ET.Element, feed_name: str) -> RawItem | None:
        title_elem = _find_child(entry_elem, "title")
        link_elem = _find_child(entry_elem, "link")
        summary_elem = _find_child(entry_elem, "summary") or _find_child(entry_elem, "content")
        pub_elem = _find_child(entry_elem, "published") or _find_child(entry_elem, "updated")

        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        link = ""
        if link_elem is not None:
            link = link_elem.attrib.get("href") or (link_elem.text or "").strip()

        if not title or not link:
            return None

        content = summary_elem.text if summary_elem is not None and summary_elem.text else ""
        clean_text = sanitize_untrusted_text(content)

        published_at = None
        if pub_elem is not None and pub_elem.text:
            try:
                published_at = datetime.fromisoformat(pub_elem.text.replace("Z", "+00:00"))
            except ValueError:
                published_at = None

        return RawItem(
            id=RawItem.make_id(SourceType.RSS.value, link),
            source_type=SourceType.RSS,
            title=title,
            url=link,
            published_at=published_at,
            raw_text=clean_text,
            metadata={"feed_name": feed_name},
        )
