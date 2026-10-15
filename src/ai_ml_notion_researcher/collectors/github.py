"""GitHub REST API collector for AI/ML framework releases and trending repositories."""

from datetime import datetime
from typing import Any

from ai_ml_notion_researcher.collectors.base import BaseCollector
from ai_ml_notion_researcher.config import GithubSourceConfig
from ai_ml_notion_researcher.logging_utils import get_logger
from ai_ml_notion_researcher.models import RawItem, SourceType
from ai_ml_notion_researcher.processing.sanitizer import sanitize_untrusted_text

logger = get_logger("collectors.github")


class GithubCollector(BaseCollector):
    """Collects releases and trending AI repositories from GitHub REST API."""

    def __init__(
        self,
        config: GithubSourceConfig,
        timeout_seconds: int = 30,
        user_agent: str = "AIMLNotionResearcher/0.1.0",
        client=None,
    ):
        super().__init__(
            name="GitHub",
            timeout_seconds=timeout_seconds,
            user_agent=user_agent,
            client=client,
        )
        self.config = config
        self.base_url = "https://api.github.com"

    def get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.config.api_token and self.config.api_token.get_secret_value().strip():
            headers["Authorization"] = f"Bearer {self.config.api_token.get_secret_value()}"
        return headers

    def collect(self) -> list[RawItem]:
        if not self.config.enabled:
            return []

        items: list[RawItem] = []
        with self.get_http_client() as http:
            # 1. Fetch latest releases from tracked repos
            for repo in self.config.tracked_repos:
                try:
                    rel_item = self._fetch_latest_release(http, repo)
                    if rel_item:
                        items.append(rel_item)
                except Exception as exc:
                    logger.warning(f"Failed to fetch release for repo '{repo}': {exc}")

            # 2. Search updated repositories by topic if topics are configured
            for topic in self.config.topics[:3]:
                try:
                    topic_items = self._search_topic_repos(http, topic)
                    items.extend(topic_items)
                except Exception as exc:
                    logger.warning(f"Failed to query GitHub topic '{topic}': {exc}")

        return items

    def _fetch_latest_release(self, http: Any, repo_full_name: str) -> RawItem | None:
        url = f"{self.base_url}/repos/{repo_full_name}/releases/latest"
        resp = http.get(url, headers=self.get_headers())
        if resp.status_code == 404:
            # Repo has no formal releases
            return None
        resp.raise_for_status()

        data = resp.json()
        tag = data.get("tag_name", "Release")
        name = data.get("name") or tag
        title = f"{repo_full_name} {name}"
        html_url = data.get("html_url", f"https://github.com/{repo_full_name}")
        body = data.get("body", "")
        clean_text = sanitize_untrusted_text(f"Release {tag}:\n{body}", max_chars=4000)

        published_at = None
        pub_str = data.get("published_at")
        if pub_str:
            try:
                published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except ValueError:
                published_at = None

        return RawItem(
            id=RawItem.make_id(SourceType.GITHUB.value, html_url),
            source_type=SourceType.GITHUB,
            title=title,
            url=html_url,
            published_at=published_at,
            raw_text=clean_text,
            metadata={"repo": repo_full_name, "tag": tag},
        )

    def _search_topic_repos(self, http: Any, topic: str) -> list[RawItem]:
        query = f"topic:{topic} stars:>={self.config.min_stars_daily}"
        url = f"{self.base_url}/search/repositories?q={query}&sort=updated&order=desc&per_page=5"
        resp = http.get(url, headers=self.get_headers())
        resp.raise_for_status()

        data = resp.json()
        items: list[RawItem] = []

        for repo in data.get("items", []):
            full_name = repo.get("full_name", "")
            description = repo.get("description") or "No description provided."
            html_url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)

            title = f"{full_name} - {description[:60]}"
            raw_text = sanitize_untrusted_text(
                f"{description}\nStars: {stars}\nLanguage: {repo.get('language')}\nTopics: {', '.join(repo.get('topics', []))}"
            )

            published_at = None
            updated_str = repo.get("updated_at")
            if updated_str:
                try:
                    published_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None

            items.append(
                RawItem(
                    id=RawItem.make_id(SourceType.GITHUB.value, html_url),
                    source_type=SourceType.GITHUB,
                    title=title,
                    url=html_url,
                    published_at=published_at,
                    raw_text=raw_text,
                    metadata={"stars": stars, "topic": topic},
                )
            )

        return items
