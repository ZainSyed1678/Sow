"""Collectors package for arXiv, RSS feeds, and GitHub."""

from ai_ml_notion_researcher.collectors.arxiv import ArxivCollector
from ai_ml_notion_researcher.collectors.base import BaseCollector
from ai_ml_notion_researcher.collectors.github import GithubCollector
from ai_ml_notion_researcher.collectors.manager import CollectorManager
from ai_ml_notion_researcher.collectors.rss import RSSCollector

__all__ = [
    "ArxivCollector",
    "BaseCollector",
    "CollectorManager",
    "GithubCollector",
    "RSSCollector",
]
