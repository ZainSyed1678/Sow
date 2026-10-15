"""Tests for configuration loading and validation."""

import os
from pathlib import Path
from unittest.mock import patch

from ai_ml_notion_researcher.config import load_config
from ai_ml_notion_researcher.models import NotionDatabaseType


def test_load_default_config():
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root=project_root)

    assert config.topic == "ai_ml"
    assert config.llm.provider == "groq"
    assert config.llm.model == "llama-3.3-70b-versatile"
    assert config.pipeline.dry_run is True
    assert config.pipeline.max_items_to_score == 50
    assert len(config.sources.arxiv.categories) >= 3
    assert len(config.sources.rss_feeds.feeds) >= 2


def test_env_override_config():
    project_root = Path(__file__).resolve().parent.parent

    with patch.dict(
        os.environ,
        {
            "DRY_RUN": "false",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "GROQ_API_KEY": "gsk_test_mock_key_12345",
            "TOPIC": "quantum_ml",
        },
        clear=False,
    ):
        config = load_config(project_root=project_root)
        assert config.pipeline.dry_run is False
        assert config.llm.model == "llama-3.1-8b-instant"
        assert config.llm.api_key is not None
        assert config.llm.api_key.get_secret_value() == "gsk_test_mock_key_12345"
        assert config.topic == "quantum_ml"


def test_notion_publishing_validation():
    project_root = Path(__file__).resolve().parent.parent

    # Without keys, is_configured_for_publishing should be False
    with patch.dict(os.environ, {}, clear=True):
        config = load_config(project_root=project_root)
        assert config.notion.is_configured_for_publishing() is False

    # With complete keys and database IDs, it should be True
    full_env = {
        "NOTION_API_KEY": "ntn_test_key_valid",
        "NOTION_DATABASE_ID_DAILY_BRIEFINGS": "00000000000000000000000000000001",
        "NOTION_DATABASE_ID_CONCEPT_LIBRARY": "00000000000000000000000000000002",
        "NOTION_DATABASE_ID_RESEARCH_PAPERS": "00000000000000000000000000000003",
        "NOTION_DATABASE_ID_TOOLS_FRAMEWORKS": "00000000000000000000000000000004",
        "NOTION_DATABASE_ID_WEEKLY_REVIEWS": "00000000000000000000000000000005",
    }
    with patch.dict(os.environ, full_env, clear=True):
        config = load_config(project_root=project_root)
        assert config.notion.is_configured_for_publishing() is True
        assert (
            config.notion.database_ids[NotionDatabaseType.DAILY_BRIEFINGS]
            == "00000000000000000000000000000001"
        )
