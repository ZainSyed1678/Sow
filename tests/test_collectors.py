"""Unit tests for research collectors (arXiv, RSS, GitHub) with mocked external APIs."""

from pathlib import Path
from unittest.mock import MagicMock

from ai_ml_notion_researcher.collectors.arxiv import ArxivCollector
from ai_ml_notion_researcher.collectors.base import BaseCollector
from ai_ml_notion_researcher.collectors.github import GithubCollector
from ai_ml_notion_researcher.collectors.manager import CollectorManager
from ai_ml_notion_researcher.collectors.rss import RSSCollector
from ai_ml_notion_researcher.config import (
    AppConfig,
    ArxivSourceConfig,
    GithubSourceConfig,
    RSSFeedItemConfig,
    RSSSourceConfig,
)
from ai_ml_notion_researcher.models import RawItem, SourceType
from ai_ml_notion_researcher.storage.sqlite_store import SQLiteStore

MOCK_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title> FlashAttention-3: Fast and Accurate Attention on GPUs </title>
    <summary> We introduce FlashAttention-3, utilizing asynchronous tensor core operations. </summary>
    <published>2026-10-10T12:00:00Z</published>
    <author><name>Tri Dao</name></author>
    <author><name>Dan Fu</name></author>
  </entry>
</feed>
"""

MOCK_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Hugging Face Blog</title>
    <item>
      <title>Announcing Transformers v4.45</title>
      <link>https://huggingface.co/blog/transformers-4-45</link>
      <description>&lt;p&gt;Major release featuring QLoRA speedups and torch.compile support.&lt;/p&gt;</description>
      <pubDate>Mon, 12 Oct 2026 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

MOCK_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Google DeepMind Research Milestone</title>
    <link href="https://deepmind.google/blog/new-reasoning-model" />
    <summary>A new frontier reasoning system evaluated across scientific benchmarks.</summary>
    <published>2026-10-11T14:30:00Z</published>
  </entry>
</feed>
"""


def test_arxiv_collector_mocked():
    mock_http = MagicMock()
    mock_resp = MagicMock(status_code=200, text=MOCK_ARXIV_XML)
    mock_http.get.return_value = mock_resp
    mock_http.__enter__.return_value = mock_http

    cfg = ArxivSourceConfig(enabled=True, categories=["cs.AI"], max_results_per_category=5)
    collector = ArxivCollector(config=cfg, client=mock_http)

    items = collector.collect()
    assert len(items) == 1
    assert items[0].source_type == SourceType.ARXIV
    assert "FlashAttention-3" in items[0].title
    assert items[0].url == "https://arxiv.org/abs/2401.00001v1"
    assert "Tri Dao" in items[0].authors


def test_rss_collector_mocked():
    mock_http = MagicMock()
    # Return RSS 2.0 on first call, Atom on second call
    mock_resp_rss = MagicMock(status_code=200, text=MOCK_RSS_XML)
    mock_resp_atom = MagicMock(status_code=200, text=MOCK_ATOM_XML)
    mock_http.get.side_effect = [mock_resp_rss, mock_resp_atom]
    mock_http.__enter__.return_value = mock_http

    cfg = RSSSourceConfig(
        enabled=True,
        feeds=[
            RSSFeedItemConfig(name="HF", url="https://hf.co/feed"),
            RSSFeedItemConfig(name="DeepMind", url="https://deepmind.com/feed"),
        ],
    )
    collector = RSSCollector(config=cfg, client=mock_http)

    items = collector.collect()
    assert len(items) == 2
    assert items[0].title == "Announcing Transformers v4.45"
    assert "QLoRA speedups" in items[0].raw_text
    assert items[1].title == "Google DeepMind Research Milestone"


def test_github_collector_mocked():
    mock_http = MagicMock()
    # Release response
    mock_rel_resp = MagicMock(status_code=200)
    mock_rel_resp.json.return_value = {
        "tag_name": "v0.6.0",
        "name": "vLLM 0.6.0 with Speculative Decoding",
        "html_url": "https://github.com/vllm-project/vllm/releases/tag/v0.6.0",
        "body": "This release adds multi-GPU speculative decoding and chunked prefill.",
        "published_at": "2026-10-12T16:00:00Z",
    }

    # Search response
    mock_search_resp = MagicMock(status_code=200)
    mock_search_resp.json.return_value = {
        "items": [
            {
                "full_name": "meta-llama/llama-models",
                "description": "Utilities and weights for Llama models",
                "html_url": "https://github.com/meta-llama/llama-models",
                "stargazers_count": 2500,
                "language": "Python",
                "topics": ["llm", "ai"],
                "updated_at": "2026-10-14T10:00:00Z",
            }
        ]
    }

    mock_http.get.side_effect = [mock_rel_resp, mock_search_resp]
    mock_http.__enter__.return_value = mock_http

    cfg = GithubSourceConfig(
        enabled=True,
        tracked_repos=["vllm-project/vllm"],
        topics=["llm"],
    )
    collector = GithubCollector(config=cfg, client=mock_http)

    items = collector.collect()
    assert len(items) == 2
    assert items[0].source_type == SourceType.GITHUB
    assert "vLLM 0.6.0" in items[0].title
    assert "speculative decoding" in items[0].raw_text.lower()


def test_collector_manager_fault_tolerance(tmp_path: Path):
    # Collector 1 throws an error
    faulty_collector = MagicMock(spec=BaseCollector)
    faulty_collector.safe_collect.return_value = []

    # Collector 2 succeeds
    working_collector = MagicMock(spec=BaseCollector)
    good_item = RawItem(
        id="item-good-1",
        source_type=SourceType.ARXIV,
        title="Quantization for LLMs",
        url="https://arxiv.org/abs/2401.99999",
        raw_text="A novel INT4 quantization scheme with zero accuracy loss for Llama 3.",
    )
    working_collector.safe_collect.return_value = [good_item]

    store = SQLiteStore(db_path=tmp_path / "test.db")
    app_cfg = AppConfig()

    manager = CollectorManager(
        config=app_cfg,
        store=store,
        collectors=[faulty_collector, working_collector],
    )

    prioritized = manager.collect_and_prioritize()
    assert len(prioritized) == 1
    assert prioritized[0].item.id == "item-good-1"
    assert prioritized[0].relevance_score >= 0.65
