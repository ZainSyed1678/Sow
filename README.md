# AI/ML Notion Researcher

Production-ready, autonomous Python application that researches AI/ML developments, synthesizes high-quality educational notes using Groq LLM inference, and publishes structured notes directly into Notion.

## Pipeline Architecture

1. **Research Collection**: Gathers arXiv papers, RSS feeds (HuggingFace, OpenAI, DeepMind, BAIR), and GitHub releases/trending tools.
2. **Normalization & Scoring**: Classifies incoming material by topic, novelty, and relevance.
3. **Deduplication**: Prevents duplicate entries using canonical URL/content hashing and Notion database querying.
4. **LLM Synthesis**: Generates 11-part structured educational notes (layman terms, technical mechanisms, architectures, trade-offs, code exercises, interview prep).
5. **Pydantic Validation**: Validates the output before any publication.
6. **Idempotent Notion Publishing**: Safely uploads structured blocks, callouts, and code snippets to the designated Notion database.

## Target Notion Databases

1. **Daily Briefings**: Crucial developments, summaries, practical implications, and source links.
2. **Concept Library**: Core machine learning concepts, mechanics, mathematical intuition, and trade-offs.
3. **Research Papers**: Problem statements, proposed methodologies, key benchmarks, limitations, and relevance.
4. **Tools and Frameworks**: Framework explanations, where they fit in the stack, alternatives, and code examples.
5. **Weekly Reviews**: Weekly synthesis connecting concepts, summarizing lessons, and recommending practical builds.

## Setup & Quickstart

### 1. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your API keys and Notion database IDs:
- `GROQ_API_KEY`: Groq API key (from console.groq.com)
- `NOTION_API_KEY`: Notion Internal Integration Secret (from notion.so/my-integrations)
- `NOTION_DATABASE_ID_*`: IDs for each of the 5 Notion databases

### 2. Run Locally in Dry-Run Mode

By default, the application runs safely with `DRY_RUN=true` to test the pipeline without modifying your Notion workspace:

```bash
python -m ai_ml_notion_researcher.main --dry-run
```

### 3. Run Tests

```bash
pytest
```
