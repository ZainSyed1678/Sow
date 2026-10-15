"""Pytest fixtures and test setup."""

from datetime import UTC, datetime

import pytest

from ai_ml_notion_researcher.models import (
    EducationalNote,
    NotionDatabaseType,
    RawItem,
    SourceType,
    TradeOffs,
)


@pytest.fixture
def sample_raw_item() -> RawItem:
    return RawItem(
        id=RawItem.make_id("arxiv", "https://arxiv.org/abs/2301.00001"),
        source_type=SourceType.ARXIV,
        title="Attention Is All You Need: Modern Insights",
        url="https://arxiv.org/abs/2301.00001",
        published_at=datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC),
        raw_text="A comprehensive retrospective on transformer architectures and self-attention mechanisms.",
        authors=["Alice Smith", "Bob Jones"],
        metadata={"category": "cs.CL"},
    )


@pytest.fixture
def sample_educational_note() -> EducationalNote:
    return EducationalNote(
        title="Transformer Architecture & Multi-Head Self-Attention",
        summary="A foundational deep learning architecture relying solely on attention mechanisms.",
        target_database=NotionDatabaseType.CONCEPT_LIBRARY,
        canonical_id="concept-transformer-001",
        layman_explanation=(
            "Imagine a team reading a long book together. Instead of reading word by word, "
            "each person highlights words related to what they are thinking about and connects them instantly."
        ),
        technical_explanation=(
            "Transformers replace recurrence with multi-head self-attention, computing Q, K, V "
            "projections with scaled dot-product attention softmax(QK^T / sqrt(d_k))V."
        ),
        step_by_step_mechanism=[
            "1. Tokenize input strings and add learned positional encodings.",
            "2. Project embeddings into Query, Key, and Value vectors across H heads.",
            "3. Compute scaled dot-product attention weights across all sequence positions.",
            "4. Concatenate multi-head outputs and pass through feedforward networks with residual connections.",
        ],
        real_world_use_cases=[
            "Large language model pre-training (e.g., Llama 3, GPT-4)",
            "Code generation and semantic code search",
            "Protein structure prediction (AlphaFold)",
        ],
        technologies_and_interactions=[
            "PyTorch / JAX for tensor acceleration",
            "FlashAttention for hardware-aware kernel optimization",
            "vLLM / TensorRT-LLM for high-throughput inference serving",
        ],
        architecture_diagram=(
            "Input Tokens -> Positional Encoding -> Multi-Head Attention -> Add & Norm -> FFN -> Add & Norm"
        ),
        trade_offs=TradeOffs(
            advantages=[
                "High degree of parallelization during training",
                "Captures long-range dependencies better than RNNs",
            ],
            limitations=[
                "O(N^2) quadratic memory complexity with naive attention",
                "High GPU memory footprint during KV-cache generation",
            ],
            trade_offs=[
                "Quadratic accuracy vs linear attention approximations",
                "Model parameter scale vs inference latency/memory requirements",
            ],
        ),
        practical_exercise=(
            "Implement a minimal scaled dot-product attention function in PyTorch in under 20 lines of code."
        ),
        interview_takeaways=[
            "Why divide by sqrt(d_k)? To avoid vanishing gradients in softmax for large dimensions.",
            "What is the difference between causal masking and bidirectional attention?",
            "How does FlashAttention achieve 2-4x speedup without changing mathematical output?",
        ],
        source_links=[
            "https://arxiv.org/abs/1706.03762",
            "https://github.com/huggingface/transformers",
        ],
        tags=["Deep Learning", "Transformers", "NLP", "LLM"],
    )
