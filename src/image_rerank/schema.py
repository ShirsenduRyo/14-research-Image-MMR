"""Typed data containers shared by dataset, embedding, and evaluation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class QueryRecord:
    """One query and its fixed candidate shortlist before embeddings are computed."""

    query_id: str
    query: str
    images: list[Any] | None
    image_ids: np.ndarray
    original_scores: np.ndarray
    relevance: np.ndarray
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.image_ids)
        if self.images is not None and len(self.images) != n:
            raise ValueError("images and image_ids must have equal lengths when images are loaded")
        if not (len(self.original_scores) == len(self.relevance) == n):
            raise ValueError("image_ids, original_scores, and relevance must have equal lengths")
        if n == 0:
            raise ValueError("A query must have at least one candidate")
        if not np.isfinite(np.asarray(self.original_scores, dtype=float)).all():
            raise ValueError("original_scores contains NaN or infinite values")
        if not np.isin(np.asarray(self.relevance), [0, 1]).all():
            raise ValueError("relevance labels must be binary (0/1)")
        for name, values in self.metadata.items():
            if len(values) != n:
                raise ValueError(f"Metadata field {name!r} has {len(values)} rows, expected {n}")


@dataclass(slots=True)
class QueryEmbeddings:
    """A query and candidate embeddings, plus labels and metadata for evaluation."""

    query_id: str
    query: str
    query_embedding: np.ndarray
    image_embeddings: np.ndarray
    image_ids: np.ndarray
    original_scores: np.ndarray
    relevance: np.ndarray
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_embedding = np.asarray(self.query_embedding, dtype=np.float32).reshape(-1)
        self.image_embeddings = np.asarray(self.image_embeddings, dtype=np.float32)
        self.image_ids = np.asarray(self.image_ids)
        self.original_scores = np.asarray(self.original_scores, dtype=np.float32)
        self.relevance = np.asarray(self.relevance, dtype=np.int8)
        if self.image_embeddings.ndim != 2:
            raise ValueError("image_embeddings must have shape (n_candidates, embedding_dim)")
        n = self.image_embeddings.shape[0]
        if self.query_embedding.shape[0] != self.image_embeddings.shape[1]:
            raise ValueError("Query and image embedding dimensions do not match")
        if not (len(self.image_ids) == len(self.original_scores) == len(self.relevance) == n):
            raise ValueError("Candidate arrays must have equal lengths")
        if n == 0:
            raise ValueError("A query must have at least one candidate")
        if not np.isfinite(self.query_embedding).all() or not np.isfinite(self.image_embeddings).all():
            raise ValueError("Embeddings contain NaN or infinite values")
        if not np.isfinite(self.original_scores).all():
            raise ValueError("original_scores contains NaN or infinite values")
        if not np.isin(self.relevance, [0, 1]).all():
            raise ValueError("relevance labels must be binary (0/1)")
        for name, values in self.metadata.items():
            if len(values) != n:
                raise ValueError(f"Metadata field {name!r} has {len(values)} rows, expected {n}")

    @property
    def encoder_scores(self) -> np.ndarray:
        """Cosine relevance scores; embeddings are normalized by the encoder."""
        q = self.query_embedding / max(float(np.linalg.norm(self.query_embedding)), 1e-12)
        x = self.image_embeddings / np.maximum(
            np.linalg.norm(self.image_embeddings, axis=1, keepdims=True), 1e-12
        )
        return (x @ q).astype(np.float32)

    @property
    def normalized_image_embeddings(self) -> np.ndarray:
        norms = np.maximum(np.linalg.norm(self.image_embeddings, axis=1, keepdims=True), 1e-12)
        return self.image_embeddings / norms
