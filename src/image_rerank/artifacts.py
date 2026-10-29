"""Artifact paths and safe, deterministic persistence helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return value[:120] or "item"


def query_cache_path(root: str | Path, split: str, encoder_key: str, query_id: str) -> Path:
    return ensure_dir(Path(root) / safe_name(split) / safe_name(encoder_key)) / f"{safe_name(query_id)}.npz"


def save_query_embeddings(path: str | Path, query_data: Any) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    metadata = {key: np.asarray(value).astype(str) for key, value in query_data.metadata.items()}
    payload: dict[str, np.ndarray] = {
        "query_id": np.asarray(query_data.query_id),
        "query": np.asarray(query_data.query),
        "query_embedding": np.asarray(query_data.query_embedding, dtype=np.float16),
        "image_embeddings": np.asarray(query_data.image_embeddings, dtype=np.float16),
        "image_ids": np.asarray(query_data.image_ids).astype(str),
        "original_scores": np.asarray(query_data.original_scores, dtype=np.float32),
        "relevance": np.asarray(query_data.relevance, dtype=np.int8),
    }
    for key, values in metadata.items():
        payload[f"metadata__{safe_name(key)}"] = values
    # Write in the same directory and atomically replace the destination so an
    # interrupted process cannot leave a truncated cache that appears complete.
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz.tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            np.savez_compressed(temporary_file, **payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_query_embeddings(path: str | Path):
    from .schema import QueryEmbeddings

    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = {
            key.removeprefix("metadata__"): payload[key].astype(str)
            for key in payload.files if key.startswith("metadata__")
        }
        return QueryEmbeddings(
            query_id=str(payload["query_id"].item()),
            query=str(payload["query"].item()),
            query_embedding=payload["query_embedding"].astype(np.float32),
            image_embeddings=payload["image_embeddings"].astype(np.float32),
            image_ids=payload["image_ids"].astype(str),
            original_scores=payload["original_scores"].astype(np.float32),
            relevance=payload["relevance"].astype(np.int8),
            metadata=metadata,
        )


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".json.tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=str)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=f".tmp{destination.suffix}", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if destination.suffix.lower() == ".parquet":
            frame.to_parquet(temporary_path, index=False)
        else:
            frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination
