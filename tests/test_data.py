from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from image_rerank.data import iter_local_manifest


def test_local_manifest_loader(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_paths = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        path = images_dir / f"{index}.png"
        Image.new("RGB", (8, 8), color=color).save(path)
        image_paths.append(str(path.relative_to(tmp_path)))
    frame = pd.DataFrame({
        "query": ["red item", "red item"],
        "image_path": image_paths,
        "relevant": [1, 0],
        "initial_score": [0.8, 0.4],
        "category": ["shirt", "pants"],
    })
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)

    record = next(iter_local_manifest(str(manifest)))
    assert record.query == "red item"
    assert len(record.images) == 2
    assert record.relevance.tolist() == [1, 0]
    assert record.metadata["category"].tolist() == ["shirt", "pants"]


def test_local_manifest_rejects_missing_columns(tmp_path: Path):
    manifest = tmp_path / "bad.csv"
    pd.DataFrame({"query": ["x"]}).to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        list(iter_local_manifest(str(manifest)))
