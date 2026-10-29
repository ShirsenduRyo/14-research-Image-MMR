"""Frozen image/text embedding backends for a 4 GB GPU-friendly workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class EncoderSettings:
    backend: str = "open_clip"  # open_clip | fashionclip
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "auto"
    batch_size: int = 8
    use_amp: bool = True


def encoder_cache_key(settings: EncoderSettings) -> str:
    """Return the same cache namespace without downloading/initializing model weights."""
    if settings.backend.lower().strip() == "open_clip":
        return f"openclip-{settings.model_name}-{settings.pretrained}".replace("/", "-")
    if settings.backend.lower().strip() == "fashionclip":
        model_id = settings.model_name or "patrickjohncyh/fashion-clip"
        return f"fashionclip-{model_id}".replace("/", "-")
    raise ValueError("backend must be 'open_clip' or 'fashionclip'")


class OpenCLIPEncoder:
    """OpenCLIP wrapper; initializes lazily so metric-only use needs no torch model."""

    def __init__(self, settings: EncoderSettings) -> None:
        import torch
        import open_clip

        self.torch = torch
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.model_name,
            pretrained=settings.pretrained,
            device=self.device,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(settings.model_name)
        self.cache_key = f"openclip-{settings.model_name}-{settings.pretrained}".replace("/", "-")

    @staticmethod
    def _resolve_device(requested: str) -> str:
        import torch

        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return requested

    def _amp_context(self):
        torch = self.torch
        if self.device.startswith("cuda") and self.settings.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        from contextlib import nullcontext
        return nullcontext()

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        torch = self.torch
        chunks: list[np.ndarray] = []
        batch_size = max(1, int(self.settings.batch_size))
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pixels = torch.stack([self.preprocess(image.convert("RGB")) for image in batch]).to(self.device)
            with torch.inference_mode(), self._amp_context():
                embeddings = self.model.encode_image(pixels)
                embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            chunks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            del pixels, embeddings
        if not chunks:
            raise ValueError("Cannot encode an empty image list")
        return np.concatenate(chunks, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        chunks: list[np.ndarray] = []
        batch_size = max(1, int(self.settings.batch_size))
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(list(texts[start : start + batch_size])).to(self.device)
            with torch.inference_mode(), self._amp_context():
                embeddings = self.model.encode_text(tokens)
                embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            chunks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            del tokens, embeddings
        if not chunks:
            raise ValueError("Cannot encode an empty text list")
        return np.concatenate(chunks, axis=0)


class FashionCLIPEEncoder:
    """Hugging Face FashionCLIP wrapper using the released CLIPModel checkpoint."""

    def __init__(self, settings: EncoderSettings) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.settings = settings
        self.device = OpenCLIPEncoder._resolve_device(settings.device)
        model_id = settings.model_name or "patrickjohncyh/fashion-clip"
        self.model_id = model_id
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.cache_key = f"fashionclip-{model_id}".replace("/", "-")

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        torch = self.torch
        chunks: list[np.ndarray] = []
        batch_size = max(1, int(self.settings.batch_size))
        for start in range(0, len(images), batch_size):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixel_values)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            chunks.append(features.cpu().numpy().astype(np.float32, copy=False))
            del pixel_values, inputs, features
        if not chunks:
            raise ValueError("Cannot encode an empty image list")
        return np.concatenate(chunks, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        chunks: list[np.ndarray] = []
        batch_size = max(1, int(self.settings.batch_size))
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(
                text=list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
                features = torch.nn.functional.normalize(features.float(), dim=-1)
            chunks.append(features.cpu().numpy().astype(np.float32, copy=False))
            del inputs, features
        if not chunks:
            raise ValueError("Cannot encode an empty text list")
        return np.concatenate(chunks, axis=0)


def build_encoder(settings: EncoderSettings):
    """Construct the requested frozen encoder backend."""
    backend = settings.backend.lower().strip()
    if backend == "open_clip":
        return OpenCLIPEncoder(settings)
    if backend == "fashionclip":
        return FashionCLIPEEncoder(settings)
    raise ValueError("backend must be 'open_clip' or 'fashionclip'")
