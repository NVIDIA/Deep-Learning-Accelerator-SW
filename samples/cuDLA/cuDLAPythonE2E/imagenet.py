# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ImageNet pre/post-processing for ResNet-50-v1 (all NumPy + PIL).

These are the "front door" and "exit door" of the pipeline: a JPEG becomes the
model's input tensor, and raw logits become a human-readable class name. Both
are light CPU work, so plain NumPy is the natural fit -- no engine needed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # older Pillow
    _BILINEAR = Image.BILINEAR


def preprocess(image_path: str) -> np.ndarray:
    """JPEG -> [1,3,224,224] fp32 NCHW: resize(256) -> center-crop(224) -> normalize."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = 256.0 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), _BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return np.ascontiguousarray(arr, dtype=np.float32)


def load_labels(path: str) -> list[str]:
    """Load ILSVRC-2012 names from an ONNX Model Zoo synset.txt.

    Each line is ``nXXXXXXXX <name>[, synonym...[, scientific...]]``. We keep
    the first human-readable name after the WordNet ID (unchanged upstream
    file; parsing only).
    """
    labels: list[str] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rest = ln.split(" ", 1)[1] if " " in ln else ln
            labels.append(rest.split(",", 1)[0].strip())
    return labels


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def topk(logits: np.ndarray, labels: list[str], k: int = 5):
    """Return the top-k [(index, label, probability)] from a [1,1000] logits array."""
    probs = softmax(logits)[0]
    idx = np.argsort(-probs)[:k]
    return [(int(i), labels[int(i)], float(probs[int(i)])) for i in idx]
