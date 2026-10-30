#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone end-to-end ResNet-50 classifier: GPU stem -> DLA backbone -> GPU head.

Classifies an image by running the heavy ResNet-50 backbone on the Jetson **DLA**
via cuda-python + cuDLA, with the stem/head on the GPU via ONNX Runtime.
Prints the predicted breed and writes an annotated image.

Examples:
  ./venv/bin/python cudla_e2e_app/run.py
  ./venv/bin/python cudla_e2e_app/run.py --image my_dog.jpg --topk 5
  ./venv/bin/python cudla_e2e_app/run.py --dla-core 1 --validate
"""
from __future__ import annotations

import os
import sys

# Put CUDA + cuDLA libs on LD_LIBRARY_PATH BEFORE importing cuda/onnxruntime.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import ensure_libs

ensure_libs()

import argparse
import time

import numpy as np

from imagenet import load_labels, preprocess, softmax, topk
from pipeline import HybridPipeline

_HERE = os.path.dirname(os.path.abspath(__file__))
DEF_IMAGE = os.path.join(_HERE, "assets", "dog.jpg")
LABELS = os.path.join(_HERE, "assets", "synset.txt")
FULL_ONNX = os.path.join(_HERE, "models", "resnet50-v1-12.onnx")
DEF_OUT = os.path.join(_HERE, "output", "classified.jpg")


def annotate(image_path: str, text: str, out_path: str) -> None:
    """Draw the predicted label on the image and save it (headless-safe)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    size = max(18, img.width // 22)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            font = ImageFont.load_default()

    pad = size // 3
    try:
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
    except AttributeError:  # very old Pillow
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(0, 255, 0), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def validate_full_gpu(x_np, labels, k: int) -> None:
    """Sanity: run the whole model on the GPU (ONNX Runtime) for comparison."""
    if not os.path.exists(FULL_ONNX):
        print("  [validate] skipped (full ONNX not present)")
        return
    from engine import make_session

    sess = make_session(FULL_ONNX)
    logits = sess.run(None, {sess.get_inputs()[0].name: x_np})[0]
    top = topk(logits, labels, k)
    print("  [validate] full-model GPU (ONNX Runtime) top-1: "
          f"[{top[0][0]}] {top[0][1]} {top[0][2] * 100:.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEF_IMAGE, help="input image (default: dog.jpg)")
    ap.add_argument("--dla-core", type=int, default=0, choices=[0, 1],
                    help="which Orin DLA core to run the backbone on (default: 0)")
    ap.add_argument("--topk", type=int, default=5, help="how many predictions to print")
    ap.add_argument("--out", default=DEF_OUT, help="annotated output image path")
    ap.add_argument("--validate", action="store_true",
                    help="also run the full model on the GPU as a cross-check")
    ap.add_argument("--verbose", action="store_true",
                    help="print cuDLA handles, tensor descriptors, and submission/copy progress")
    args = ap.parse_args()

    labels = load_labels(LABELS)
    x = preprocess(args.image)

    print(f"Image      : {args.image}")
    print(f"DLA core   : {args.dla_core}")
    print("Pipeline   : GPU stem (ORT) -> DLA backbone (cuDLA) -> GPU head (ORT)")

    pipe = HybridPipeline(dla_core=args.dla_core, verbose=args.verbose)
    try:
        pipe(x)  # warm-up (build kernels, load DLA) before timing
        t0 = time.perf_counter()
        logits = pipe(x)
        dt = (time.perf_counter() - t0) * 1000.0

        preds = topk(logits, labels, args.topk)
        print(f"\nResult (in {dt:.1f} ms):")
        for rank, (idx, name, prob) in enumerate(preds, 1):
            mark = " <-- prediction" if rank == 1 else ""
            print(f"  {rank}. [{idx:4d}] {name:<28} {prob * 100:6.2f}%{mark}")

        top_idx, top_name, top_prob = preds[0]
        annotate(args.image, f"{top_name}  {top_prob * 100:.1f}%", args.out)
        print(f"\nAnnotated image written to: {args.out}")

        if args.validate:
            print("\nValidation:")
            validate_full_gpu(x, labels, args.topk)
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
