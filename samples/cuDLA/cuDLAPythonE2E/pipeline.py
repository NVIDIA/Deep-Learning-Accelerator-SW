# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end hybrid pipeline: GPU stem -> DLA backbone -> GPU head.

    image [1,3,224,224]
        -> stem  (ONNX Runtime, CUDA EP)  -> [1,64,56,56]     # GPU pre
        -> DLA   (cuda-python + cuDLA)     -> [1,2048,7,7]     # DLA real workload
        -> head  (ONNX Runtime, CUDA EP)  -> [1,1000] logits  # GPU post

The GPU stages use ONNX Runtime, the DLA stage uses NumPy + the
cuda.bindings packages, and pre/post-processing is NumPy.
"""
from __future__ import annotations

import os

from dla_runner import DLABackend
from engine import make_session

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(_HERE, "models")

STEM_ONNX = os.path.join(MODELS, "resnet50_stem.onnx")
HEAD_ONNX = os.path.join(MODELS, "resnet50_head.onnx")
LOADABLE = os.path.join(MODELS, "resnet50_v1_prepared.loadable")


class HybridPipeline:
    """GPU(stem) -> DLA(backbone) -> GPU(head), each stage exposed for timing."""

    def __init__(self, dla_core: int = 0, verbose: bool = False):
        self.stem = make_session(STEM_ONNX)
        self.head = make_session(HEAD_ONNX)
        self.dla = DLABackend(LOADABLE, dla_core=dla_core, verbose=verbose)
        self._stem_in = self.stem.get_inputs()[0].name
        self._head_in = self.head.get_inputs()[0].name

    def run_stem(self, x_np):
        """GPU pre: image -> DLA input feature [1,64,56,56]."""
        return self.stem.run(None, {self._stem_in: x_np})[0]

    def run_dla(self, feat_np):
        """DLA: [1,64,56,56] -> [1,2048,7,7] (the real backbone compute)."""
        return self.dla.infer(feat_np)

    def run_head(self, mid_np):
        """GPU post: DLA output -> logits [1,1000]."""
        return self.head.run(None, {self._head_in: mid_np})[0]

    def __call__(self, x_np):
        return self.run_head(self.run_dla(self.run_stem(x_np)))

    def close(self):
        self.dla.close()
