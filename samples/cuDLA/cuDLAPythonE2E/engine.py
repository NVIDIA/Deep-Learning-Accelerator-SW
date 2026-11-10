# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX Runtime (CUDA EP) session builder for the GPU stem/head stages.

The DLA runs only the ResNet-50 *backbone*. The lightweight **stem** (conv/bn/
relu/maxpool) and **head** (globalavgpool/flatten/gemm) are real neural-network
layers, so they run on the GPU through ONNX Runtime's CUDA execution provider.
NumPy is just the data container at the API boundary; the math runs on the GPU.
"""
from __future__ import annotations

import onnxruntime as ort


def make_session(model_path: str) -> ort.InferenceSession:
    """Build a GPU (CUDA EP) inference session for an ONNX subgraph.

    log_severity_level=3 (ERROR) silences ORT's chatty provider/optimizer logs so
    the console shows the classification result, not noise. CPU is listed only as
    a last-resort fallback for any unsupported op (there are none for stem/head).
    """
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(
        model_path, sess_options=so,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
