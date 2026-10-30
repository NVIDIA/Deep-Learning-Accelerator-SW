# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NumPy NCHW <-> CHW16 surface packing for DLA FP16 loadables.

The DLA reads/writes FP16 tensors in the **CHW16** surface layout
(``[C/16, H, W, 16]``), not plain NCHW. These helpers only *reshuffle memory
order* -- the numbers are unchanged, there is no math here. Kept separate from
``dla_runner.py`` so the cuDLA code there stays uncluttered.
"""
from __future__ import annotations

import numpy as np


def pack_chw16(x_nchw: np.ndarray) -> np.ndarray:
    """NCHW fp16 -> contiguous CHW16 ``[C/16, H, W, 16]`` fp16 (N=1).

    Split C into (C/16, 16) and move the 16-channel group to the innermost axis.
    """
    n, c, h, w = x_nchw.shape
    assert n == 1 and c % 16 == 0, f"need N=1 and C%16==0, got {x_nchw.shape}"
    packed = x_nchw.reshape(c // 16, 16, h, w).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(packed, dtype=np.float16)


def unpack_chw16(buf: np.ndarray, c: int, h: int, w: int) -> np.ndarray:
    """CHW16 ``[C/16,H,W,16]`` fp16 -> NCHW fp16 (N=1)."""
    out = buf.reshape(c // 16, h, w, 16).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(out.reshape(1, c, h, w), dtype=np.float16)
