# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic CUDA host/device plumbing (cuda-python driver + runtime).

None of this is DLA-specific -- it is the ordinary CUDA bookkeeping that
surrounds *any* accelerator call: initialise CUDA, allocate device buffers,
create a stream, and copy bytes host<->device. It is deliberately factored out
of ``dla_runner.py`` so that the cuDLA code there reads as a clean, uninterrupted
story instead of being interleaved with malloc/memcpy/stream calls.

If you are here to learn the **cuDLA** API, you can skip this file -- read
``dla_runner.py`` instead. This is just the supporting cast.
"""
from __future__ import annotations

from cuda.bindings import driver, runtime


def check(result):
    """Unwrap a cuda-python ``(err, *vals)`` tuple, raising on a non-success code.

    Every ``cuda.bindings`` call returns ``(error_code, *return_values)``. This
    raises if the code is non-zero and returns the lone value (or tuple of them).
    """
    if not isinstance(result, (tuple, list)):
        return None
    err = result[0]
    if hasattr(err, "value") and err.value:
        raise RuntimeError(f"CUDA error: {err}")
    rest = result[1:]
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


def init(device: int = 0) -> None:
    """cuInit + set the active CUDA device (cuDLA hybrid mode rides on this)."""
    check(driver.cuInit(0))
    check(runtime.cudaSetDevice(device))


def alloc(nbytes: int) -> int:
    """Allocate ``nbytes`` of device memory; return the raw device pointer."""
    return int(check(runtime.cudaMalloc(nbytes)))


def free(ptr: int) -> None:
    runtime.cudaFree(ptr)


def create_stream() -> int:
    return int(check(runtime.cudaStreamCreate()))


def destroy_stream(stream: int) -> None:
    runtime.cudaStreamDestroy(stream)


def sync(stream: int) -> None:
    """Block until all work queued on the stream has finished."""
    check(runtime.cudaStreamSynchronize(stream))


def copy_h2d(dst_dev_ptr: int, src_host_ptr: int, nbytes: int, stream: int) -> None:
    """Host -> device copy on the given stream (call ``sync`` to wait)."""
    check(runtime.cudaMemcpyAsync(
        dst_dev_ptr, src_host_ptr, nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))


def copy_d2h(dst_host_ptr: int, src_dev_ptr: int, nbytes: int, stream: int) -> None:
    """Device -> host copy on the given stream (call ``sync`` to wait)."""
    check(runtime.cudaMemcpyAsync(
        dst_host_ptr, src_dev_ptr, nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
