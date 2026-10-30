# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a compiled DLA loadable from Python with the **cuDLA** API.

============================================================================
THIS FILE IS THE POINT OF THE DEMO.
Everything here that starts with ``cudla.`` is the cuDLA Python API. Generic
CUDA memory/stream bookkeeping is delegated to ``cuda_mem.py`` and the FP16
surface reshaping to ``packing.py``, so that what remains below is *only* the
cuDLA story, in order.
============================================================================

The cuDLA lifecycle, start to finish -- the entire API surface used:

    setup (once) ─ cudla.create_device            open a handle to a DLA core
                 ├ cudla.module_load_from_memory  load the compiled loadable
                 ├ cudla.module_get_attributes    read the I/O tensor descriptors
                 ├ cudla.mem_register             let the DLA see our GPU buffers
                 └ cudla.Task                      bind module + registered I/O

    per image    ─ cudla.submit_task              run the loadable on the DLA

    teardown     ─ cudla.mem_unregister
                 ├ cudla.module_unload
                 └ cudla.destroy_device

The loadable here is a single-input/single-output FP16 ResNet-50 backbone,
``[1,64,56,56] -> [1,2048,7,7]`` in the DLA's CHW16 surface layout.
"""
from __future__ import annotations

import numpy as np
from cuda.bindings import cudla

import cuda_mem
from packing import pack_chw16, unpack_chw16


class DLABackend:
    """NumPy in / NumPy out runner for a single-IO FP16 DLA loadable.

    Read ``__init__`` and ``infer`` top-to-bottom: the ``cudla.*`` calls are the
    cuDLA API; the ``cuda_mem.*`` calls are ordinary CUDA plumbing kept out of
    the way. See ``cuda_mem.py`` only if you care about the malloc/memcpy details.
    """

    def __init__(self, loadable_path: str, dla_core: int = 0, verbose: bool = False):
        self.verbose = verbose
        # CUDA must be initialised before any cuDLA call (hybrid mode rides on
        # the CUDA context). Details in cuda_mem.py.
        cuda_mem.init(device=0)

        # ---- cuDLA 1: open a handle to a DLA core (0 or 1 on Orin) ----
        self.dev = cudla.create_device(dla_core, int(cudla.Mode.CUDA_DLA))
        if self.verbose:
            print(f"cuDLA device created: {int(self.dev):#x}")

        # ---- cuDLA 2: load the pre-compiled loadable into a module ----
        with open(loadable_path, "rb") as f:
            loadable_bytes = f.read()
        self.mod = cudla.module_load_from_memory(
            self.dev, loadable_bytes, len(loadable_bytes), 0)
        if self.verbose:
            print(f"Module loaded: {int(self.mod):#x}")

        # ---- cuDLA 3: read the input/output tensor descriptors ----
        # These tell us the CHW16 byte size to allocate and the logical NCHW
        # shape to unpack back to.
        in_descs = cudla.module_get_attributes(
            self.mod, int(cudla.ModuleAttributeType.INPUT_TENSOR_DESCRIPTORS))
        out_descs = cudla.module_get_attributes(
            self.mod, int(cudla.ModuleAttributeType.OUTPUT_TENSOR_DESCRIPTORS))
        if self.verbose:
            print(f"numInputTensors = {len(in_descs)}")
            print(f"numOutputTensors = {len(out_descs)}")
            for kind, descriptors in (("Input", in_descs), ("Output", out_descs)):
                for i, desc in enumerate(descriptors):
                    shape = (desc.n, desc.c, desc.h, desc.w)
                    print(f"{kind} {i}: size={int(desc.size_)}, shape={shape}, name={desc.name}")
        in_desc, out_desc = in_descs[0], out_descs[0]
        self.in_shape = (in_desc.n, in_desc.c, in_desc.h, in_desc.w)
        self.out_shape = (out_desc.n, out_desc.c, out_desc.h, out_desc.w)
        self.in_size = int(in_desc.size_)    # bytes of the CHW16 input surface
        self.out_size = int(out_desc.size_)  # bytes of the CHW16 output surface

        # Allocate the device buffers the DLA will read from / write to
        # (plain CUDA malloc; see cuda_mem.py).
        self.in_gpu = cuda_mem.alloc(self.in_size)
        self.out_gpu = cuda_mem.alloc(self.out_size)

        # ---- cuDLA 4: register those buffers so the DLA engine can access them ----
        self.in_reg = cudla.mem_register(self.dev, self.in_gpu, self.in_size, 0)
        self.out_reg = cudla.mem_register(self.dev, self.out_gpu, self.out_size, 0)

        # A CUDA stream to order the copies + submit on (cuDLA rejects handle 0).
        self.stream = cuda_mem.create_stream()

        # ---- cuDLA 5: build a reusable Task binding the module to its I/O ----
        self.task = cudla.Task()
        self.task.module_handle = self.mod
        self.task.input_tensor = [int(self.in_reg)]
        self.task.output_tensor = [int(self.out_reg)]
        self.task.wait_events = 0
        self.task.signal_events = 0

        self._host_out = np.empty(self.out_size, dtype=np.uint8)

    def infer(self, x_nchw_fp32: np.ndarray) -> np.ndarray:
        """Run the loadable on the DLA: NumPy ``[1,Cin,H,W]`` -> ``[1,Cout,H,W]``."""
        # Prepare + upload the input (NumPy pack + CUDA H2D copy).
        packed = pack_chw16(x_nchw_fp32.astype(np.float16))
        assert packed.nbytes == self.in_size, (packed.nbytes, self.in_size)
        # H2D -> submit -> D2H are all on the same stream (FIFO), so only the
        # final sync below is needed before the CPU reads the host output.
        cuda_mem.copy_h2d(self.in_gpu, packed.ctypes.data, self.in_size, self.stream)

        # ---- cuDLA 6: submit the task -- THIS is the DLA running the network ----
        cudla.submit_task(self.dev, int(self.task), 1, int(self.stream), 0)
        if self.verbose:
            print("SUBMIT IS DONE (task queued; completion pending)")

        # Download + unpack the output (CUDA D2H copy + NumPy unpack).
        cuda_mem.copy_d2h(self._host_out.ctypes.data, self.out_gpu,
                          self.out_size, self.stream)
        cuda_mem.sync(self.stream)
        if self.verbose:
            print("Output tensor written to host")
        _, c, h, w = self.out_shape
        return unpack_chw16(self._host_out.view(np.float16), c, h, w).astype(np.float32)

    def close(self) -> None:
        """Release everything, cuDLA handles first, then the CUDA resources."""
        try:
            # ---- cuDLA teardown ----
            cudla.mem_unregister(self.dev, self.in_reg)
            cudla.mem_unregister(self.dev, self.out_reg)
            cudla.module_unload(self.mod, 0)
            cudla.destroy_device(self.dev)
            # ---- generic CUDA teardown ----
            cuda_mem.free(self.in_gpu)
            cuda_mem.free(self.out_gpu)
            cuda_mem.destroy_stream(self.stream)
        except Exception:
            pass
