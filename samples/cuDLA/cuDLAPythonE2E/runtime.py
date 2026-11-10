# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime environment bootstrap for the standalone cuDLA app.

The Jetson-built cuda-python / onnxruntime-gpu bindings need both the
pip-installed CUDA libs (cuDNN/cuBLAS/...) AND the system cuDLA libs on
``LD_LIBRARY_PATH`` *before* they are imported. ``ensure_libs()`` augments the
loader path and re-execs the process once (idempotent via an env flag), so the
entry scripts can just call it at the top and then import cuda/onnxruntime.
"""
from __future__ import annotations

import os
import sys

_FLAG = "_CUDLA_APP_LD"


def ensure_libs() -> None:
    """Put the CUDA + cuDLA shared libs on LD_LIBRARY_PATH, re-exec once if needed."""
    if os.environ.get(_FLAG) == "1":
        return
    import glob
    import site

    roots = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    roots.append(os.path.dirname(os.path.dirname(os.__file__)) + "/site-packages")
    dirs: list[str] = []
    for sp in roots:
        # pip-installed NVIDIA CUDA runtime libs (cuDNN, cuBLAS, ...).
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
        dirs += glob.glob(os.path.join(sp, "nvpl", "lib"))
    # CUDA toolkit libraries. Use the unversioned symlink when available and
    # discover versioned installations so this does not depend on one release.
    dirs += glob.glob("/usr/local/cuda/targets/*/lib")
    dirs += glob.glob("/usr/local/cuda-*/targets/*/lib")
    dirs += [
        "/usr/local/cuda/lib64",
        # system cuDLA libs shipped with JetPack (libcudla.so*).
        "/usr/lib/aarch64-linux-gnu/nvidia",
    ]
    dirs = [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]

    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    missing = [d for d in dirs if d not in parts]

    os.environ[_FLAG] = "1"
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024 * 1024 * 1024))
    if missing:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(missing + parts)
        os.execv(sys.executable, [sys.executable] + sys.argv)
