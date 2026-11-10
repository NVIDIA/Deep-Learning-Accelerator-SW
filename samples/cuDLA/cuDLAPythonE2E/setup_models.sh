#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run on the target Jetson after installing the README setup dependencies.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-./venv/bin/python}"
TRTEXEC="${TRTEXEC:-${TRT_DIR:+${TRT_DIR}/bin/}trtexec}"

command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON (see README Setup)" >&2; exit 1; }
command -v "$TRTEXEC" >/dev/null || { echo "trtexec not found; set TRTEXEC or TRT_DIR to the target TensorRT installation." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required to download the model." >&2; exit 1; }
"$PYTHON" -c 'import onnx, onnx_graphsurgeon, numpy'

mkdir -p models
model=models/resnet50-v1-12.onnx
if [[ ! -s "$model" ]]; then
    # Use the media endpoint to fetch the ONNX bytes, not a Git LFS pointer.
    trap 'rm -f "$model.part"' EXIT
    curl --fail --location --retry 3 --output "$model.part" \
        https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx
    mv "$model.part" "$model"
fi

"$PYTHON" model_prep/prepare_resnet50_dla.py \
    --fp16 \
    --input "$model" \
    --output models/resnet50_v1_prepared.onnx \
    --stem models/resnet50_stem.onnx \
    --head models/resnet50_head.onnx

"$TRTEXEC" \
    --useDLACore=0 \
    --memPoolSize=dlaSRAM:1 \
    --inputIOFormats=chw16 \
    --outputIOFormats=chw16 \
    --onnx=models/resnet50_v1_prepared.onnx \
    --saveEngine=models/resnet50_v1_prepared.loadable \
    --buildDLAStandalone

echo "Models ready. Run: $PYTHON run.py"
