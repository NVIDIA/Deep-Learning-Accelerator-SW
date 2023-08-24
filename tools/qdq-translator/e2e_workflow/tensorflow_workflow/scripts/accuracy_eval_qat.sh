#
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

ROOT_DIR=$(realpath ../../models)
ROOT_DIR_SCRIPTS=$(realpath ./)
MODEL_NAME="resnet_50v1"
DATA_LOADER_NAME="resnet_v1"

# ========== Build engine ==========
echo "Building engine: ${MODEL_NAME}"
trtexec --onnx=${ROOT_DIR}/${MODEL_NAME}.onnx \
        --int8 \
        --saveEngine=${ROOT_DIR}/${MODEL_NAME}.engine \
        --profilingVerbosity=detailed \
        --exportLayerInfo=${ROOT_DIR}/${MODEL_NAME}.engine.graph.json \
        --exportProfile=${ROOT_DIR}/${MODEL_NAME}.engine.profile.json \
        --separateProfileRun \
        --verbose &> ${ROOT_DIR}/${MODEL_NAME}_trtexec.log
wait

# ======= Evaluate engine: accuracy ========
cd $ROOT_DIR_SCRIPTS/tensorflow-quantization
export PYTHONPATH=$PWD:$PYTHONPATH
cd examples
echo "Running inference on ${MODEL_NAME} (model) with ${DATA_LOADER_NAME} (dataloader)"
python infer_engine.py \
       --engine=${ROOT_DIR}/${MODEL_NAME}.engine \
       --model_name=${DATA_LOADER_NAME} \
       --log_file=${MODEL_NAME}_accuracy.log
