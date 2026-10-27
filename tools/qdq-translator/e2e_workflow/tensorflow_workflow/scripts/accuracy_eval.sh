#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ROOT_DIR=$(realpath ../../translated)
ROOT_DIR_SCRIPTS=$(realpath ./)
MODEL_NAME="resnet_50v1"
DATA_LOADER_NAME="resnet_v1"

# ========== Build engine ==========
echo "Building engine: ${MODEL_NAME}"
CALIB_FILE="${ROOT_DIR}/${MODEL_NAME}_precision_config_calib.cache"
LAYER_PRECISION_FILE="${ROOT_DIR}/${MODEL_NAME}_precision_config_layer_arg.txt"
trtexec --onnx=${ROOT_DIR}/${MODEL_NAME}_noqdq.onnx \
        --calib=${CALIB_FILE} \
        --best \
        --saveEngine=${ROOT_DIR}/${MODEL_NAME}_noqdq.engine \
        --precisionConstraints=obey \
        --layerPrecisions=$(cat $LAYER_PRECISION_FILE) \
        --profilingVerbosity=detailed \
        --exportLayerInfo=${ROOT_DIR}/${MODEL_NAME}_noqdq.engine.graph.json \
        --exportProfile=${ROOT_DIR}/${MODEL_NAME}_noqdq.engine.profile.json \
        --separateProfileRun \
        --verbose &> ${ROOT_DIR}/${MODEL_NAME}_noqdq_trtexec.log
wait

# ======= Evaluate engine: accuracy ========
cd $ROOT_DIR_SCRIPTS/tensorflow-quantization
export PYTHONPATH=$PWD:$PYTHONPATH
cd examples
echo "Running inference on ${MODEL_NAME} (model) with ${DATA_LOADER_NAME} (dataloader)"
python infer_engine.py \
       --engine=${ROOT_DIR}/${MODEL_NAME}_noqdq.engine \
       --model_name=${DATA_LOADER_NAME} \
      --log_file=${MODEL_NAME}_noqdq_accuracy.log
