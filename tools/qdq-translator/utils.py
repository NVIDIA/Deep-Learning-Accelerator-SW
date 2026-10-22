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

import json
import struct


def float_to_hex(f):
    hex_val = hex(struct.unpack('<I', struct.pack('<f', f))[0])
    hex_val = hex_val[2:]
    return hex_val


def export_to_trt_calib(filename, trt_version):
    # Load precision config file
    with open(filename, "r") as f:
        json_dict = json.load(f)

    # Create new files
    with open(filename.replace(".json", "_calib.cache"),
              "w") as f_calib, open(filename.replace(".json", "_layer_arg.txt"),
                                    "w") as f_layer_precision_arg:

        f_calib.write(f"TRT-{trt_version}-EntropyCalibration2\n")
        int8_tensor_scales = json_dict["int8_tensor_scales"]
        for layer_name, scale in int8_tensor_scales.items():
            # Convert INT8 ranges to scales to HEX
            scale_hex = float_to_hex(scale)
            f_calib.write(f"{layer_name}: {scale_hex}\n")
        fp16_nodes = json_dict["fp16_nodes"]
        # Save list of all layers that need to run in FP16 for later use with TensorRT
        f_layer_precision_list = [f"{x}:fp16" for x in fp16_nodes]
        f_layer_precision_arg.write(",".join(f_layer_precision_list))
