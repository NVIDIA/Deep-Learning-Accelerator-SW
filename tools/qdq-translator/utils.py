#
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
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
