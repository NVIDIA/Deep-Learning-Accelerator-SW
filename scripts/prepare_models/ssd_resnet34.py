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
"""ONNX preparation for SSD-ResNet-34."""
import os
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import common


def main():
    onnx_file_orig = './resnet34-ssd1200.onnx'
    if not os.path.isfile(onnx_file_orig):
        print(f'Could not find {onnx_file_orig}, first download it as detailed in instructions.')
        exit(0)
    onnx_file_chopped = onnx_file_orig.replace('.onnx', '_prepared.onnx')
    graph = gs.import_onnx(onnx.load(onnx_file_orig))
    input_shape = (-1, 3, 1200, 1200)
    input_names = ['image']
    fuse_conv_outputs = [('Conv_338', 'Conv_349'), ('Conv_360', 'Conv_371'),
                         ('Conv_382', 'Conv_393'), ('Conv_404', 'Conv_415'),
                         ('Conv_426', 'Conv_437'), ('Conv_448', 'Conv_459')]
    output_names = list()
    for fused_pair in fuse_conv_outputs:
        output_names.extend(fused_pair)
    common.extract_subgraph(graph, input_names, output_names, input_shape)
    common.fuse_convs_horizontally(graph, fuse_conv_outputs)
    print(f'Saving the resulting model to {onnx_file_chopped}...')
    onnx_model = gs.export_onnx(graph)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, onnx_file_chopped)


if __name__ == '__main__':
    main()
