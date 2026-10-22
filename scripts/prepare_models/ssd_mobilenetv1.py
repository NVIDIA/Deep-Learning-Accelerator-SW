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
"""ONNX preparation for SSD-MobileNetV1."""
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import os
import common


def main():
    onnx_file_orig = './ssd_mobilenet_v1_coco_2018_01_28.onnx'
    if not os.path.isfile(onnx_file_orig):
        print(f'Could not find {onnx_file_orig}, first download it as detailed in instructions.')
        exit(0)

    graph = gs.import_onnx(onnx.load(onnx_file_orig))
    input_shape = (-1, 3, 300, 300)
    input_names = ['Preprocessor/sub:0']
    output_names = list()
    fuse_conv_outputs = list()
    for idx in range(6):
        fused_nodes = list()
        for appendix in ['/BoxEncodingPredictor/BiasAdd:0', '/ClassPredictor/BiasAdd:0']:
            fused_nodes.append(f'BoxPredictor_{idx}{appendix}')
        output_names.extend(fused_nodes)
        fuse_conv_outputs.append(fused_nodes)
    common.extract_subgraph(graph, input_names, output_names, input_shape)
    onnx_model = gs.export_onnx(graph)
    onnx_model = shape_inference.infer_shapes(onnx_model)

    onnx_file_chopped = onnx_file_orig.replace('.onnx', '_prepared.onnx')
    graph = gs.import_onnx(onnx_model)
    common.fuse_into_conv(graph, common.fuse_mul_into_conv, ('Conv', 'Mul'))
    common.fuse_into_conv(graph, common.fuse_add_into_conv, ('Conv', 'Add'))
    common.fuse_convs_horizontally(graph, fuse_conv_outputs)
    model_opt = gs.export_onnx(graph)
    onnx.save(model_opt, onnx_file_chopped)
    print(f'Saving the resulting model to {onnx_file_chopped}...')


if __name__ == '__main__':
    main()
