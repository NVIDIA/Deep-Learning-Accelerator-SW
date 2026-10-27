#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""ONNX preparation for RetinaNet-ResNeXt-50."""
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference
import os
import common


def main():
    onnx_file_orig = './resnext50_32x4d_fpn_sanitized.onnx'
    if not os.path.isfile(onnx_file_orig):
        print(
            f'Could not find {onnx_file_orig}, first download it and sanitize it as detailed in instructions.'
        )
        exit(0)

    graph = gs.import_onnx(onnx.load(onnx_file_orig))
    input_shape = (-1, 3, 800, 800)
    input_names = ['input']
    output_names = [
        'onnx::Shape_1640', 'onnx::Shape_1597', 'onnx::Shape_1554', 'onnx::Shape_1511',
        'onnx::Shape_1468', 'onnx::Shape_1856', 'onnx::Shape_1813', 'onnx::Shape_1770',
        'onnx::Shape_1727', 'onnx::Shape_1684'
    ]
    common.extract_subgraph(graph, input_names, output_names, input_shape)
    onnx_model = gs.export_onnx(graph)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx_file_chopped = onnx_file_orig.replace('.onnx', '_prepared.onnx')
    graph = gs.import_onnx(onnx_model)
    common.to_resize_with_scales(graph)
    common.fuse_into_conv(graph, common.fuse_mul_into_conv, ('Conv', 'Mul'))
    common.fuse_into_conv(graph, common.fuse_add_into_conv, ('Conv', 'Add'))
    model_opt = gs.export_onnx(graph)
    onnx.save(model_opt, onnx_file_chopped)
    print(f'Saving the resulting model to {onnx_file_chopped}...')


if __name__ == '__main__':
    main()
