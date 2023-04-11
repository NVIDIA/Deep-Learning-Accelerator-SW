#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
