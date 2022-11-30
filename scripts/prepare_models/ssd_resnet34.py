#
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
