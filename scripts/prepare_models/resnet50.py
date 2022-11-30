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
"""ONNX preparation for ResNet-50."""
import os
import onnx
import numpy as np
import onnx_graphsurgeon as gs
from onnx import shape_inference
import common


def simplify_classification_head(graph):
    node_dict = dict()
    for node in graph.nodes:
        node_dict[node.name] = node

    # ReduceMean -> AveragePool:
    node_dict['resnet_model/Mean'].op = 'AveragePool'
    node_dict['resnet_model/Mean'].attrs = {'kernel_shape': [7, 7]}

    # {Squeeze, MatMul, Add} -> 1x1 Conv:
    node_dict['resnet_model/Squeeze'].inputs.clear()
    node_dict['resnet_model/Squeeze'].outputs.clear()
    node_dict['resnet_model/dense/MatMul'].op = 'Conv'
    matmul_weights = node_dict['resnet_model/dense/MatMul'].inputs[1]
    matmul_weights.values = matmul_weights.values.T
    matmul_weights.values = np.expand_dims(matmul_weights.values, [-2, -1])
    bias_weights = node_dict['resnet_model/dense/BiasAdd'].inputs[1]
    node_dict['resnet_model/dense/MatMul'].inputs = [
        node_dict['resnet_model/Mean'].outputs[0], matmul_weights, bias_weights
    ]
    node_dict['resnet_model/dense/MatMul'].outputs = [
        node_dict['resnet_model/dense/BiasAdd'].outputs[0]
    ]
    node_dict['resnet_model/dense/MatMul'].attrs = {'kernel_shape': [1, 1]}
    node_dict['resnet_model/dense/BiasAdd'].inputs.clear()
    node_dict['resnet_model/dense/BiasAdd'].outputs.clear()
    graph.cleanup()


def main():
    onnx_file_orig = './resnet50_v1.onnx'
    if not os.path.isfile(onnx_file_orig):
        print(f'Could not find {onnx_file_orig}, first download it as detailed in instructions.')
        exit(0)
    onnx_file_chopped = onnx_file_orig.replace('.onnx', '_prepared.onnx')
    graph = gs.import_onnx(onnx.load(onnx_file_orig))
    input_shape = (-1, 3, 224, 224)
    input_names = ['input_tensor:0']
    output_names = ['resnet_model/dense/BiasAdd:0']
    common.extract_subgraph(graph, input_names, output_names, input_shape)
    simplify_classification_head(graph)
    print(f'Saving the resulting model to {onnx_file_chopped}...')
    onnx_model = gs.export_onnx(graph)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, onnx_file_chopped)


if __name__ == '__main__':
    main()
