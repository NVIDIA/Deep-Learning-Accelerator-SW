#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""ONNX preparation for ResNet-50 translated by QDQ Translator."""
import os
import onnx
import numpy as np
import onnx_graphsurgeon as gs
from onnx import shape_inference
import common


def expand_first_conv_pads(graph, first_conv_name):
    node_dict = dict()
    for node in graph.nodes:
        node_dict[node.name] = node
    # add padding
    node_dict[first_conv_name].attrs['pads'] = (3, 3, 3, 3)


def main():
    onnx_file_orig = 'tools/qdq-translator/translated/resnet_50v1_noqdq.onnx'
    if not os.path.isfile(onnx_file_orig):
        print(f'Could not find {onnx_file_orig}, first prepare it as detailed in instructions.')
        exit(0)
    onnx_file_chopped = onnx_file_orig.replace('.onnx', '_prepared.onnx')
    graph = gs.import_onnx(onnx.load(onnx_file_orig))
    input_shape = (-1, 3, 224, 224)
    input_names = ['StatefulPartitionedCall/resnet50/quant_conv1_conv/BiasAdd__511:0']
    output_names = ['Identity:0']
    common.extract_subgraph(graph, input_names, output_names, input_shape)
    expand_first_conv_pads(
        graph, first_conv_name='StatefulPartitionedCall/resnet50/quant_conv1_conv/BiasAdd')
    common.simplify_classification_head(
        graph,
        mean_name='StatefulPartitionedCall/resnet50/quant_avg_pool/Mean',
        squeeze_name='StatefulPartitionedCall/resnet50/quant_avg_pool/Mean_Squeeze__1117',
        matmul_name='StatefulPartitionedCall/resnet50/quant_predictions/MatMul',
        add_name='StatefulPartitionedCall/resnet50/quant_predictions/BiasAdd'
        , softmax_name='StatefulPartitionedCall/resnet50/quant_predictions/Softmax'
    )
    print(f'Saving the resulting model to {onnx_file_chopped}...')
    onnx_model = gs.export_onnx(graph)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, onnx_file_chopped)


if __name__ == '__main__':
    main()
