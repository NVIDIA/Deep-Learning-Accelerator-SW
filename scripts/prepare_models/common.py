#
# SPDX-FileCopyrightText: Copyright (c) 2022 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Common utilities to prepare ONNX models."""
import onnx_graphsurgeon as gs
import numpy as np


def fuse_mul_into_conv(conv_node, mul_node):
    assert len(conv_node.inputs) == 2, 'Not supporting conv with bias for fuse_mul_into_conv'
    conv_weights = conv_node.inputs[1]
    assert len(mul_node.inputs) == 2
    mul_factors = mul_node.inputs[1]
    assert isinstance(mul_factors, gs.Constant)
    assert tuple(mul_factors.shape[:1] + mul_factors.shape[2:4]) == (1, 1, 1)
    mul_factors_data = mul_factors.values.reshape(-1, 1, 1, 1)
    conv_weights.values *= mul_factors_data
    mul_node.inputs.clear()
    conv_node.outputs = [mul_node.outputs[0]]
    mul_node.outputs.clear()


def fuse_add_into_conv(conv_node, add_node):
    assert len(conv_node.inputs) == 2, 'Not supporting conv with bias for fuse_add_into_conv'
    assert len(add_node.inputs) == 2
    add_params = add_node.inputs[1]
    assert isinstance(add_params, gs.Constant)
    assert tuple(add_params.shape[:1] + add_params.shape[2:4]) == (1, 1, 1)
    add_params.values = add_params.values.flatten()
    conv_node.inputs.append(add_params)
    add_node.inputs.clear()
    conv_node.outputs = [add_node.outputs[0]]
    add_node.outputs.clear()


def fuse_into_conv(graph, fusion_func, node_pattern):
    graph.toposort()
    for node in graph.nodes:
        qualifies = len(node.outputs) == 1
        qualifies = qualifies and node.outputs[0] not in graph.outputs
        qualifies = qualifies and len(node.outputs[0].outputs) == 1
        next_node = node.outputs[0].outputs[0] if qualifies else None
        next_node_var_inputs = [inp for inp in next_node.inputs if not isinstance(inp, gs.Constant)] if next_node is not None else list()
        qualifies = qualifies and len(next_node_var_inputs) == 1 and next_node_var_inputs[0] == node.outputs[0]
        if qualifies and (node.op, next_node.op) == node_pattern:
            fusion_func(node, next_node)
            graph = graph.cleanup()


def fuse_convs_horizontally(graph, fuse_conv_outputs, new_names=None):
    def fuse_convs(convs, new_name=None):
        assert len(convs) > 0
        first_conv = convs[0]
        first_weights = first_conv.inputs[1]
        first_bias = first_conv.inputs[2] if len(first_conv.inputs) > 1 else np.zeros(
            first_weights.shape[0], dtype=first_weights.dtype)
        for conv in convs[1:]:
            conv_weights = conv.inputs[1]
            bias = conv.inputs[2] if len(conv.inputs) > 1 else np.zeros(conv_weights.shape[0],
                                                                        dtype=conv_weights.dtype)
            assert conv.attrs == first_conv.attrs
            assert conv_weights.shape[1] == first_weights.shape[1]
            assert tuple(conv_weights.shape[2:4]) == tuple(first_weights.shape[2:4])
            first_weights.values = np.concatenate([first_weights.values, conv_weights.values],
                                                  axis=0)
            first_bias.values = np.concatenate([first_bias.values, bias.values], axis=0)
            if new_name is None:
                first_conv.name += f'+{conv.name}'
                first_conv.outputs[0].name += f'+{conv.outputs[0].name}'
            else:
                first_conv.name = new_name
                first_conv.outputs[0].name = new_name
            graph.outputs.remove(conv.outputs[0])
            conv.inputs.clear()
            conv.outputs.clear()
        graph.cleanup()
    if new_names is not None:
        assert len(new_names) == len(fuse_conv_outputs)
    tensors = graph.tensors()
    for idx, fused_outputs in enumerate(fuse_conv_outputs):
        fused_convs = list()
        for output in fused_outputs:
            tensor = tensors[output]
            assert len(tensor.inputs) == 1
            conv = tensor.inputs[0]
            assert conv.inputs[0] not in graph.outputs
            fused_convs.append(conv)
        new_name = new_names[idx] if new_names is not None else None
        fuse_convs(fused_convs, new_name)


def extract_subgraph(graph, input_names, output_names, input_shape):
    tensors = graph.tensors()
    for tensor in tensors.values():
        if tensor.shape is not None and not isinstance(tensor, gs.Constant) and len(tensor.shape) > 0:
            tensor.shape[0] = input_shape[0]
    graph.inputs = [
        tensors[name].to_variable(dtype=np.float32, shape=input_shape) for name in input_names
    ]
    graph.outputs = [tensors[name].to_variable(dtype=np.float32) for name in output_names]
    graph = graph.cleanup()
    # makes node transformations easier:
    for node in graph.nodes:
        for out in node.outputs:
            out.shape = None

def to_resize_with_scales(graph):
    for node in graph.nodes:
        if node.op == 'Resize' and len(node.inputs) > 3:
            sizes = node.inputs[3].values.astype(np.float32)
            input_shape = np.array(node.inputs[0].shape, dtype=np.float32)
            input_shape[0] = 1
            scales_vals = sizes / input_shape
            scales = gs.Constant(name=f'{node.name}_scales', values=scales_vals)
            node.inputs = node.inputs[:2] + [scales]
