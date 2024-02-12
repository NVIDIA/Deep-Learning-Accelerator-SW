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
"""Parsing the ONNX model and retrieving the scaling factors."""

import logging
from argparse import ArgumentParser
import json
import os
from typing import Dict, Union, Set, List
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon import Variable, Node
import onnxoptimizer
import numpy as np
from utils import export_to_trt_calib

DEFAULT_TRT_VERSION = 8600
WEIGHT_SCALE_TOLERANCE = 1e-3
DEFAULT_OPS_TO_INFER_ADJACENT_SCALES = {
    'MaxPool', 'Flatten', 'Relu', 'Pad', 'Transpose', 'Reshape', 'Squeeze', 'Sigmoid'
}

ARGPARSER = ArgumentParser('Translate an ONNX model with Q/DQ nodes into an ONNX model'
                           ' without Q/DQ nodes plus a JSON file with int8 tensor scales'
                           ' and tensors that need to run at higher precision.'
                           ' This script also generates a calibration cache file and the argument for'
                           ' --layerPrecisions to use with trtexec.')
ARGPARSER.add_argument('--input_onnx_models',
                       '-i',
                       type=str,
                       required=True,
                       nargs='+',
                       help='A space-separated list of input ONNX models with Q/DQ nodes.')
ARGPARSER.add_argument('--output_dir',
                       '-o',
                       type=str,
                       default='./translated',
                       help='The output directory where the ONNX file(s) without'
                       ' Q/DQ nodes and their respective precision configs are placed.'
                       ' Additionally exports this info in a trtexec-friendly format.'
                       ' Default: "./translated"')
ARGPARSER.add_argument('--trt_calib_version',
                       type=str,
                       default=DEFAULT_TRT_VERSION,
                       help=f'The TensorRT version used for exporting a trtexec-compatible calib'
                       f' cache. Default: "{DEFAULT_TRT_VERSION}".')
ARGPARSER.add_argument('--infer_average_pool_scales',
                       action='store_true',
                       help='If set, derive missing input or output scales of (Global)AveragePool and ReduceMean from'
                            ' upstream or downstream tensors (assuming that they do not change). Can help with reducing'
                            ' latency but may also result in lower accuracy compared to running more ops'
                            ' at higher precision.')
ARGPARSER.add_argument('--infer_concat_scales',
                       action='store_true',
                       help='If set, derive missing input/output scales for Concat layer. Since concat is a multi-input'
                            ' layer, the output scale is set to the maximum existing input scale. This may help'
                            ' some models, such as Inception-V3, maintain their accuracy.')
ARGPARSER.add_argument('--infer_mul_scales',
                       action='store_true',
                       help='If set, derive missing scales for Mul operation.'
                            ' Can help with reducing latency but may also result in lower accuracy compared to running more'
                            ' ops at higher precision.')
ARGPARSER.add_argument('--calibration_type',
                       type=str,
                       default='max',
                       help='If the maximum (max) calibration method is employed, it is imperative to conduct a range check on the domain of values for the scale parameter.'
                            'In contrast, if the histogram calibration method is adopted, this particular examination can be deemed unnecessary and therefore skipped.')
ARGPARSER.add_argument('--addtl_ops_to_infer_adjacent_scales',
                       type=str,
                       nargs='+',
                       default=list(),
                       help='A space-separated list of ONNX operators whose scales can be'
                       ' propagated upstream and/or downstream, in addition to the default'
                       f' ops used in this manner: {DEFAULT_OPS_TO_INFER_ADJACENT_SCALES}.')
ARGPARSER.add_argument('--rename_node_outputs',
                       action='store_true',
                       help='Rename each tensor to its producer node\'s name.')
ARGPARSER.add_argument(
    '--add_unary_ew_scales_for_dla',
    action='store_true',
    help=
    'For element-wise nodes consuming from Convs w/o Q/DQ nodes in between, insert unary scales.')
ARGPARSER.add_argument('--verbose', action='store_true', help='Increase verbosity.')

class QATModelParser:
    """
    Parse the QAT ONNX model with the following steps:
        1. Iterate the model to find/save the scaling factor for each activation.
        2. Remove QuantizeLinear and DequantizeLinear nodes.
    """
    _fake_quantize_per_tensor_affine_err_hint = (
        "This is possibly caused by symbolic variables were not converted to tensors during PyTorch "
        "to ONNX exporting. Please ensure that the exporting function follows the official "
        "fake_quantize_per_tensor_affine exporting scheme: "
        "https://github.com/pytorch/pytorch/blob/18dd5cd/torch/onnx/symbolic_opset10.py#L300-L308")
    _fake_quantize_per_channel_affine_err_hint = (
        "This is possibly caused by symbolic variables were converted to tensors, or `scale` variable "
        "was cased to `Float` type during PyTorch to ONNX exporting. Please ensure that "
        "the exporting function follows the official fake_quantize_per_channel_affine exporting "
        "scheme: "
        "https://github.com/pytorch/pytorch/blob/18dd5cd/torch/onnx/symbolic_opset13.py#L132-L146")

    @staticmethod
    def get_quantized_tensor(node: gs.Node, graph: gs.Graph) -> Union[gs.Variable, gs.Constant]:
        """
        Return the input tensor from a quantize node.
          If the quantize node is applied to model weights, this function returns the model weights tensor.
          If the quantize node is applied to activation, this function returns the activation tensor.

        Args:
            node(gs.Node): QuantizeLinear node.
        Returns:
            (gs.Tensor): The tensor quantized by the quantize node.
        """
        def convert_constant_to_variable_node(nodes_to_convert: List[gs.Node]):
            """
            Ensure support for TF-generated ONNX models by converting selected gs.Constant nodes to gs.Variable
              filled by a gs.Constant operator. The proposed fix updates node_input in-memory.

            Error being fixed: "RuntimeError: Expected activation quantizer arguments to be Variables, but got
                (<class 'onnx_graphsurgeon.ir.tensor.Variable'>, <class 'onnx_graphsurgeon.ir.tensor.Constant'>,
                <class 'onnx_graphsurgeon.ir.tensor.Constant'>). This is possibly caused by symbolic variables
                were not parsed to tensors during PyTorch to ONNX exporting. Please ensure that the exporting
                function follows the official fake_quantize_per_tensor_affine exporting scheme.

            Args:
                nodes_to_convert(List[gs.Node]): list of nodes to convert.
            """
            for node_input in nodes_to_convert:
                # Copy Constant into temporary variable
                node_input_copy = gs.Constant(name=node_input.name + "_constant",
                                              values=node_input.values,
                                              data_location=node_input.data_location)
                # Make Constant Node and append to 'graph'
                node_input_copy_node = gs.Node(op="Constant",
                                               attrs={'value': node_input_copy},
                                               inputs=[],
                                               outputs=[node_input_copy])
                graph.nodes.append(node_input_copy_node)
                # Convert original Constant to Variable type with the copied Constant as input
                node_input.to_variable(dtype=node_input.dtype, shape=node_input.shape)
                node_input.inputs.append(node_input_copy_node)

        if not node.op == "QuantizeLinear" or len(node.inputs) != 3:
            raise RuntimeError(f"Expected QuantizeLinear with 3 arguments, but got {node.op} with "
                               f"{len(node.inputs)} arguments.")
        # For weight quantizers: Exported as per-channel QuantLinear operators, `x` and
        # `y_zero_point` are parsed as gs.Constants and `y_scale` is a gs.Variable filled by a
        # gs.Constant operator.
        if type(node.inputs[0]) == gs.Constant:
            if type(node.inputs[1]) == gs.Constant:
                convert_constant_to_variable_node([node.inputs[1]])
            if (not type(node.inputs[1]) == gs.Variable and type(node.inputs[2]) == gs.Constant):
                raise RuntimeError(
                    f"Expected weight quantizer scale and zero_point to be Variable and Constant, "
                    f"resp., but got {tuple(type(node_input) for node_input in node.inputs[1:])}. "
                    f"{QATModelParser._fake_quantize_per_channel_affine_err_hint}.")
            if not (len(node.inputs[1].inputs) == 1 and node.inputs[1].inputs[0].op == "Constant"):
                raise RuntimeError(
                    f"Expected QuantizeLinear operator's scale argument to be parsed as "
                    f"gs.Variable filled by gs.Constant operator, but got "
                    f"{node.inputs[1].inputs[0].op} operator. "
                    f"{QATModelParser._fake_quantize_per_channel_affine_err_hint}.")
            quantize_tensor = node.inputs[0]
        # For activation quantizers: Exported as per-tensor QuantizeLinear operators, `x`, `y_scale`
        # and `y_zero_point` are all parsed to gs.Variables and scale and zero-point are filled by
        # gs.Constant operators.
        else:
            nodes_to_convert = [
                node_input for node_input in node.inputs if type(node_input) == gs.Constant
            ]
            convert_constant_to_variable_node(nodes_to_convert)
            if not all(type(node_input) == gs.Variable for node_input in node.inputs):
                raise RuntimeError(
                    f"Expected activation quantizer arguments to be Variables, but got "
                    f"{tuple(type(node_input) for node_input in node.inputs)}. "
                    f"{QATModelParser._fake_quantize_per_tensor_affine_err_hint}.")
            if not all(
                    len(var.inputs) == 1 and var.inputs[0].op == "Constant"
                    for var in node.inputs[1:]):
                raise RuntimeError(
                    f"Expected QuantizeLinear operator's scale and zero_point arguments to be "
                    f"parsed as gs.Variables filled by gs.Constant operators, but got "
                    f"{tuple(var.inputs[0].op for var in node.inputs[1:])} operators. "
                    f"{QATModelParser._fake_quantize_per_tensor_affine_err_hint}.")
            quantize_tensor = node.inputs[0]
        return quantize_tensor

    @staticmethod
    def verify_weight_scales(tensor_data: np.ndarray, quant_scales: np.ndarray, node_name: str,
                             node_op: str):
        """Verify that the weight scales correspond to how TensorRT performs weight quantization."""
        tensor_data_reshaped = tensor_data.copy()
        if node_op == 'MatMul':
            tensor_data_reshaped = tensor_data_reshaped.T
        K = tensor_data_reshaped.shape[0]
        tensor_data_reshaped = tensor_data_reshaped.reshape(K, -1)
        t_min = tensor_data_reshaped.min(axis=1)
        t_max = tensor_data_reshaped.max(axis=1)
        dyn_range = np.max([np.abs(t_min), np.abs(t_max)], axis=0)
        derived_scales = dyn_range / 127.0
        assert derived_scales.size == quant_scales.size, f'node {node_name} scale sizes did not match: expected_scales.size={derived_scales.size}, quant_scales.size={quant_scales.size}.\n' \
        'Note: Only per-channel/per-filter weight quantization is supported, unless you set `--calibration_type histogram` which skips this step (not recommended).'
        assert np.isclose(derived_scales, quant_scales, atol=WEIGHT_SCALE_TOLERANCE).all(
        ), f'node {node_name} scales do not match: expected_scales={derived_scales},\n quant_scales={quant_scales}'

    @staticmethod
    def node_replace_input(node: gs.Node, name: str, tensor: Union[gs.Variable, gs.Constant],
                           quant_scales: np.ndarray, calibration_type: str):
        """ For a given node, try to replace one of its inputs to the given tensor.

        Args:
            node(gs.Node): The node to replace the input.
            name(str): For all the inputs of node, if it has one that matches with name,
                replace it with given tensor.
            tensor(Union[gs.Variable, gs.Constant]): Used to replace one of the inputs of node.
        """
        for index, node_input in enumerate(node.inputs):
            if node_input.name == name:
                assert node_input.shape == tensor.shape
                assert node_input.dtype == tensor.dtype
                if isinstance(tensor, gs.Constant) and calibration_type == 'max':
                    QATModelParser.verify_weight_scales(tensor.values, quant_scales, node.name,
                                                        node.op)
                node.inputs[index] = tensor

    @staticmethod
    def graph_replace_output(
        graph: gs.Graph,
        name: str,
        tensor: Union[gs.Variable, gs.Constant],
    ):
        """ For a graph, try to replace one of its outputs to the given tensor.

        Args:
            graph(gs.Graph): graph to replace its outputs.
            name(str): For all the outputs of the graph, if it has one that matches with name,
                replace it with given tensor.
            tensor(Union[gs.Variable, gs.Constant]): Used to replace one of the outputs of graph.
        """
        for index, graph_output in enumerate(graph.outputs):
            if graph_output.name == name:
                assert graph_output.shape == tensor.shape
                assert graph_output.dtype == tensor.dtype
                tensor.name = name
                graph.outputs[index] = tensor

    @staticmethod
    def extract_qdq_scales(quantize_node: gs.Node, dequantize_node: gs.Node):
        if dequantize_node.op != "DequantizeLinear":
            raise ValueError("The dequantize node must be DequantizeLinear type.")
        assert len(dequantize_node.outputs) == 1
        if isinstance(quantize_node.inputs[1], gs.Constant):
            quant_scales = quantize_node.inputs[1].values
        else:
            quant_scales = quantize_node.inputs[1].inputs[0].attrs["value"].values
            
        if len(quantize_node.inputs) > 2 and isinstance(quantize_node.inputs[2], gs.Constant):
            quant_zero_points = quantize_node.inputs[2].values
            assert (quant_zero_points == 0).all(), 'zero_points for '
        if isinstance(dequantize_node.inputs[1], gs.Constant):
            dequant_scales = dequantize_node.inputs[1].values
        else:
            dequant_scales = dequantize_node.inputs[1].inputs[0].attrs["value"].values
        if len(dequantize_node.inputs) > 2 and isinstance(quantize_node.inputs[2], gs.Constant):
            dequant_zero_points = dequantize_node.inputs[2].values
            assert (dequant_zero_points == 0).all()
        assert (quant_scales == dequant_scales).all()
        return quant_scales

    @staticmethod
    def extract_precision_config(graph: gs.Graph, calibration_type: str):
        precision_config = {}
        # Check for all zero weighted inputs of QuantizeLinear and
        # Conv nodes and add to this set to skip for the later check
        zero_check_skip = set()
        for node in graph.nodes:
            if node.op != "QuantizeLinear":
                if node.op in ("Conv", "ConvTranspose", "Gemm"):
                    for i in node.inputs:
                        ti = i.copy()
                        if isinstance(i, gs.ir.tensor.Constant) and not ti.values.any():
                            zero_check_skip.add(i.name)
                continue
            for i in node.inputs:
                # Make a shallow copy of input due to a bug in ONNX GS while calling i.values
                ti = i.copy()
                if i.name.endswith("weight") and not ti.values.any():
                    zero_check_skip.add(i.name)
            quantize_node = node
            # Ensure support for TF-generated ONNX models
            quantize_tensor = QATModelParser.get_quantized_tensor(quantize_node, graph)
            # Only quantized activation has input node.
            is_activation_quantizer = len(quantize_tensor.inputs) > 0
            is_input_quantizer = len(quantize_tensor.inputs) == 0 and quantize_tensor.name in [
                i.name for i in graph.inputs
            ]
            if is_activation_quantizer or is_input_quantizer:
                # This assumes the quantization for activation is per-tensor quantization.
                # Note, TensorRT_Optimization tools requires the key needs to be the name
                # of the quantized tensor.
                config_input = quantize_node.inputs[1].inputs[0].attrs["value"].values
                # Scales need to be multiplied by the quant_max value for activation
                # quantizer
                precision_config[quantize_tensor.name] = float(config_input)
            # The dequantize node is followed by the quantize node.
            dequantize_node = node.o()
            quant_scales = QATModelParser.extract_qdq_scales(quantize_node, dequantize_node)
            dequantize_output = dequantize_node.outputs[0]
            if(np.isin(dequantize_output, graph.outputs)):
                precision_config[dequantize_output.name] = float(quant_scales)
            # Find all nodes whose inputs has dequantize_output.
            # It seems to be a bug, dequantize_node.outputs[0].outputs only return one nodes.
            for node in graph.nodes:
                QATModelParser.node_replace_input(node, dequantize_output.name, quantize_tensor,
                                                  quant_scales, calibration_type)
            # If The DequantizeLinear is a termination node, we also need to replace one of its
            # outputs to quantize_tensor
            QATModelParser.graph_replace_output(graph, dequantize_output.name, quantize_tensor)
            # Remove quantize node and dequantize node
            quantize_node.outputs.clear()
            dequantize_node.outputs.clear()
        # Check if no conv nodes other than the initial
        # zero-weighted conv nodes have all zero weights
        for node in graph.nodes:
            if node.op in ("Conv", "ConvTranspose", "Gemm"):
                for i in node.inputs:
                    ti = i.copy()
                    if (isinstance(i, gs.ir.tensor.Constant) and i.name not in zero_check_skip):
                        assert ti.values.any()
        graph.cleanup()
        graph.toposort()
        return precision_config
    
    @staticmethod 
    def find_with_input_node(graph, name):
        for node in graph.nodes:
            if len(node.inputs) > 0 and name in node.inputs:
                return node

    @staticmethod        
    def infer_mul_scales(graph, node, precision_config):
        out_name = node.outputs[0].name
        input_scales = [None, None]
        output_scale = None
        if out_name in precision_config.keys():
            output_scale = precision_config[out_name]
        if output_scale is None:
            return 
        
        for ind, inp in enumerate(node.inputs):
            if inp.name in precision_config.keys():
               input_scales[ind] = precision_config[inp.name]
               
        if input_scales[0] is not None and input_scales[1] is not None: 
            return   
        
        # if 2 of the 3 I/O scales (2 input scales, 1 output scale) are already known, the missing scale can be inferred through deduction.
        if input_scales[0] is not None:
            precision_config[node.inputs[1].name] = (output_scale / input_scales[0]) / 127.
        elif input_scales[1] is not None:
             precision_config[node.inputs[0].name] = (output_scale / input_scales[1]) / 127.           
    
    @staticmethod
    def infer_unchanged_scales(
        graph: gs.Graph,
        precision_config: Dict[str, float],
        downstream: bool,
        ops_to_infer_adjacent_scales: Set[str],
        zero_check_skip: Set[str]
    ):
        graph.toposort()
        node_list = graph.nodes
        if not downstream:
            node_list = node_list[::-1]
        for node in node_list:
            if node.op in ops_to_infer_adjacent_scales:
                in_name = node.inputs[0].name
                out_name = node.outputs[0].name
                if downstream:
                    if node.op == "Concat":
                        max_input_scale = -np.inf
                        for concat_inp in node.inputs:
                            if concat_inp.name in precision_config.keys() and precision_config[concat_inp.name] >= max_input_scale:
                                max_input_scale = precision_config[concat_inp.name]
                        if max_input_scale != -np.inf:
                            precision_config[out_name] = max_input_scale
                    elif node.op == 'Sigmoid':
                        if out_name not in precision_config.keys() and 'Mul' in ops_to_infer_adjacent_scales and QATModelParser.find_with_input_node(graph,  node.outputs[0]).op == 'Mul':
                                zero_check_skip.add(out_name)
                                precision_config[out_name] = 1/127.            
                    else:
                        if in_name in precision_config.keys(
                        ) and out_name not in precision_config.keys():
                            precision_config[out_name] = precision_config[in_name]
                elif out_name in precision_config.keys() and in_name not in precision_config.keys(
                ):
                    if node.op == "Sigmoid":  # update Sigmoid scales during backward pass.
                        if out_name in zero_check_skip:
                            continue
                        
                        y = precision_config[out_name] * 127.
                        try:
                            precision_config[in_name] = np.log(y / (1 - y)) / 127.
                        except ZeroDivisionError:
                            print("Illegal Division by 0 detected!")
                            raise
                    elif node.op == "Concat":
                        for concat_inp in node.inputs:
                            if concat_inp.name not in precision_config.keys():
                                precision_config[concat_inp.name] = precision_config[out_name]
                    elif node.op == "Mul":
                        QATModelParser.infer_mul_scales(graph, node, precision_config)
                    else:
                        precision_config[in_name] = precision_config[out_name]

    @staticmethod
    def prepare_for_bn_fusion(graph: gs.Graph, rename_node_outputs: bool):
        for node in graph.nodes:
            # conv_bn_fusion in onnx_optimizer only works if there's a Conv bias
            if node.op in {'Conv'} and len(node.inputs) == 2:
                weight = node.inputs[1]
                output_channels = weight.shape[0]
                bias = gs.Constant(f'{node.name}_bias',
                                   np.zeros(output_channels, dtype=weight.dtype))
                node.inputs.append(bias)
            # we want to preserve the bn's output tensor name (which most likely has the scaling factor)
            if node.op in {'BatchNormalization'}:
                orig_in_name = node.inputs[0].name
                orig_out_name = node.outputs[0].name
                node.inputs[0].name = orig_out_name
                node.outputs[0].name = orig_in_name
                if rename_node_outputs and len(node.inputs[0].inputs) == 1:
                    node.inputs[0].inputs[0].name = orig_out_name

    @staticmethod
    def fold_reshape_transpose_into_conv(graph: gs.Graph):
        """
        Delete Reshape->Transpose between DQ and Conv layer:
          Original: QL -> DQL (weight) -> Reshape -> Transpose -> Conv
          Processed: (manually transposed QL) -> (manually transposed DQL) -> Conv
        This happens in ConvTranspose and grouped Convolutions (where group > 1). This optimization is needed
           in order to successfully build a TensorRT engine.
        """
        def check_descendants(
                graph: gs.Graph,
                node: Node,
                pattern: List = ["DequantizeLinear", "Reshape", "Transpose", "Conv"]) -> bool:
            """ Check if node's descendants follow a specific 'pattern'.

            Args:
                node (Node): initial node.
                pattern (List): list containing node's descendants ([child, grandchild, great-grandchild, ...]).
            Returns:
                bool: indicating whether node's descendants follow the given pattern.
                node_out[0]: first child node.
                node_out[-2]: second-to-last child node (input to the last layer in 'pattern').
            """
            node_out = [node.o()]
            for i, p in enumerate(pattern):
                if node_out[i].op == p and not np.isin(node_out[i].outputs[0], graph.outputs):
                    node_out.append(node_out[i].o())
                else:
                    return False, node_out[0], None
            return True, node_out[0], node_out[-2]

        def _transpose_gs_variable(variable_node, new_shape, new_dtype):
            quant_var_new_shape = np.asarray(new_shape)
            quant_var_new_shape = quant_var_new_shape.astype(np.int64).tolist()
            quant_new_var_output = Variable(name=variable_node.name,
                                            dtype=new_dtype,
                                            shape=quant_var_new_shape)
            quant_new_var_output.inputs = variable_node.inputs
            quant_new_var_output.outputs = variable_node.outputs
            variable_node.inputs.clear()
            variable_node.outputs.clear()

            # Bring the 4D Variable Matrix back to index 0.
            # Otherwise, the following error is given:
            #   [E] [TRT] ModelImporter.cpp:726: ERROR: builtin_op_importers.cpp:1039 In function QuantDequantLinearHelper:
            #   [6] Assertion failed: scaleAllPositive && "Scale coefficients must all be positive"
            quant_new_var_output.outputs[0].inputs.insert(
                0, quant_new_var_output.outputs[0].inputs[-1])
            quant_new_var_output.outputs[0].inputs.pop(-1)

            return quant_new_var_output

        # 1. Find all the QuantLinear nodes
        quant_nodes = [node for node in graph.nodes if node.op == "QuantizeLinear"]

        # 2. Remove Reshape->Transpose layers between DQ and Conv layers
        pattern = ["DequantizeLinear", "Reshape", "Transpose", "Conv"]
        for (i, node) in enumerate(quant_nodes):
            has_pattern, node_out, node_conv_input = check_descendants(graph, node, pattern)

            if has_pattern:
                # A. Transpose QuantizeLinear weights and output variable (3x3x960x1 -> 960x1x3x3)
                quant_weights_tensor = node.inputs[0]
                quant_weights_transposed = np.transpose(quant_weights_tensor.values, [2, 3, 0, 1])
                node.inputs[0].values = quant_weights_transposed

                quant_var_output = node.outputs[0]
                node.outputs[0] = _transpose_gs_variable(
                    quant_var_output,
                    new_shape=quant_weights_tensor.shape,
                    new_dtype=np.int8  # The output of QuantLinear should be INT8
                )

                # B. Transpose DequantizeLinear, with output precision = QuantLinear input's type (np.fp32)
                dequant_var_output = node_out.outputs[0]
                node_out.outputs[0] = _transpose_gs_variable(dequant_var_output,
                                                             new_shape=quant_weights_tensor.shape,
                                                             new_dtype=quant_weights_tensor.dtype)

                # C. Connect the output of DQLinear to Conv
                # Note: input at index 0 is from the input quantization
                node_conv_input.inputs[1] = node_out.outputs[0]

        # 3. Remove unused nodes, and topologically sort the graph.
        graph.cleanup().toposort()
        new_model = gs.export_onnx(graph)
        graph = gs.import_onnx(new_model)
        return graph

    @staticmethod
    def fuse_with_conv_through_unary_scales(node: gs.Node, tensor: gs.Variable,
                                            precision_config: Dict[str, float],
                                            unary_scales_tensors: Set[str]):
        can_fuse = False
        is_conv_ew_fusion = node.op == 'Conv'
        single_downstream_node = tensor.outputs[0] if len(tensor.outputs) == 1 else None
        is_conv_ew_fusion = is_conv_ew_fusion and single_downstream_node is not None
        is_conv_ew_fusion = is_conv_ew_fusion and single_downstream_node.op in {'Add'}
        is_conv_ew_fusion = is_conv_ew_fusion and len(single_downstream_node.inputs) == 2
        other_input = single_downstream_node.inputs[1] if is_conv_ew_fusion else None
        is_conv_ew_fusion = is_conv_ew_fusion and other_input.name not in unary_scales_tensors
        if is_conv_ew_fusion:
            unary_scales_tensors.add(tensor.name)
            precision_config[tensor.name] = 1.0
            logging.info(
                f'No tensor scales for {node.name}\'s output tensor {tensor.name} but assuming {node.op} + {single_downstream_node.op} fusion'
            )
            can_fuse = True
        return can_fuse

    @staticmethod
    def parse(model_path: str, output_dir: str, post_opt_passes: List[str],
              ops_to_infer_adjacent_scales: Set[str], trt_calib_version: str,
              rename_node_outputs: bool, add_unary_ew_scales_for_dla: bool,
              calibration_type: str):
        """ Process the ONNX model.

        Args:
            model_path(str): Path to the ONNX model.
            output_dir(str): Output folder for saving the results.
        """
        model = onnx.load(model_path)

        model = onnx.shape_inference.infer_shapes(model)
        graph = gs.import_onnx(model)
        if rename_node_outputs:
            for node in graph.nodes:
                for idx, out in enumerate(node.outputs):
                    name = node.name
                    if idx > 0:
                        name = f'{name}_{idx}'
                    out.name = name
        # Pre-process model to remove Reshape+Transpose layers after weight DQ and before Conv layer
        logging.debug(f'Calling fold_reshape_transpose_into_conv()...')
        graph = QATModelParser.fold_reshape_transpose_into_conv(graph)

        # Extract precision
        logging.debug(f'Calling extract_precision_config()...')
        precision_config = QATModelParser.extract_precision_config(graph, calibration_type)
        
        # forward pass
        zero_check_skip = set()
        logging.debug(f'Calling infer_unchanged_scales() with downstream=True...')
        QATModelParser.infer_unchanged_scales(graph, precision_config, True,
                                              ops_to_infer_adjacent_scales, zero_check_skip)
        # backward pass
        logging.debug(f'Calling infer_unchanged_scales() with downstream=False...')
        QATModelParser.infer_unchanged_scales(graph, precision_config, False,
                                              ops_to_infer_adjacent_scales, zero_check_skip)
     
        if 'fuse_bn_into_conv' in post_opt_passes:
            logging.debug(f'Calling fuse_bn_into_conv()...')
            QATModelParser.prepare_for_bn_fusion(graph, rename_node_outputs)

        # Export translated model and scales
        new_model = gs.export_onnx(graph)
        if len(post_opt_passes) > 0:
            logging.debug(f'Calling onnxoptimizer.optimize()...')
            new_model = onnxoptimizer.optimize(new_model, passes=post_opt_passes)
        logging.debug(f'Calling onnx.checker.check_model()...')
        onnx.checker.check_model(new_model)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_onnx_path = os.path.join(output_dir, f"{model_name}_noqdq.onnx")
        logging.debug(f'Saving the new model as {output_onnx_path}...')
        onnx.save(new_model, output_onnx_path)
        new_graph = gs.import_onnx(new_model)
        out_tensors = [(None, input) for input in new_graph.inputs]
        out_tensors += [(node, node.outputs[0]) for node in new_graph.nodes]
        fp16_nodes = list()
        unary_scales_tensors = set()
        for node, tensor in out_tensors:
            if not isinstance(tensor, gs.Constant) and tensor.name not in precision_config.keys():
                can_fuse_with_conv = add_unary_ew_scales_for_dla and QATModelParser.fuse_with_conv_through_unary_scales(
                    node, tensor, precision_config, unary_scales_tensors)
                if not can_fuse_with_conv:
                    consumers = [node.name for node in tensor.outputs]
                    fp16_nodes.extend(consumers)
                    qualifier = 'input' if node is None else f'{node.name}\'s output'
                    addition = f', recommended to set its consumer nodes {consumers} to fp16' if len(
                        consumers) > 0 else ''
                    logging.info(
                        f'No tensor scales for {qualifier} tensor {tensor.name}{addition}')
        precision_config = {'int8_tensor_scales': precision_config, 'fp16_nodes': fp16_nodes}
        output_json_path = os.path.join(output_dir, f'{model_name}_precision_config.json')
        logging.debug(f'Saving the extracted precision config as {output_json_path}...')
        with open(output_json_path, 'w') as f:
            json.dump(precision_config, f, indent=4)
        export_to_trt_calib(output_json_path, trt_calib_version)


def main(args):
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    # For all passes: https://github.com/onnx/optimizer/tree/master/onnxoptimizer/passes
    opt_passes = [
        'extract_constant_to_initializer', 'fuse_bn_into_conv', 'fuse_pad_into_conv',
        'fuse_pad_into_pool'
    ]
    ops_to_infer_adjacent_scales = DEFAULT_OPS_TO_INFER_ADJACENT_SCALES
    for op in args.addtl_ops_to_infer_adjacent_scales:
        ops_to_infer_adjacent_scales.add(op)
    if args.infer_average_pool_scales:
        ops_to_infer_adjacent_scales.add('AveragePool')
        ops_to_infer_adjacent_scales.add('GlobalAveragePool')
        ops_to_infer_adjacent_scales.add('ReduceMean')
    if args.infer_concat_scales:
        ops_to_infer_adjacent_scales.add("Concat")
    if args.infer_mul_scales:
        ops_to_infer_adjacent_scales.add('Mul')
    os.makedirs(args.output_dir, exist_ok=True)
    parser = QATModelParser()
    for onnx_model in args.input_onnx_models:
        logging.info(f'Parsing {onnx_model}...')
        parser.parse(onnx_model,
                     args.output_dir,
                     opt_passes,
                     ops_to_infer_adjacent_scales,
                     args.trt_calib_version,
                     rename_node_outputs=args.rename_node_outputs,
                     add_unary_ew_scales_for_dla=args.add_unary_ew_scales_for_dla, calibration_type = args.calibration_type)


if __name__ == '__main__':
    main(ARGPARSER.parse_args())
