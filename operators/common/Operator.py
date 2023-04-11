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
"""Generic Operator class for op reconstructions."""

from logging import info

from onnx import shape_inference
import onnx
import onnx_graphsurgeon as gs
import os
import json
from collections import OrderedDict
from common.onnxruntime_utils import onnx_inference

import numpy as np

class Operator():
    def __init__(self):
        self.node_count = 0
        self.tensor_count = 0
        self.op = type(self).__name__

    def generate(self, input_shapes, **kwargs):
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        dtype = kwargs.get('dtype', np.float32)
        inputs = [
            gs.Variable(name=self.new_tensor_name(), dtype=dtype, shape=shape)
            for shape in input_shapes
        ]
        outputs = [gs.Variable(name=self.new_tensor_name(), dtype=dtype)]
        node = gs.Node(op=self.op, inputs=inputs, outputs=outputs, name=self.new_node_name())
        if 'attrs' in kwargs:
            node.attrs = kwargs['attrs']
        nonconst_inputs = [x for x in node.inputs if not isinstance(x, gs.Constant)]
        graph = gs.Graph(nodes=[node], inputs=nonconst_inputs, outputs=node.outputs)
        graph.name = f'{self.op}_orig'
        return graph

    @classmethod
    def op_to_reconstruct(cls):
        return cls.__name__

    @classmethod
    def qualifies_for_reconstruction(cls, node):
        return node.op == cls.op_to_reconstruct()

    @classmethod
    def reconstruct(cls, node, graph, **kwargs):
        if cls.qualifies_for_reconstruction():
            # specialize here
            pass

    def test(self, input_shapes, **kwargs):
        orig_graph = self.generate(input_shapes, **kwargs)
        orig_path = Operator.save_gs_graph(orig_graph, run_shape_inference=True)
        reconstructed_graph = gs.import_onnx(onnx.load(orig_path))
        for node in reconstructed_graph.nodes:
            self.reconstruct(node, reconstructed_graph)
        reconstructed_graph.name = orig_graph.name.replace('_orig', '_reconstructed')
        reconstructed_graph = Operator.save_gs_graph(reconstructed_graph)
        maxabsdiff = self.run_comparison([orig_path, reconstructed_graph], input_shapes)
        return maxabsdiff

    def new_node_name(self):
        result = f'{self.op}_{self.node_count}'
        self.node_count += 1
        return result

    def new_tensor_name(self):
        result = f'tensor_{self.tensor_count}'
        self.tensor_count += 1
        return result

    @staticmethod
    def save_gs_graph(graph,
                      top_dir='./models',
                      run_shape_inference=False,
                      opset=13,
                      producer='onnx_graphsurgeon'):
        graph.opset = opset
        graph.producer = producer
        if run_shape_inference:
            for node in graph.nodes:
                for out in node.outputs:
                    out.shape = None
        model = gs.export_onnx(graph)
        if run_shape_inference:
            model = shape_inference.infer_shapes(model)
        os.makedirs(top_dir, exist_ok=True)
        nonconst_inputs = [x for x in graph.inputs if not isinstance(x, gs.Constant)]
        input_shape = 'x'.join([str(x) for x in nonconst_inputs[0].shape])
        path_name = os.path.join(top_dir, f'{graph.name}_{input_shape}.onnx')
        info(f'Saving file {path_name}...')
        onnx.save(model, path_name)
        return path_name

    def verify_reconstruction(self, onnx_file):
        graph = gs.import_onnx(onnx.load(onnx_file))
        for node in graph.nodes:
            assert node.op != self.op, f'Node "{node.name}" of op type "{node.op}" has not been reconstructed'

    @staticmethod
    def save_scales(json_path, key_to_range):
        ranges_dict = OrderedDict()
        offset = 0
        for key, (low, high) in key_to_range.items():
            scale = max(abs(low), abs(high)) / 127.0
            ranges_dict[key] = dict(scale=scale, min=float(low), max=float(high), offset=offset)
        with open(json_path, 'w') as outfile:
            outfile.write(json.dumps(ranges_dict, indent=4))

    def save_inputs_and_ref_outputs(cls, reconstructed_path, inputs, output, dtypes=[np.float16, np.int8]):
        DTYPE_STRINGS = {np.float16: 'fp16', np.int8: 'int8'}
        def preprocess(data, dtype):
            data = data.copy()
            if dtype == np.int8 and data.dtype != np.bool:
                dyn_range = max(np.abs(data.max()), np.abs(data.min()))
                scaling_factor = dyn_range / 127.0
                data /= scaling_factor
                data = np.round(data).clip(-127, 127)
            data = data.astype(dtype)
            return data
        results_dir = reconstructed_path.replace('.onnx', '')
        graph = gs.import_onnx(onnx.load(reconstructed_path))
        nonconst_inputs = [x for x in graph.inputs if not isinstance(x, gs.Constant)]
        assert len(nonconst_inputs) == len(inputs)
        os.makedirs(results_dir, exist_ok=True)
        key_to_range = dict()
        for dtype in dtypes:
            for idx, (inp_data, inp_tensor) in enumerate(zip(inputs, nonconst_inputs)):
                np.save(os.path.join(results_dir, f'inputs_{idx}_{DTYPE_STRINGS[dtype]}.npy'),
                        preprocess(inp_data, dtype))
                if dtype == np.int8:
                    key_to_range[inp_tensor.name] = (inp_data.min(), inp_data.max())
            np.save(os.path.join(results_dir, f'outputs_0_{DTYPE_STRINGS[dtype]}.npy'),
                    preprocess(output, dtype))
            if dtype == np.int8:
                key_to_range[graph.outputs[0].inputs[0].name] = (output.min(), output.max())
        Operator.save_scales(os.path.join(results_dir, 'int8.json'), key_to_range)

    def run_comparison(self,
                       onnx_paths,
                       input_shapes=None,
                       incremental_vals=False,
                       input_data=None,
                       index_reconstructed=-1,
                       session_options=None):
        assert len(onnx_paths) == 2
        self.verify_reconstruction(onnx_paths[index_reconstructed])
        if input_data is None:
            assert input_shapes is not None
            if incremental_vals:
                input_data = [
                    np.arange(np.product(shape), dtype=np.float32).reshape(shape)
                    for shape in input_shapes
                ]
            else:
                input_data = [np.random.rand(*shape).astype(np.float32) for shape in input_shapes]
        else:
            assert not incremental_vals
            assert input_shapes is None
        session_outputs = list()
        for onnx_file in onnx_paths:
            output_data = onnx_inference(onnx_file, input_data, session_options)
            session_outputs.append(output_data[0])
        maxabsdiff = np.abs(session_outputs[0].flatten().astype(np.float32) -
                            session_outputs[1].flatten().astype(np.float32)).max()
        info(f'Max absdiff between {onnx_paths[0]} and {onnx_paths[1]}: {maxabsdiff}')
        self.save_inputs_and_ref_outputs(onnx_paths[1], input_data, session_outputs[0])
        return maxabsdiff
