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
"""Reconstruction of a Where op."""

from logging import info
import onnx
import numpy as np
import onnx_graphsurgeon as gs

from common.IndexSelectionOperator import IndexSelectionOperator


class Where(IndexSelectionOperator):
    expected_num_inputs = 3

    def generate(self, input_shapes, axis=0):
        assert len(input_shapes) == 2
        assert tuple(input_shapes[0]) == tuple(input_shapes[1])
        input_shape = input_shapes[0]
        dtype = np.float32
        var_inputs = [
            gs.Variable(name=self.new_tensor_name(), dtype=dtype, shape=input_shape)
            for _ in input_shapes
        ]
        outputs = [gs.Variable(name=self.new_tensor_name(), dtype=dtype)]
        condition_shape = [1] * 4
        condition_shape[axis] = input_shape[axis]
        condition_vals = np.random.choice(a=[False, True], size=condition_shape)
        condition_constant = gs.Constant(name=self.new_tensor_name(), values=condition_vals)
        node = gs.Node(op=self.op,
                       inputs=[condition_constant] + var_inputs,
                       outputs=outputs,
                       name=self.new_node_name())
        graph = gs.Graph(nodes=[node], inputs=var_inputs, outputs=node.outputs, name=node.name)
        graph.name = f'{self.op}_axis{axis}_orig'
        return graph

    @staticmethod
    def get_orig_axis(condition_shape, var_input_shape):
        C, H, W = var_input_shape[1:]
        condition_shape = tuple(condition_shape)
        orig_axis = 0
        if condition_shape == (1, C, 1, 1):
            orig_axis = 1
        elif condition_shape == (1, 1, H, 1):
            orig_axis = 2
        elif condition_shape == (1, 1, 1, W):
            orig_axis = 3
        return orig_axis

    @staticmethod
    def insert_transposes_if_needed(node, graph):
        condition = node.inputs[0]
        var_input_shape = node.inputs[1].shape
        orig_axis = Where.get_orig_axis(condition.shape, var_input_shape)
        assert orig_axis != 0, 'Got orig_axis == 0, this op should not have qualified for reconstruction in the first place'
        transposes_needed = IndexSelectionOperator.insert_transposes_if_needed(
            node, graph, orig_axis)
        if transposes_needed:
            if orig_axis == 2:
                first_permute = (0, 2, 3, 1)
            elif orig_axis == 3:
                first_permute = (0, 3, 2, 1)
            elif orig_axis == 1:
                assert False, 'Should not have required to transpose'
            else:
                assert False, 'Not implemented'
            condition.values = np.transpose(condition.values, first_permute)

    @classmethod
    def qualifies_for_reconstruction(cls, node):
        result = node.op == cls.op_to_reconstruct()
        if result:
            result = result and len(node.inputs) == cls.expected_num_inputs
            result = result and isinstance(node.inputs[0], gs.Constant)
            result = result and not isinstance(node.inputs[1], gs.Constant) and len(
                node.inputs[1].shape) == 4
            result = result and not isinstance(node.inputs[2], gs.Constant) and len(
                node.inputs[0].shape) == 4
            result = result and tuple(node.inputs[1].shape) == tuple(node.inputs[2].shape)
        if result:
            condition_shape = node.inputs[0].shape
            var_input_shape = node.inputs[1].shape
            orig_axis = Where.get_orig_axis(condition_shape, var_input_shape)
            result &= orig_axis in {1, 2, 3}
        return result

    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            Where.insert_transposes_if_needed(node, graph)
            condition = node.inputs[0]
            var_inputs = node.inputs[1:]
            channels_out = var_inputs[0].shape[1]
            channels_in = 2 * channels_out
            dtype = var_inputs[0].dtype
            concat_name = f'{node.name}_concat'
            concat_out = gs.Variable(name=concat_name, dtype=dtype)
            concat_node = gs.Node(op='Concat',
                                  inputs=var_inputs,
                                  outputs=[concat_out],
                                  name=concat_name,
                                  attrs=dict(axis=1))
            graph.nodes.append(concat_node)
            node.op = 'Conv'
            weight_shape = (channels_out, channels_in, 1, 1)
            weight_vals = np.zeros(weight_shape, dtype=np.float32)
            for c_out, selector in enumerate(condition.values.flatten()):
                selector_idx = 0 if selector else 1
                weight_vals[c_out, selector_idx * channels_out + c_out, :, :] = 1
            weight_constant = gs.Constant(name=f'{node.name}_tmp0', values=weight_vals)
            node.inputs = [concat_out, weight_constant]
            graph.cleanup()

    def test(self, input_shapes=None, axes=None):
        input_shapes = input_shapes or [(1, 3, 5, 7)] * 2
        axes = axes or [1, 2, 3]
        maxabsdiff = 0.0
        for axis in axes:
            orig_graph = self.generate(input_shapes, axis)
            orig_path = IndexSelectionOperator.save_gs_graph(orig_graph, run_shape_inference=True)
            reconstructed_graph = gs.import_onnx(onnx.load(orig_path))
            for node in reconstructed_graph.nodes:
                self.reconstruct(node, reconstructed_graph)
            reconstructed_graph.name = orig_graph.name.replace('_orig', '_reconstructed')
            reconstructed_path = IndexSelectionOperator.save_gs_graph(reconstructed_graph)
            maxabsdiff_axis = self.run_comparison([orig_path, reconstructed_path],
                                                  input_shapes=input_shapes)
            maxabsdiff = max(maxabsdiff_axis, maxabsdiff)
        return maxabsdiff


def main():
    op = Where()
    op.test()


if __name__ == '__main__':
    main()
