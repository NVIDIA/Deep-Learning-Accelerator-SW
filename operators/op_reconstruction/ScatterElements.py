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
"""Reconstruction of a ScatterElements op."""

from logging import info
import onnx
import numpy as np
import onnx_graphsurgeon as gs

from common.IndexSelectionOperator import IndexSelectionOperator
from pdb import set_trace

SCATTER_ELEMENTS_SUPPORTED_REDUCTIONS = {'none'}


class ScatterElements(IndexSelectionOperator):
    expected_num_inputs = 3

    def generate(self, input_shapes, attrs, num_indices):
        assert len(input_shapes) == 2
        data_shape, updates_shape = input_shapes
        dtype = np.float32
        var_input_data = gs.Variable(name=self.new_tensor_name(), dtype=dtype, shape=data_shape)
        var_input_updates = gs.Variable(name=self.new_tensor_name(),
                                        dtype=dtype,
                                        shape=updates_shape)
        outputs = [gs.Variable(name=self.new_tensor_name(), dtype=dtype)]
        axis = attrs.get('axis', 0)
        index_vals = np.arange(data_shape[axis], dtype=np.int64)
        np.random.shuffle(index_vals)
        index_shape = [1] * 4
        index_shape[axis] = num_indices
        index_vals = index_vals[:num_indices].reshape(index_shape)
        index_tile_dims = list(updates_shape)
        index_tile_dims[0] = 1
        index_tile_dims[axis] = 1
        index_vals = np.tile(index_vals, index_tile_dims)

        index_constant = gs.Constant(name=self.new_tensor_name(), values=index_vals)
        node = gs.Node(op=self.op,
                       inputs=[var_input_data, index_constant, var_input_updates],
                       outputs=outputs,
                       name=self.new_node_name())
        node.attrs = attrs
        graph = gs.Graph(nodes=[node],
                         inputs=[var_input_data, var_input_updates],
                         outputs=node.outputs,
                         name=node.name)
        graph.name = f'{self.op}_axis{axis}_orig'
        return graph

    @classmethod
    def qualifies_for_reconstruction(cls, node):
        result = super().qualifies_for_reconstruction(node)
        nonconst_inputs = [x for x in node.inputs if not isinstance(x, gs.Constant)]
        axis = node.attrs.get('axis', 0)
        if result:
            reduction = node.attrs.get('reduction', 'none')
            result = reduction in SCATTER_ELEMENTS_SUPPORTED_REDUCTIONS
            result &= len(nonconst_inputs) == 2
        if result:
            indices = node.inputs[1].values
            for slice_idx in range(indices.shape[axis]):
                slice_data = np.take(indices, slice_idx, axis).flatten()
                if not (slice_data[0] == slice_data).all():
                    result = False
                    break
        return result

    @staticmethod
    def insert_transposes_if_needed(node, graph):
        orig_axis = node.attrs.get('axis', 0)
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
            node.attrs['axis'] = 1
            indices = node.inputs[1]
            indices.values = np.transpose(indices.values, first_permute)

    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            cls.insert_transposes_if_needed(node, graph)
            axis = node.attrs.get('axis', 0)
            assert axis == 1
            data, indices, updates = node.inputs
            index_vals = indices.values[0, :, 0, 0]
            indices_set = set(index_vals)
            channels_in_conv = updates.shape[1]
            channels_out_conv = data.shape[1]
            conv_weight_shape = (channels_out_conv, channels_in_conv, 1, 1)
            conv_weight_vals = np.zeros(conv_weight_shape, dtype=np.float32)
            for c_in, c_out in enumerate(index_vals):
                conv_weight_vals[c_out, c_in, :, :] = 1
            conv_output = gs.Variable(name=f'{node.name}_conv_tmp0', dtype=updates.dtype)
            conv_weight_constant = gs.Constant(name=f'{node.name}_conv_tmp1',
                                               values=conv_weight_vals)
            conv_node = gs.Node(op='Conv',
                                inputs=[updates, conv_weight_constant],
                                outputs=[conv_output],
                                name=f'{node.name}_conv')
            graph.nodes.append(conv_node)
            scale_weight_shape = (1, channels_out_conv, 1, 1)
            scale_weight_vals = np.zeros(scale_weight_shape, dtype=np.float32)
            for c in range(scale_weight_vals.size):
                if c not in indices_set:
                    scale_weight_vals[0][c] = 1.0
            scale_output = gs.Variable(name=f'{node.name}_scale_tmp0', dtype=updates.dtype)
            scale_weight_constant = gs.Constant(name=f'{node.name}_scale_tmp1',
                                                values=scale_weight_vals)
            scale_node = gs.Node(op='Mul',
                                 inputs=[data, scale_weight_constant],
                                 outputs=[scale_output],
                                 name=f'{node.name}_scale')
            graph.nodes.append(scale_node)
            node.op = 'Add'
            node.inputs = [scale_output, conv_output]
            node.attrs.clear()

            graph.cleanup()

    def test(self, input_shapes=None, num_indices=None, axes=None):
        input_shapes = input_shapes or [(1, 5, 6, 7)]
        assert isinstance(input_shapes, list) and len(input_shapes) == 1
        input_shape_data = input_shapes[0]
        num_indices = num_indices or 3
        axes = axes or [1, 2, 3]
        maxabsdiff = 0.0
        for axis in axes:
            attrs = dict(axis=axis)
            input_shape_updates = list(input_shape_data)
            input_shape_updates[axis] = num_indices
            input_shape_updates = tuple(input_shape_updates)
            var_input_shapes = [input_shape_data, input_shape_updates]
            orig_graph = self.generate(var_input_shapes, attrs, num_indices)
            orig_path = IndexSelectionOperator.save_gs_graph(orig_graph, run_shape_inference=True)
            reconstructed_graph = gs.import_onnx(onnx.load(orig_path))
            for node in reconstructed_graph.nodes:
                self.reconstruct(node, reconstructed_graph)
            reconstructed_graph.name = orig_graph.name.replace('_orig', '_reconstructed')
            reconstructed_path = IndexSelectionOperator.save_gs_graph(reconstructed_graph,
                                                                      run_shape_inference=True)
            maxabsdiff_axis = self.run_comparison([orig_path, reconstructed_path],
                                                  input_shapes=var_input_shapes)
            maxabsdiff = max(maxabsdiff_axis, maxabsdiff)
        return maxabsdiff


def main():
    op = ScatterElements()
    op.test()


if __name__ == '__main__':
    main()
