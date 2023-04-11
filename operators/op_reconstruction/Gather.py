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
"""Reconstruction of a Gather op."""

from logging import info
import onnx
import numpy as np
import onnx_graphsurgeon as gs

from common.IndexSelectionOperator import IndexSelectionOperator


class Gather(IndexSelectionOperator):
    expected_num_inputs = 2

    def generate(self, input_shapes, attrs, num_indices):
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        dtype = np.float32
        var_input = gs.Variable(name=self.new_tensor_name(), dtype=dtype, shape=input_shape)
        outputs = [gs.Variable(name=self.new_tensor_name(), dtype=dtype)]
        axis = attrs.get('axis', 0)
        index_vals = np.arange(input_shape[axis], dtype=np.int64)
        np.random.shuffle(index_vals)
        index_vals = index_vals[:num_indices]
        index_constant = gs.Constant(name=self.new_tensor_name(), values=index_vals)
        node = gs.Node(op=self.op,
                       inputs=[var_input, index_constant],
                       outputs=outputs,
                       name=self.new_node_name())
        node.attrs = attrs
        graph = gs.Graph(nodes=[node], inputs=[var_input], outputs=node.outputs, name=node.name)
        graph.name = f'{self.op}_axis{axis}_orig'
        return graph

    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            orig_axis = node.attrs.get('axis', 0)
            transposes_needed = IndexSelectionOperator.insert_transposes_if_needed(
                node, graph, orig_axis)
            if transposes_needed:
                node.attrs['axis'] = 1
            axis = node.attrs.get('axis', 0)
            assert axis == 1
            channels_in = node.inputs[0].shape[1]
            indices = node.inputs[1]
            assert len(indices.shape) == 1
            channels_out = indices.shape[0]
            node.op = 'Conv'
            node.attrs.clear()
            weight_shape = (channels_out, channels_in, 1, 1)
            weight_vals = np.zeros(weight_shape, dtype=np.float32)
            for c_out, c_in in enumerate(indices.values.flatten()):
                weight_vals[c_out, c_in, :, :] = 1
            weight_constant = gs.Constant(name=f'{node.name}_tmp0', values=weight_vals)
            node.inputs = [node.inputs[0], weight_constant]
            graph.cleanup()

    def test(self, input_shapes=None, num_indices=None, axes=None):
        input_shapes = input_shapes or [(1, 5, 6, 7)]
        num_indices = num_indices or 3
        axes = axes or [1, 2, 3]
        maxabsdiff = 0.0
        for axis in axes:
            attrs = dict(axis=axis)
            orig_graph = self.generate(input_shapes, attrs, num_indices)
            orig_path = IndexSelectionOperator.save_gs_graph(orig_graph, run_shape_inference=True)
            reconstructed_graph = gs.import_onnx(onnx.load(orig_path))
            for node in reconstructed_graph.nodes:
                self.reconstruct(node, reconstructed_graph)
            reconstructed_graph.name = orig_graph.name.replace('_orig', '_reconstructed')
            reconstructed_path = IndexSelectionOperator.save_gs_graph(reconstructed_graph)
            maxabsdiff_axis = self.run_comparison([orig_path, reconstructed_path],
                                                  input_shapes=input_shapes,
                                                  incremental_vals=True)
            maxabsdiff = max(maxabsdiff_axis, maxabsdiff)
        return maxabsdiff


def main():
    op = Gather()
    op.test()


if __name__ == '__main__':
    main()
