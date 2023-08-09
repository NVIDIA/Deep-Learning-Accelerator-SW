#
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Reconstruction of a Xor op."""

from logging import info
import numpy as np
import onnx_graphsurgeon as gs

from common.BinaryOperator import BinaryOperator


class Xor(BinaryOperator):
    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            node.op = 'Sub'
            for tensor in node.inputs + node.outputs:
                tensor.dtype = np.float32
            tmp_tensor = gs.Variable(name=f'{node.name}_tmp0', dtype=np.float32)
            abs_node = gs.Node(op='Abs',
                               inputs=[tmp_tensor],
                               outputs=[node.outputs[0]],
                               name=f'{node.name}_abs')
            node.outputs = [tmp_tensor]
            graph.nodes.append(abs_node)

    def test(self):
        input_shapes = [(1, 4, 1, 1)] * 2
        input_data = list()
        input_data.append(np.array([True, True, False, False]).reshape(input_shapes[0]))
        input_data.append(np.array([True, False, True, False]).reshape(input_shapes[1]))
        return super().test(input_data=input_data)


def main():
    op = Xor()
    op.test()


if __name__ == '__main__':
    main()
