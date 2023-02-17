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
"""Reconstruction of a Not op."""

from logging import info
import numpy as np
import onnx_graphsurgeon as gs

from common.BinaryOperator import BinaryOperator


class Not(BinaryOperator):
    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            node.op = 'Sub'
            for tensor in node.inputs + node.outputs:
                tensor.dtype = np.float32
            not_var = node.inputs[0]
            not_constant = gs.Constant(name=f'{node.name}_tmp0',
                                       values=np.ones(not_var.shape, dtype=not_var.dtype))
            node.inputs = [not_constant, not_var]

    def test(self):
        input_shape = (1, 2, 1, 1)
        input_data = np.array([True, False]).reshape(input_shape)
        return super().test(input_data=input_data)


def main():
    op = Not()
    op.test()


if __name__ == '__main__':
    main()
