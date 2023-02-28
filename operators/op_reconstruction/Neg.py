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
"""Reconstruction of a Neg op."""

from logging import info
import numpy as np
import onnx_graphsurgeon as gs

from common.Operator import Operator


class Neg(Operator):
    def generate(self, input_shapes, **kwargs):
        # modify here
        graph = super().generate(input_shapes, **kwargs)
        return graph

    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            node.op = 'Mul'
            node.attrs.clear()
            mul_shape = (1, )
            mul_vals = np.zeros(mul_shape, dtype=np.float32)
            mul_vals.fill(-1.0)
            mul_constant = gs.Constant(name=f'{node.name}_tmp0', values=mul_vals)
            node.inputs.append(mul_constant)

    def test(self, input_shapes=[(1, 2, 3, 4)], **kwargs):
        return super().test(input_shapes, **kwargs)


def main():
    op = Neg()
    op.test()


if __name__ == '__main__':
    main()
