#
# SPDX-FileCopyrightText: Copyright (c) 2022 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Reconstruction of a SpaceToDepth op."""

from logging import info
import numpy as np
import onnx_graphsurgeon as gs

from common.Operator import Operator


class SpaceToDepth(Operator):
    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            blocksize = node.attrs.get('blocksize', 1)
            channels_in, height_in, width_in = node.inputs[0].shape[1:]
            assert height_in % blocksize == 0
            assert width_in % blocksize == 0
            channels_out = channels_in * blocksize * blocksize
            node.op = 'Conv'
            node.attrs = dict(kernel_shape=(blocksize, blocksize), strides=(blocksize, blocksize))
            weight_shape = (channels_out, channels_in, blocksize, blocksize)
            weight_vals = np.zeros(weight_shape, dtype=np.float32)
            stacked_kernels = weight_vals.reshape(channels_out,
                                                  channels_in * blocksize * blocksize)
            for block_idx in range(channels_out):
                kernel_one_idx = block_idx // channels_in + (block_idx %
                                                             channels_in) * blocksize * blocksize
                stacked_kernels[block_idx][kernel_one_idx] = 1
            weight_vals = stacked_kernels.reshape(weight_shape)
            weight_constant = gs.Constant(name=f'{node.name}_tmp0', values=weight_vals)
            node.inputs.append(weight_constant)

    def test(self, input_shapes=None, blocksize=None):
        input_shapes = input_shapes or [(1, 5, 768, 768)]
        blocksize = blocksize or 4
        attrs = dict(blocksize=blocksize)
        return super().test(input_shapes, attrs=attrs)


def main():
    op = SpaceToDepth()
    op.test()


if __name__ == '__main__':
    main()
