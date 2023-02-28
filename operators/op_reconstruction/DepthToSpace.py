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
"""Reconstruction of a DepthToSpace op."""

from logging import info
import numpy as np
import onnx_graphsurgeon as gs

from common.Operator import Operator


class DepthToSpace(Operator):
    def generate(self, input_shapes, attrs):
        graph = super().generate(input_shapes, attrs=attrs)
        mode = attrs.get('mode', 'CRD')
        graph.name = f'{self.op}_{mode}_orig'
        return graph

    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            mode = node.attrs.get('mode', 'CRD')
            blocksize = node.attrs.get('blocksize', 1)
            channels_in = node.inputs[0].shape[1]
            assert channels_in % (blocksize * blocksize) == 0
            channels_out = channels_in // (blocksize * blocksize)
            node.op = 'ConvTranspose'
            node.attrs = dict(kernel_shape=(blocksize, blocksize), strides=(blocksize, blocksize))
            weight_shape = (channels_in, channels_out, blocksize, blocksize)
            weight_vals = np.zeros(weight_shape, dtype=np.float32)
            if mode == 'CRD':
                for c in range(channels_in):
                    filter_slice = weight_vals[c, :].flatten()
                    filter_slice[c] = 1
                    weight_vals[c, :] = filter_slice.reshape(channels_out, blocksize, blocksize)
            elif mode == 'DCR':
                stacked_kernels = weight_vals.reshape(channels_in,
                                                      channels_out * blocksize * blocksize)
                for block_idx in range(channels_in):
                    kernel_one_idx = block_idx // channels_out + (
                        block_idx % channels_out) * blocksize * blocksize
                    stacked_kernels[block_idx][kernel_one_idx] = 1
                weight_vals = stacked_kernels.reshape(weight_shape)
            else:
                assert False, f'Unknown DepthToSpace mode: {mode}'
            weight_constant = gs.Constant(name=f'{node.name}_tmp0', values=weight_vals)
            node.inputs.append(weight_constant)

    def test(self, input_shapes=None, blocksize=None, modes=None):
        input_shapes = input_shapes or [(1, 512, 48, 48)]
        blocksize = blocksize or 4
        modes = modes or ['CRD', 'DCR']
        maxabsdiff = 0.0
        for mode in modes:
            attrs = dict(blocksize=blocksize, mode=mode)
            maxabsdiff = max(super().test(input_shapes, attrs=attrs), maxabsdiff)
        return maxabsdiff


def main():
    op = DepthToSpace()
    op.test()


if __name__ == '__main__':
    main()
