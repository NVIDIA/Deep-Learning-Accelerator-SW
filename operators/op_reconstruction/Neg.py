#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
