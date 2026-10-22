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
