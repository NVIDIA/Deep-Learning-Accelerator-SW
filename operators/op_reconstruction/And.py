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
"""Reconstruction of an And op."""

from logging import info
from common.BinaryOperator import BinaryOperator
import numpy as np


class And(BinaryOperator):
    @classmethod
    def reconstruct(cls, node, graph):
        if cls.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            node.op = 'Mul'
            for tensor in node.inputs + node.outputs:
                tensor.dtype = np.float32

    def test(self):
        input_shapes = [(1, 4, 1, 1)] * 2
        input_data = list()
        input_data.append(np.array([True, True, False, False]).reshape(input_shapes[0]))
        input_data.append(np.array([True, False, True, False]).reshape(input_shapes[1]))
        return super().test(input_data=input_data)


def main():
    op = And()
    op.test()


if __name__ == '__main__':
    main()
