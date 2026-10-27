#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""IndexSelectionOperator class for reconstructions of Gather/Scatter-style ops."""

import numpy as np
import onnx_graphsurgeon as gs

from common.Operator import Operator


class IndexSelectionOperator(Operator):
    expected_num_inputs = 1

    @staticmethod
    def insert_transposes_if_needed(node, graph, axis):
        transposes_needed = False
        first_permute = None
        second_permute = None
        if axis == 1:
            transposes_needed = False
        elif axis == 2:
            # move axis 2 to 1
            first_permute = (0, 2, 3, 1)
            # move axis 1 to 2 again
            second_permute = (0, 3, 1, 2)
            transposes_needed = True
        elif axis == 3:
            # move axis 3 to 1
            first_permute = (0, 3, 2, 1)
            # move axis 1 to 3 again
            second_permute = (0, 3, 2, 1)
            transposes_needed = True
        else:
            assert False, 'Not implemented'
        if transposes_needed:
            nonconst_inputs = [(idx, x) for idx, x in enumerate(node.inputs)
                               if not isinstance(x, gs.Constant)]
            new_inputs = list(node.inputs)
            for orig_idx, inp in nonconst_inputs:
                orig_input_shape = inp.shape
                new_input_shape = [orig_input_shape[x] for x in first_permute]
                first_tmp_tensor = gs.Variable(name=f'{node.name}_transpose0_{orig_idx}',
                                               dtype=np.float32,
                                               shape=new_input_shape)
                first_transpose = gs.Node(op='Transpose',
                                          inputs=[inp],
                                          outputs=[first_tmp_tensor],
                                          attrs=dict(perm=first_permute),
                                          name=first_tmp_tensor.name)
                graph.nodes.append(first_transpose)
                new_inputs[orig_idx] = first_tmp_tensor
            node.inputs = new_inputs
            second_tmp_tensor = gs.Variable(name=f'{node.name}_tmp1', dtype=np.float32, shape=None)
            second_transpose = gs.Node(op='Transpose',
                                       inputs=[second_tmp_tensor],
                                       outputs=[node.outputs[0]],
                                       attrs=dict(perm=second_permute),
                                       name=second_tmp_tensor.name)
            graph.nodes.append(second_transpose)
            node.outputs = [second_tmp_tensor]
            graph.cleanup().toposort()
        return transposes_needed

    @classmethod
    def qualifies_for_reconstruction(cls, node):
        result = super().qualifies_for_reconstruction(node)
        if result:
            axis = node.attrs.get('axis', 0)
            result = result and len(node.inputs) == cls.expected_num_inputs
            result &= result and isinstance(node.inputs[1], gs.Constant)
            result &= result and axis in {1, 2, 3}
        return result
