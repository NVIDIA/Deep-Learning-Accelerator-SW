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
"""BinaryOperator class for op reconstructions."""

import onnx
import numpy as np
import onnx_graphsurgeon as gs
from common.Operator import Operator


class BinaryOperator(Operator):
    def generate(self, input_shapes, **kwargs):
        kwargs['dtype'] = bool
        return super().generate(input_shapes, **kwargs)

    def test(self, input_data):
        if not isinstance(input_data, list):
            input_data = [input_data]
        input_shapes = [data.shape for data in input_data]
        orig_graph = self.generate(input_shapes)
        orig_path = Operator.save_gs_graph(orig_graph, run_shape_inference=True)
        reconstructed_graph = gs.import_onnx(onnx.load(orig_path))
        for node in reconstructed_graph.nodes:
            self.reconstruct(node, reconstructed_graph)
        reconstructed_graph.name = orig_graph.name.replace('_orig', '_reconstructed')
        reconstructed_path = Operator.save_gs_graph(reconstructed_graph)
        maxabsdiff = self.run_comparison([orig_path, reconstructed_path], input_data=input_data)
        return maxabsdiff
