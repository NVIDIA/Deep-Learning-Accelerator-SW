#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""BinaryOperator class for op reconstructions."""

import onnx
import numpy as np
import onnx_graphsurgeon as gs
from common.Operator import Operator


class BinaryOperator(Operator):
    def generate(self, input_shapes, **kwargs):
        kwargs['dtype'] = np.bool
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
