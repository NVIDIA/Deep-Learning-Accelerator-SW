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
"""Utils for ONNX Runtime inference."""
import onnxruntime as ort
import numpy as np

ONNXRUNTIME_DT_TO_NUMPY_DT = {'tensor(bool)': bool, 'tensor(float)': np.float32}



def onnx_inference(onnx_file, input_data, session_options=None):
    session = ort.InferenceSession(onnx_file, session_options)
    if not isinstance(input_data, list):
        input_data = [input_data]
    session_inputs = session.get_inputs()
    assert len(session_inputs) == len(input_data)
    input_dict = dict()
    for tensor, data in zip(session_inputs, input_data):
        assert tensor.type in ONNXRUNTIME_DT_TO_NUMPY_DT, 'No mapping from ONNX RT to NumPy data type detected, you may need to extend it'
        dtype = ONNXRUNTIME_DT_TO_NUMPY_DT[tensor.type]
        input_dict[tensor.name] = data.astype(dtype)
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], input_dict)
    return output
