#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
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
