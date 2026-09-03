<!--- SPDX-License-Identifier: Apache-2.0 -->

# Supported ONNX Operators & Functions on Orin DLA

DLA operator functionality is exposed through the TensorRT builder, which internally links to DLA SW libraries (see [DLA Workflow](https://developer.nvidia.com/deep-learning-accelerator)). While some ONNX operators or functions may already be available in DLA SW, TensorRT may not expose them yet.
See below for the support matrix of ONNX operators & functions on Orin DLA. If you are interested in a specific DLA operator that is not supported through TensorRT yet, feel free to raise a [GitHub Issue](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/issues) and/or inform your NVIDIA representative (in particular for NVIDIA DRIVE customers).

See [General Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest) that apply to all operations below. Many of those ops are supported on Xavier DLA as well, see [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest).

TensorRT 8.6 supports operators up to Opset 17. Latest information of ONNX operators can be found [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md).

Note that the scripts in `op_reconstruction/` are intended as a recipe for how ops currently not supported on DLA can be decomposed into supported ops. Depending on your setup, you may choose to perform such op reconstructions in the ONNX domain post-training (as done here) or during the training process (for example in TensorFlow or PyTorch). The case of "Native" in the DLA SW support column and "Reconstruction" in the TensorRT support column indicates that an op can be supported through TensorRT by decomposing it into other DLA ops already supported by TensorRT.

Below Operator Support Matrix requires the following minimum system config (the OS by default gets shipped with the DLA SW and TensorRT versions to its right):

| **Hardware platform** | **OS**               | **DLA SW version** | **TensorRT version** |
| ----------------- | ---------------- | -------------- | ---------------- |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.6.0 | DLA 3.12.0     | TensorRT 8.5.10  |
| Jetson Orin (Embedded)      | JetPack 5.1.1    | DLA 3.12.1     | TensorRT 8.5.2   |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.7.0 | DLA 3.13.0     | TensorRT 8.6.10  |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.8.0 | DLA 3.14.0     | TensorRT 8.6.11  |
| Jetson Orin (Embedded)      | JetPack 6    | DLA 3.14.1     | TensorRT 8.6.2   |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.8.1 | DLA 3.14.1     | TensorRT 8.6.11  |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.9.0 | DLA 3.14.2     | TensorRT 8.6.12  |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.9.1 | DLA 3.14.3     | TensorRT 8.6.12  |


## Request for enhancements (RFE)

If you are interested in a specific DLA operator to be enabled in TensorRT, feel free to raise a [GitHub Issue](https://github.com/NVIDIA/TensorRT/issues) and/or inform your NVIDIA representative (in particular for NVIDIA DRIVE customers).

## ONNX Operator/Function Support Matrix

| Operator/Function         | Orin DLA support through TensorRT | Orin DLA support through DLA SW   | Restrictions                                                                                                           |
|---------------------------|--------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------|
| Abs                       | Native                   | Native | See **Unary layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Acos                      | [See RFE](#request-for-enhancements-rfe)                   | Native (as of DLA 3.14.0) |
| Acosh                     | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0)  |
| Add                       | Native                   | Native | See **Scale layer** and **ElementWise Layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| And                       | Reconstruction          | Reconstruction | See [op_reconstruction/And.py](op_reconstruction/And.py)
| Asin                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0)  |
| Asinh                     | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0)  |
| Atan                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0)  |
| Atanh                     | Native (as of TensorRT 8.6.12)          | Native (as of DLA 3.14.0)  | See **Unary layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| AveragePool               | Native        | Native | See **Pooling layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)                                                                                                           |
| BatchNormalization        | Native          | Native | See **Scale layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Ceil                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Celu                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Clip                      | Native          | Native |  See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)                                                                                      |
| Concat                    | Native          | Native | See **Concatenation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Constant                  | Native          | [See RFE](#request-for-enhancements-rfe) | TensorRT performs constant folding into supported DLA ops. You may need to allow GPU fallback to trigger the folding but the final operator would only run on DLA.
| ConstantOfShape           | Can be inferred at build time          | Can be inferred at build time  |
| Conv                      | Native          | Native | See **Convolution and Fully Connected layers** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| ConvTranspose             | Native          | Native | See **Deconvolution layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Cos                       | Native (as of TensorRT 8.6.12)          | Native (as of DLA 3.14.0)  | See **Unary layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Cosh                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| CumSum                       | Reconstruction          | Reconstruction | With `axis=1`, it can be expressed through a 1x1 Conv
| DepthToSpace              | Reconstruction          | Native (as of DLA 3.14.0) | See [op_reconstruction/DepthToSpace.py](op_reconstruction/DepthToSpace.py)
| DequantizeLinear          | Reconstruction          | Reconstruction | Can be collapsed to INT8 scaling factor, switching from explicit to implicit quantization
| Div                       | Native (as of TensorRT 8.6.12) | Native (as of DLA 3.14.0) | See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Elu                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Equal                     | Native         | Native | See **Comparison operations** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Erf                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Exp                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Expand                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Flatten                       | [See RFE](#request-for-enhancements-rfe)          | [See RFE](#request-for-enhancements-rfe) |
| Floor                       | [See RFE](#request-for-enhancements-rfe)        | Native (as of DLA 3.14.0) |
| Gather                    | Reconstruction          | Reconstruction | See [op_reconstruction/Gather.py](op_reconstruction/Gather.py)
| GatherElements                    | Reconstruction          | Reconstruction | See [op_reconstruction/Gather.py](op_reconstruction/Gather.py) which works similarly
| GatherND                    | Reconstruction          | Reconstruction | See [op_reconstruction/Gather.py](op_reconstruction/Gather.py) which works similarly
| Gemm                      | Native          | [See RFE](#request-for-enhancements-rfe) | The second input must be a constant, also see **Convolution and Fully Connected layers** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest). TensorRT can translate this to a 1x1 Conv internally.
| GlobalAveragePool         | [See RFE](#request-for-enhancements-rfe)          | Native |
| GlobalLpPool             | [See RFE](#request-for-enhancements-rfe)          |  Native (as of DLA 3.14.0) |
| GlobalMaxPool             | Native (as of TensorRT 8.6.12)          | Native | See **Reduce layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest). Older releases: Can be reconstructed by decomposing into several MaxPool nodes.
| Greater                   | Native (as of TensorRT 8.6.12)         | Native (as of DLA 3.14.0) | See **Comparison operations** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| GreaterOrEqual            | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| GroupNormalization            | [See RFE](#request-for-enhancements-rfe)      | Reconstruction (as of DLA 3.14.0) | Can be expressed with ReduceMean, Sub, Pow, Add, Sqrt, Div, Mul
| GRU                       | Reconstruction          | Reconstruction | See [op_reconstruction/GRU.py](op_reconstruction/GRU.py)
| HardSigmoid               | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| HardSwish                 | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Identity                 | Reconstruction | Native (as of DLA 3.14.0) | Can be expressed with identity BatchNormalization (TensorRT Scale layer) or identity Slice for example
| InstanceNormalization            | [See RFE](#request-for-enhancements-rfe)      | Reconstruction (as of DLA 3.14.0) | Can be expressed with ReduceMean, Sub, Pow, Add, Sqrt, Div, Mul
| LayerNormalization            | [See RFE](#request-for-enhancements-rfe)      | Reconstruction (as of DLA 3.14.0) | Can be expressed with ReduceMean, Sub, Pow, Add, Sqrt, Div, Mul
| LeakyRelu                 | Native          | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Less                      | Native (as of TensorRT 8.6.12)          | Native (as of DLA 3.14.0) | See **Comparison operations** in [Layer Support and Restrictions]
| LessOrEqual               | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Log                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| LogSoftmax                       | [See RFE](#request-for-enhancements-rfe)          | Reconstruction (as of DLA 3.14.0) | Can be expressed with Log and Softmax
| LpNormalization                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| LpPool                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| LRN                       | Native          | Native | See **LRN (Local Response Normalization) layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| LSTM                       | Reconstruction          | Reconstruction | Can be unrolled similarly to [op_reconstruction/GRU.py](op_reconstruction/GRU.py)
| MatMul                    | Native          | [See RFE](#request-for-enhancements-rfe) | The second input must be a constant, also see **Convolution and Fully Connected layers** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest). TensorRT can translate this to a 1x1 Conv internally.
| Max                       | Native          | Native| See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| MaxPool                   | Native          | Native| See **Pooling layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Mean                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Min                       | [See RFE](#request-for-enhancements-rfe)          | Native |
| Mish                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Mod                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| MeanVarianceNormalization            | [See RFE](#request-for-enhancements-rfe)      | Reconstruction (as of DLA 3.14.0) | Can be expressed with ReduceMean, Sub, Pow, Add, Sqrt, Div, Mul
| Mul                       | Native          | Native | See **Scale layer** and **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Neg                       | Reconstruction          | Reconstruction |  See [op_reconstruction/Neg.py](op_reconstruction/Neg.py)
| Not                       | Reconstruction          | Reconstruction | See [op_reconstruction/Not.py](op_reconstruction/Not.py)
| Or                        | Reconstruction          | Reconstruction | See [op_reconstruction/Or.py](op_reconstruction/Or.py)
| Pad                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| PRelu                     | Native          | Native | See **Parametric ReLU layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| QuantizeLinear          | Reconstruction          | Reconstruction | Can be collapsed to INT8 scaling factor, switching from explicit to implicit quantization	
| Reciprocal                | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| ReduceMean                | [See RFE](#request-for-enhancements-rfe)          | Native| Can be reconstructed by decomposing into several AveragePool or Conv nodes
| ReduceMin                 | [See RFE](#request-for-enhancements-rfe)          | Native|
| ReduceLogSum                       | [See RFE](#request-for-enhancements-rfe)          | Reconstruction (as of DLA 3.14.0) | Can be expressed with Log, ReduceSum and Exp
| ReduceLogSumExp                       | [See RFE](#request-for-enhancements-rfe)          | Reconstruction (as of DLA 3.14.0) | Can be expressed with ReduceSum and Log
| ReduceL1                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| ReduceL2                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| ReduceSum                | Reconstruction          |  Native (as of DLA 3.14.0) | Can be reconstructed by decomposing into Conv nodes or several AveragePool nodes while re-scaling with for example BatchNormalization (TensorRT Scale layer)
| ReduceSumSquare                | [See RFE](#request-for-enhancements-rfe)          |  Native (as of DLA 3.14.0) |
| Relu                      | Native          | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Reshape                   | Native          | Native | See **Shuffle layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Resize                    | Native         | Native | See **Resize layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)   |
| RNN                       | Reconstruction          | Reconstruction | Can be unrolled similarly to [op_reconstruction/GRU.py](op_reconstruction/GRU.py)
| Round                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Scatter                    | Reconstruction          | Reconstruction | See [op_reconstruction/ScatterElements.py](op_reconstruction/ScatterElements.py)
| ScatterElements           | Reconstruction          | Reconstruction | See [op_reconstruction/ScatterElements.py](op_reconstruction/ScatterElements.py) which works similarly
| Selu                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Shape                     |Can be inferred at build time          | Can be inferred at build time  |
| Shrink                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Sigmoid                   | Native          | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Sign                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Sin                       | Native (as of TensorRT 8.6.12)          | Native (as of DLA 3.14.0)  | See **Unary layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Sinh                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Slice                     | Native         | Native | See **Slice layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)                                                                                                |
| Softmax                   | Native         | Native | See **SoftMax layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Softplus                  | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Softsign                  | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| SpaceToDepth              | Reconstruction          | Native (as of DLA 3.14.0) | See [op_reconstruction/SpaceToDepth.py](op_reconstruction/SpaceToDepth.py)
| Split                     | Native         | Native | Translated to Slice inside TensorRT's ONNX parser. See **Slice layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Sqrt                      | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Squeeze                       | [See RFE](#request-for-enhancements-rfe)          | [See RFE](#request-for-enhancements-rfe) |
| Sub                       | Native         | Native | See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Sum                       | Native         | Native | See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Tan                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Tanh                      | Native         | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| ThresholdedRelu           | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Transpose                 | Native         | Native | See **Shuffle layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Tile                       | [See RFE](#request-for-enhancements-rfe)          | Native (as of DLA 3.14.0) |
| Unsqueeze                       | [See RFE](#request-for-enhancements-rfe)          | [See RFE](#request-for-enhancements-rfe) |
| Upsample                  | Native         | Native | See **Resize layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Where                     | Reconstruction          | Reconstruction | See [op_reconstruction/Where.py](op_reconstruction/Where.py)
| Xor                       | Reconstruction          | Reconstruction | See [op_reconstruction/Xor.py](op_reconstruction/Xor.py)
