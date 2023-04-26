<!--- SPDX-License-Identifier: Apache-2.0 -->

# Supported ONNX Operators on Orin DLA

DLA operator functionality is exposed through the TensorRT builder, which internally links to DLA SW libraries (see [DLA Workflow](https://developer.nvidia.com/deep-learning-accelerator)). While some operators may already be available in DLA SW, TensorRT may not expose them yet.
See below for the support matrix of ONNX operators on Orin DLA. If you are interested in a specific DLA operator that is not supported through TensorRT yet, feel free to raise a [GitHub Issue](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/issues) and/or inform your NVIDIA representative (in particular for NVIDIA DRIVE customers).

See [General Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest) that apply to all operators. Many of those operators are supported on Xavier DLA as well, see [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest).

TensorRT 8.5 supports operators up to Opset 17. Latest information of ONNX operators can be found [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md).

Note that the scripts in `op_reconstruction/` are intended as a recipe for how ops currently not supported on DLA can be decomposed into supported ops. Depending on your setup, you may choose to perform such op reconstructions in the ONNX domain post-training (as done here) or during the training process (for example in TensorFlow or PyTorch).

Below Operator Support Matrix requires the following minimum system config (the OS by default gets shipped with the DLA SW and TensorRT versions to its right):

| **Hardware platform** | **OS**               | **DLA SW version** | **TensorRT version** |
| ----------------- | ---------------- | -------------- | ---------------- |
| DRIVE Orin (Automotive)        | DRIVE OS 6.0.6.0 | DLA 3.12.0     | TensorRT 8.5.10  |
| Jetson Orin (Embedded)      | JetPack 5.1.1    | DLA 3.12.1     | TensorRT 8.5.2   |

## Request for enhancements (RFE)

If you are interested in a specific DLA operator to be enabled in TensorRT, feel free to raise a [GitHub Issue](https://github.com/NVIDIA/TensorRT/issues) and/or inform your NVIDIA representative (in particular for NVIDIA DRIVE customers).

## Operator Support Matrix

| Operator                  | Orin DLA support through TensorRT | Orin DLA support through DLA SW   | Restrictions                                                                                                           |
|---------------------------|--------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------|
| Abs                       | Native                   | Native | See **Unary layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Acos                      | [See RFE](#request-for-enhancements-rfe)                   | Planned in future release |
| Acosh                     | [See RFE](#request-for-enhancements-rfe)          | Planned in future release  |
| Add                       | Native                   | Native | See **Scale layer** and **ElementWise Layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| And                       | Reconstruction          | Reconstruction | See [op_reconstruction/And.py](op_reconstruction/And.py)
| Asin                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release  |
| Asinh                     | [See RFE](#request-for-enhancements-rfe)          | Planned in future release  |
| Atan                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release  |
| Atanh                     | [See RFE](#request-for-enhancements-rfe)          | Planned in future release  |
| AveragePool               | Native        | Native | See **Pooling layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)                                                                                                           |
| BatchNormalization        | Native          | Native | See **Scale layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Celu                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Clip                      | Native          | Native |  See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/indexhtml#dla-lay-supp-rest)                                                                                      |
| Concat                    | Native          | Native | See **Concatenation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Constant                  | Native          | Native | TensorRT performs constant folding into supported DLA ops.
| ConstantOfShape           | Can be inferred at build time          | Can be inferred at build time  |
| Conv                      | Native          | Native | See **Convolution and Fully Connected layers** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| ConvTranspose             | Native          | Native | See **Deconvolution layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Cos                       | Native          | Planned in future release|
| Cosh                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| DepthToSpace              | Reconstruction          | Reconstruction | See [op_reconstruction/DepthToSpace.py](op_reconstruction/DepthToSpace.py)
| Div                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Elu                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Equal                     | Native         | Native | See **Equal operation** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Erf                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Exp                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Gather                    | Reconstruction          | Reconstruction | See [op_reconstruction/Gather.py](op_reconstruction/Gather.py)
| Gemm                      | Native          | Native | See **Convolution and Fully Connected layers** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| GlobalAveragePool         | [See RFE](#request-for-enhancements-rfe)          | Native |
| GlobalMaxPool             | [See RFE](#request-for-enhancements-rfe)          | Native |
| Greater                   | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| GreaterOrEqual            | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| GRU                       | Reconstruction          | Reconstruction | See [op_reconstruction/GRU.py](op_reconstruction/GRU.py)
| HardSwish                 | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| HardSigmoid               | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| LeakyRelu                 | Native          | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Less                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| LessOrEqual               | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Log                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| LogSoftmax                | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| LRN                       | Native          | Native | See **LRN (Local Response Normalization) layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| MatMul                    | Native          | Native | The second input must be a constant, also see **Convolution and Fully Connected layers** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Max                       | Native          | Native| See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| MaxPool                   | Native          | Native| See **Pooling layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Min                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Mod                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Mul                       | Native          | Native | See **Scale layer** and **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Neg                       | Reconstruction          | Reconstruction |  See [op_reconstruction/Neg.py](op_reconstruction/Neg.py)
| Not                       | Reconstruction          | Reconstruction | See [op_reconstruction/Not.py](op_reconstruction/Not.py)
| Or                        | Reconstruction          | Reconstruction | See [op_reconstruction/Or.py](op_reconstruction/Or.py)
| Pow                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| PRelu                     | Native          | Native | See **Parametric ReLU layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Reciprocal                | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| ReduceMax                 | [See RFE](#request-for-enhancements-rfe)          | Native|
| ReduceMean                | [See RFE](#request-for-enhancements-rfe)          | Native|
| ReduceMin                 | [See RFE](#request-for-enhancements-rfe)          | Native|
| Relu                      | Native          | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Reshape                   | Native          | Native | See **Shuffle layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Resize                    | Native         | Native | See **Resize layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)   |
| ScatterElements           | Reconstruction          | Reconstruction | See [op_reconstruction/ScatterElements.py](op_reconstruction/ScatterElements.py)
| Shape                     |Can be inferred at build time          | Can be inferred at build time  |
| Sigmoid                   | Native          | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Sign                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Sin                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Sinh                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Slice                     | Native         | Native | See **Slice layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)                                                                                                |
| Softmax                   | Native         | Native | See **SoftMax layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Softplus                  | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Softsign                  | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| SpaceToDepth              | Reconstruction          | Reconstruction | See [op_reconstruction/SpaceToDepth.py](op_reconstruction/SpaceToDepth.py)
| Sqrt                      | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Sub                       | Native         | Native | See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Sum                       | Native         | Native | See **ElementWise layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Tan                       | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Tanh                      | Native         | Native | See **Activation layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| ThresholdedRelu           | [See RFE](#request-for-enhancements-rfe)          | Planned in future release |
| Transpose                 | Native         | Native | See **Shuffle layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Upsample                  | Native         | Native | See **Resize layer** in [Layer Support and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-lay-supp-rest)
| Where                     | Reconstruction          | Reconstruction | See [op_reconstruction/Where.py](op_reconstruction/Where.py)
| Xor                       | Reconstruction          | Reconstruction | See [op_reconstruction/Xor.py](op_reconstruction/Xor.py)
