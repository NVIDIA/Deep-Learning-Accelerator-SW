# cuDLA samples

[cuDLA](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#cudla) is an extension of the CUDA programming model that integrates DLA (Deep Learning Accelerator) with CUDA, thereby making it possible to submit DLA tasks using CUDA programming constructs such as streams and graphs.

Refer to the [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-standalone-mode) to learn how to generate DLA loadables that can be run with cuDLA.

Here a list of cuDLA samples hosted in the official NVIDIA CUDA repo and in this repo.

# cuDLA samples in [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) repo

These are the samples showcasing how to use cuDLA APIs to run DLA loadables.

### [cuDLAErrorReporting - cuDLA Error Reporting](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/cuDLAErrorReporting)
This sample demonstrates how DLA errors can be detected via CUDA.

### [cuDLAHybridMode - cuDLA Hybrid Mode](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/cuDLAHybridMode)
This sample demonstrates [cuDLA hybrid mode](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-model) wherein DLA can be programmed using CUDA.

### [cuDLAStandaloneMode - cuDLA Standalone Mode](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/cuDLAStandaloneMode)
This sample demonstrates [cuDLA standalone mode](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-model) wherein DLA can be programmed without using CUDA.

# cuDLA samples in this repo

### [cuDLALayerwiseStatsHybrid - cuDLA Layerwise statistics HybridMode](./cuDLALayerwiseStatsHybrid)
This sample is used to provide layerwise statistics to the application in the [cuDLA hybrid mode](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-model) wherein DLA is programmed using CUDA.

### [cuDLALayerwiseStatsStandalone - cuDLA Layerwise Statistics Standalone Mode](./cuDLALayerwiseStatsStandalone)
This sample is used to provide layerwise statistics to the application in [cuDLA standalone mode](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-model) where DLA is programmed without using CUDA.

# YOLOv5 cuDLA sample
The repo [YOLOv5 cuDLA sample](https://github.com/NVIDIA-AI-IOT/cuDLA-samples) demonstrates how to train YOLOv5s with [NVIDIA's pytorch-quantization toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) and deploy the resulting INT8 model on DLA using cuDLA. Also refer to [QDQ Translator](../../tools/qdq-translator/).
