# Deep Learning Accelerator

NVIDIA DLA hardware is a fixed-function accelerator engine targeted for deep learning operations. It’s designed to do full hardware acceleration of convolutional neural networks, supporting various layers such as convolution, deconvolution, fully connected, activation, pooling, batch normalization, and others. NVIDIA’s Orin SoCs feature up to two second-generation DLAs while Xavier SoCs feature up to two first-generation DLAs.

DLA software consists of the DLA compiler and the DLA runtime stack. The offline compiler translates the neural network graph into a DLA loadable binary and can be invoked using NVIDIA TensorRT™, NvMedia-DLA or cuDLA. The runtime stack consists of the DLA firmware, kernel mode driver, and user mode driver.

Reference for more details: [DLA product page](https://developer.nvidia.com/deep-learning-accelerator)


## DLA Reference Models

In this repo, we will cover a few key DLA-related metrics for **standard deep learning model architectures** in the context of common reference application implementations. 

The goal is to provide a reference baseline about network architectures and how they map to DLA as well as the INT8 accuracy of these networks.

**Use case**|**Network**|**INT8 Accuracy on Orin’s DLA**|**Layers always running on GPU**|**Instructions**
:-----:|:-----:|:-----:|:-----:|:-----:
Classification|[ResNet-50](https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/results/Orin_TRT/resnet50/Offline/accuracy)|Top-1 ImageNet 2012*: 76.34%<br>(GPU INT8: 76.42%, [FP32 reference: 76.46%](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection))|Top-K (Last node of the network)|See [ResNet-50](scripts/prepare_models/README.md/#resnet-50) section in `scripts/prepare_models/README.md`
Object Detection|[SSD-ResNet-34](https://github.com/mlcommons/inference_results_v2.0/tree/master/closed/NVIDIA/results/Orin_TRT/ssd-resnet34/Offline/accuracy)|mAP COCO 2017*: 0.21<br>(GPU INT8: 0.21, [FP32 reference 0.20](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection))|NMS (Last node of the network)|See [SSD-ResNet-34](scripts/prepare_models/README.md/#ssd-resnet-34) section in `scripts/prepare_models/README.md`
Object Detection|[SSD-MobileNetV1](https://github.com/mlcommons/inference_results_v2.0/tree/master/closed/NVIDIA/results/Orin_TRT/ssd-mobilenet/Offline/accuracy)|mAP COCO 2017*: 0.23<br>(GPU INT8: 0.23, [FP32 reference: 0.23](https://github.com/mlcommons/inference/blob/f301a4bdbd1d4bf6781a8925a3bd3967f9855458/README.md))|NMS (Last node of the network)|See [SSD-MobileNetV1](scripts/prepare_models/README.md/#ssd-mobilenetv1) section in `scripts/prepare_models/README.md`

**Accuracy measured internally by NVIDIA, there may be slight differences compared to previous MLPerf Inference submissions.*

Key takeaways:
- Networks tend to have common network backbones and some variation at the end. You can run the compute-intensive backbones on DLA and final layers for post-processing on the GPU.
- GPU and DLA are not bitwise accurate. So the difference in the math is expected and the difference would be within a certain acceptable % of the FP32 reference.

More resources:
- MLPerf Inference 2.1 (Orin DLA): https://mlcommons.org/en/inference-edge-21/
- MLPerf Inference 2.0 (Xavier & Orin DLA): https://mlcommons.org/en/inference-edge-20/

## Setup
Install the Python dependencies with:
```bash
python3 -m pip install requirements.txt
```
