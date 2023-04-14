# Deep Learning Accelerator

NVIDIA DLA hardware is a fixed-function accelerator engine targeted for deep learning operations. It’s designed to do full hardware acceleration of convolutional neural networks, supporting various layers such as convolution, deconvolution, fully connected, activation, pooling, batch normalization, and others. NVIDIA’s Orin SoCs feature up to two second-generation DLAs while Xavier SoCs feature up to two first-generation DLAs.

DLA software consists of the DLA compiler and the DLA runtime stack. The offline compiler translates the neural network graph into a DLA loadable binary and can be invoked using NVIDIA TensorRT™, NvMedia-DLA or cuDLA. The runtime stack consists of the DLA firmware, kernel mode driver, and user mode driver.

Reference for more details: [DLA product page](https://developer.nvidia.com/deep-learning-accelerator)

## Why is DLA essential on Orin?

* DLA peak performance contributes between 38% and 74% to Orin's total Deep Learning (DL) performance (depending on the power mode, see table below)
* DLA is on average 3x to 5x more power-efficient than the GPU (depending on the power mode and the workload, see [DLA Performance per Watt (Power Efficiency)](#dla-performance-per-watt-power-efficiency) for details)

Here the distribution of DL TOPs between GPU and DLA on a Jetson AGX Orin 64GB, depending on the [power mode](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency):

|                                 | Power mode: MAXN | Power mode: 50W | Power mode: 30W | Power mode: 15W |
| ------------------------------- | ---------------- | --------------- | --------------- | --------------- |
| GPU sparse INT8 peak DL performance     | 171 TOPs         | 109 TOPs        | 41 TOPs         | 14 TOPs         |
| **2x DLA sparse INT8 peak performance**  | **105 TOPs**         | **92 TOPs**         | **90 TOPs**         | **40 TOPs**         |
| Total Orin peak INT8 DL performance | 275 TOPs         | 200 TOPs        | 131 TOPs        | 54 TOPs         |
| **Percentage: DLA peak INT8 performance of total Orin peak DL INT8 performance**         | **38%**              | **46%**             | **69%**             | **74%**             |

Note: 
- The DLA TOPs of the 30W & 50W power modes on Jetson AGX Orin 64GB are comparable to the maximum clocks on DRIVE Orin platforms for Automotive
- The maximum DLA TOPs on Jetson Orin NX 16GB are comparable to the 15W power mode on Jetson AGX Orin 64GB

## DLA Reference Models

In this repo, we will cover a few key DLA-related metrics for **standard deep learning model architectures** in the context of common reference application implementations. 

The goal is to provide a reference baseline about network architectures and how they map to DLA as well as the INT8 accuracy of these networks.
### Accuracy on DLA and GPU
**Use case**|**Network**|**INT8 Accuracy on Orin’s DLA**|**Layers always running on GPU**|**Instructions**
:-----:|:-----:|:-----:|:-----:|:-----:
Object Detection|[RetinaNet ResNeXt-50](https://github.com/mlcommons/inference_results_v3.0/blob/d9f23eb124be29a02833f55f6d518e78e8f6433d/closed/NVIDIA/results/Orin_TRT/retinanet/Offline/accuracy/accuracy.txt)|mAP OpenImages MLPerf validation set*: 0.3741<br>(GPU INT8: 0.3740, [FP32 reference 0.3757](https://github.com/mlcommons/inference/blob/bc2a5450f55ef923a3669420292c442ef46767f2/vision/classification_and_detection/README.md))|NMS (Last node of the network)|See [RetinaNet ResNeXt-50](scripts/prepare_models/README.md/#retinanet-resnext-50) section in `scripts/prepare_models/README.md`
Classification|[ResNet-50](https://github.com/mlcommons/inference_results_v2.1/blob/502ce1f6b71d77e07d3a12838cd55e0b0497b53d/closed/NVIDIA/results/Orin_TRT/resnet50/Offline/accuracy/accuracy.txt)|Top-1 ImageNet 2012*: 76.34%<br>(GPU INT8: 76.42%, [FP32 reference: 76.46%](https://github.com/mlcommons/inference/blob/bc2a5450f55ef923a3669420292c442ef46767f2/vision/classification_and_detection/README.md))|Top-K (Last node of the network)|See [ResNet-50](scripts/prepare_models/README.md/#resnet-50) section in `scripts/prepare_models/README.md`
Object Detection|[SSD-ResNet-34](https://github.com/mlcommons/inference_results_v2.0/blob/a1f0ac83143cf67be5ffbc75c930369eebd665b0/closed/NVIDIA/results/Orin_TRT/ssd-resnet34/Offline/accuracy/accuracy.txt)|mAP COCO 2017*: 0.21<br>(GPU INT8: 0.21, [FP32 reference 0.20](https://github.com/mlcommons/inference/blob/bc2a5450f55ef923a3669420292c442ef46767f2/vision/classification_and_detection/README.md))|NMS (Last node of the network)|See [SSD-ResNet-34](scripts/prepare_models/README.md/#ssd-resnet-34) section in `scripts/prepare_models/README.md`
Object Detection|[SSD-MobileNetV1](https://github.com/mlcommons/inference_results_v2.0/blob/a1f0ac83143cf67be5ffbc75c930369eebd665b0/closed/NVIDIA/results/Orin_TRT/ssd-mobilenet/Offline/accuracy/accuracy.txt)|mAP COCO 2017*: 0.23<br>(GPU INT8: 0.23, [FP32 reference: 0.23](https://github.com/mlcommons/inference/blob/bc2a5450f55ef923a3669420292c442ef46767f2/vision/classification_and_detection/README.md))|NMS (Last node of the network)|See [SSD-MobileNetV1](scripts/prepare_models/README.md/#ssd-mobilenetv1) section in `scripts/prepare_models/README.md`

**Accuracy measured internally by NVIDIA, there may be slight differences compared to previous MLPerf Inference submissions.*

Key takeaways:
- Networks tend to have common network backbones and some variation at the end. You can run the compute-intensive backbones on DLA and final layers for post-processing on the GPU.
- GPU and DLA do not produce bitwise identical results. So the difference in the math is expected and the difference would be within a certain acceptable % of the FP32 reference.

More resources:
- [MLPerf Inference 3.0 (Orin DLA)](https://developer.nvidia.com/blog/setting-new-records-in-mlperf-inference-v3-0-with-full-stack-optimizations-for-ai/)
- [MLPerf Inference 2.1 (Orin DLA)](https://developer.nvidia.com/blog/full-stack-innovation-fuels-highest-mlperf-inference-2-1-results-for-nvidia/)
- [MLPerf Inference 2.0 (Xavier & Orin DLA)](https://developer.nvidia.com/blog/getting-the-best-performance-on-mlperf-inference-2-0/)

### Structured Sparsity Case Study: Object Detection Accuracy with RetinaNet ResNet-34
DLA on the NVIDIA Orin platform supports Structured Sparsity that offers the opportunity to minimize latency and maximize throughput in production. See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla-sparsity) for details (note that the listed restrictions may not apply anymore to most recent DLA releases).

Below case study presents that training models for Structured Sparsity is expected to maintain accuracy:

|**Network**|**Weight mask pattern**|**Waymo Open Dataset Test Accuracy – 2D Detection 2020<br>IoU=0.50:0.95 (all sizes) @ FP16 on Host**|**Steps involved**
:-----:|:-----:|:-----:|:-----:
[RetinaNet ResNet-34](scripts/prepare_models/README.md/#retinanet-resnet-34)|No mask / Dense|44.7|Trained from scratch for 26 epochs.
[RetinaNet ResNet-34](scripts/prepare_models/README.md/#retinanet-resnet-34)|2:4 Sparse|44.8|Sparsified dense model as detailed in [sparsity whitepaper](https://arxiv.org/abs/2104.08378), then trained for another 26 epochs.


## Orin DLA Performance

### DLA Dense performance
RetinaNet ResNet-34
2x DLA images per second on a Jetson AGX Orin 64GB in dense operation measured with JetPack 5.1.1, depending on the [power mode](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency):

| Network                              | Power mode: MAXN | Power mode: 50W | Power mode: 30W | Power mode: 15W |
| ------------------------------------ | ---------------- | --------------- | --------------- | --------------- |
| [RetinaNet ResNeXt-50](scripts/prepare_models/README.md/#retinanet-resnext-50) (800x800, bs=1) | 78               | 72              | 71              | 36              |
| [ResNet-34 backbone](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1)  | 285              | 260             | 255             | 121             |
| [RetinaNet ResNet-34](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1) | 108              | 98              | 96              | 45              |
| [SSD-ResNet-34](scripts/prepare_models/README.md/#ssd-resnet-34) (1200x1200, bs=1)      | 83               | 76              | 74              | 36              |
| [ResNet-50](scripts/prepare_models/README.md/#resnet-50) (224x224, bs=2)            | 2037             | 1948            | 1928            | 1072            |
| [SSD-MobileNetV1](scripts/prepare_models/README.md/#ssd-mobilenetv1) (300x300, bs=2)      | 2664             | 2506            | 2472            | 1313            |

Key takeaways:
- DLA's peak performance for these models in MAXN power mode is close to the achieved performance with 50W and 30W power modes - even at a reduced power budget, your DLA-based inference pipeline can sustain high throughput.
- Attaching a RetinaNet head to a ResNet-34 backbone adds a considerable number of theoretical MAC operations per inference, and is hence expected to result in lower end-to-end throughput. Choose your task-specific head that you attach to backbones carefully.
- Even at the lowest power mode, all of the benchmarks with high input resolution and batch size 1 can still sustain 30 images per second on 2 DLAs combined.  

### DLA Sparse performance

Generally, Structured Sparsity shows perf improvements over dense operation for dense Convolution layers that are already math-bound. The more math-bound a layer in dense operation, the higher the expected dense->sparse speedup after applying a 2:4 sparsity pattern.

2x DLA images per second on a Jetson AGX Orin 64GB in sparse operation measured with JetPack 5.1.1, depending on the [power mode](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency) (with dense->sparse speedup):

| Network                              | Power mode: MAXN | Power mode: 50W | Power mode: 30W | Power mode: 15W |
| ------------------------------------ | ---------------- | --------------- | --------------- | --------------- |
| [RetinaNet ResNeXt-50](scripts/prepare_models/README.md/#retinanet-resnext-50) (800x800, bs=1) | 102 (1.31x)      | 96 (1.34x)      | 95 (1.34x)      | 51 (1.43x)      |
| [ResNet-34 backbone](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1)  | 384 (1.35x)      | 360 (1.38x)     | 354 (1.39x)     | 176 (1.45x)     |
| [RetinaNet ResNet-34](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1) | 143 (1.32x)      | 133 (1.36x)     | 131 (1.37x)     | 66 (1.47x)      |
| [SSD-ResNet-34](scripts/prepare_models/README.md/#ssd-resnet-34) (1200x1200, bs=1)     | 103 (1.24x)      | 97 (1.28x)      | 95 (1.28x)      | 49 (1.36x)      |

*Just add `--sparsity=force` to the `trtexec` commands from `scripts/prepare_models/README.md` to reproduce.*

Key takeaways:
- Decreasing the power budget on Jetson Orin generally implies reducing the ratio of DLA TOPs to DRAM bandwidth (the hardware's op per Byte ratio). Lowering the hardware's op per Byte ratio means that more layers get the chance to become math-bound, and hence we can also observe higher dense->sparse speedups with lower power modes.


### DLA Performance per Watt (Power Efficiency)
DLA & GPU form your perfect team for Deep Learning inference on the SoC. While the GPU delivers the most TOPs in high-power profiles, DLA excels at power efficiency.

Below table shows the Perf/W ratio of DLA relative to the GPU (accelerator power only, perf metric: images per second) on a Jetson AGX Orin 64GB measured with JetPack 5.1.1, depending on the [power mode](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency):

| Network                                         | Weight mask mode                                                                  | Power mode: MAXN | Power mode: 50W | Power mode: 30W | Power mode: 15W |
| ----------------------------------------------- | --------------------------------------------------------------------- | ---- |---- |---- |---- |
| [RetinaNet ResNeXt-50](scripts/prepare_models/README.md/#retinanet-resnext-50) (800x800, bs=1)            | Dense                                                                 | 3.8x | 3.1x | 2.8x | 4.2x |
| [RetinaNet ResNeXt-50](scripts/prepare_models/README.md/#retinanet-resnext-50) (800x800, bs=1)            | Sparse                                                                | 4.7x | 4.1x | 3.7x | 5.6x |
| [ResNet-34 backbone](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1)             | Dense                                                                 | 4.3x | 3.7x | 3.4x | 4.9x |
| [ResNet-34 backbone](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1)             | Sparse                                                                | 3.4x | 3.0x | 2.7x | 4.1x |
| [RetinaNet ResNet-34](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1)            | Dense                                                                 | 4.2x | 3.5x | 3.1x | 4.6x |
| [RetinaNet ResNet-34](scripts/prepare_models/README.md/#retinanet-resnet-34) (1280x768, bs=1)            | Sparse                                                                | 3.5x | 2.9x | 2.6x | 3.9x |
| [SSD-ResNet-34](scripts/prepare_models/README.md/#ssd-resnet-34) (1200x1200, bs=1)                 | Dense                                                                 | 3.7x | 3.0x | 2.7x | 4.1x |
| [SSD-ResNet-34](scripts/prepare_models/README.md/#ssd-resnet-34) (1200x1200, bs=1)                 | Sparse                                                                | 3.5x | 3.3x | 2.9x | 4.3x |
| [ResNet-50](scripts/prepare_models/README.md/#resnet-50) (224x224, DLA bs=2, GPU bs=256)       | Dense                                                                 | 3.5x | 2.9x | 2.6x | 3.9x |
| [SSD-MobileNetV1](scripts/prepare_models/README.md/#ssd-mobilenetv1) (300x300, DLA bs=2, GPU bs=128) | Dense                                                                 | 4.2x | 3.6x | 3.2x | 5.0x |

Key takeaways:
- DLA is about 3x to 5x more power efficient than GPU for these benchmarks
- Enabling Structured Sparsity generally improves DLA's power efficiency
- At the lowest power mode of 15W, DLA's power efficiency is the highest (where 74% total Orin peak DL INT8 performance comes from the DLAs)

## ONNX operators supported on DLA

See [operators/README.md](operators/README.md) for details on ONNX operators already supported on DLA and planned to be supported in future releases.


## ONNX operators supported on DLA

See [operators/README.md](operators/README.md) for details on ONNX operators already supported on DLA and planned to be supported in future releases.


## Setup
Install the Python dependencies with (only supported on x86 hosts):
```bash
python3 -m pip install requirements.txt
```
