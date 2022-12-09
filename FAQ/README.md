# Frequently Asked Questions about DLA

## Table of contents
- [Which NVIDIA Jetson or DRIVE Modules have DLA?](#which-nvidia-jetson-modules-have-dla-)
- [What is DLA Core and what is DLA Falcon?](#what-is-dla-core-and-what-is-dla-falcon-)
- [Are there specific networks or a list of networks recommended for DLA?](#are-there-specific-networks-or-a-list-of-networks-recommended-for-dla-)
- [Does DLA support Point-Pillar Networks?](#does-dla-support-point-pillar-networks-)
- [What kind of overhead exists when you deploy a network with layers alternating between GPU and DLA?](#what-kind-of-overhead-exists-when-you-deploy-a-network-with-layers-alternating-between-gpu-and-dla-)
- [Why is the latency higher when running workloads on two DLA cores and GPU?](#why-is-the-latency-higher-when-running-workloads-on-two-dla-cores-and-gpu-)
- [What precision formats are supported on DLA?](#what-precision-formats-are-supported-on-dla-)
- [How does FP16 performance compare to int8?](#how-does-fp16-performance-compare-to-int8-)
- [How do you quantize the network to INT8 for DLA?](#how-do-you-quantize-the-network-to-int8-for-dla-)
- [Is a calibration file always necessary to convert a model to int8?](#is-a-calibration-file-always-necessary-to-convert-a-model-to-int8-)
- [Is ONNX the recommended way to go from PyTorch to TensorRT?](#is-onnx-the-recommended-way-to-go-from-pytorch-to-tensorrt-)
- [What's the difference between Dense TOPs and Sparse TOPS?](#what-s-the-difference-between-dense-tops-and-sparse-tops-)
- [Can we scale the clock frequency on DLA?](#can-we-scale-the-clock-frequency-on-dla-)
- [Can the DLA directly access the data in the DRAM system memory?](#can-the-dla-directly-access-the-data-in-the-dram-system-memory-)
- [How much internal SRAM does the DLA have?](#how-much-internal-sram-does-the-dla-have-)
- [How does the inference latency on DLA compare to that of GPU?](#how-does-the-inference-latency-on-dla-compare-to-that-of-gpu-)
- [Is it possible to run multiple networks concurrently on two DLA cores?](#is-it-possible-to-run-multiple-networks-concurrently-on-two-dla-cores-)
- [Can multiple models run on a single DLA?](#can-multiple-models-run-on-a-single-dla-)
- [What tools and utilities are available for profiling and debugging DLA workloads?](#what-tools-and-utilities-are-available-for-profiling-and-debugging-dla-workloads-)
- [Will DLA help in reducing power consumption?](#will-dla-help-in-reducing-power-consumption-)
- [Where can we learn more about how DLA is leveraged in ISAAC reference applications?](#where-can-we-learn-more-about-how-dla-is-leveraged-in-isaac-reference-applications-)





### Which NVIDIA Jetson or DRIVE Modules have DLA?
All Jetson AGX Orin & Orin NX boards, and all prior generation Jetson AGX Xavier & Xavier NX modules have DLA Cores. [This](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency) is a very handy reference for all platforms with at least one DLA instance and their corresponding clock settings. DRIVE Xavier and DRIVE Orin also have DLA Cores.

### What is DLA Core and what is DLA Falcon?
The DLA Core is responsible for performing all math operations, while Falcon is the microcontroller orchestrating/scheduling workloads on the DLA Core.

### Are there specific networks or a list of networks recommended for DLA?
You can run any network on DLA, where the [supported layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla_layers) run on DLA and the unsupported layers [fallback to GPU](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#gpu_fallback).
If you want an entire model running on DLA, DLA supports backbones such as EfficientNet-Lite, EfficientNet-EdgeTPU, several MobileNet versions such as MobileNet-EdgeTPU, models from the Inception family (such as GoogLeNet or InceptionV3), ResNets, YOLO models. For example, you can attach Segmentation heads (such as FCN), object detection heads (such as SSD) or classification heads (such as AveragePool + FC/1x1 convolution) to one of the above backbones for your specific use case. 
Models with high arithmetic intensity (that means: many math operations per memory operation) are best suited for DLA if you are interested in high utilization. Examples for this are ResNet-18, ResNet-34 or ResNet-50. But also models with lower arithmetic intensity such as MobileNet have low latency on DLA.

### Does DLA support Point-Pillar Networks?
DLA supports Convolution and Transposed Convolution. With a few tweaks in the parameters (for example, use Transposed Convolution with stride==kernel size or switch to Nearest-Neighbor Resize), it should be possible to support Point Pillar Networks. You can accomplish this by modifying the ONNX graph or original model definition.

Re-expressing operations not currently supported on DLA in terms of [supported operations](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#dla_layers) is a common technique that you can use to run your entire network on DLA.

### What kind of overhead exists when you deploy a network with layers alternating between GPU and DLA?
There can be overhead due to couple of common reasons:
Reformatting of the tensors between GPU and DLA memory formats as of today. The kCHW16 (FP16), and kCHW32 (INT8) formats are common on DLA & GPU; however, the default formats are not the same. So ensure that you use commonly supported tensor formats to avoid the reformatting overhead. See [here](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#restrictions-with-dla) for more details.
Overhead due to small workloads. Ensure you have a network where subgraphs (a contiguous portion of the network graph) maps to DLA rather than individual layers going back and forth between GPU and DLA.


### Why is the latency higher when running workloads on two DLA cores and GPU?
There can be several reasons for this, let’s focus on the three most common ones: 

1) DLA and GPU both consume the same resource: System DRAM. The more bandwidth-bound a workload is, the higher the chances that both DLA and GPU will become bottlenecked for memory access when running in parallel.
2) If you do not run DLA with native DLA formats through TensorRT, there will be GPU reformat kernels inserted around every DLA inference. This means you have added additional GPU workload (since the GPU is busy with reformatting). To reduce this, you can use the [reformat-free I/O options](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors) for DLA.
3) Interference due to inefficient scheduling can also be a reason: Especially at low batch sizes and in those scenarios when input frames arrive one after another at a fixed rate, the DLA task does not get scheduled quickly or does not register the completion signal back to the GPU quickly enough. This happens both for single-context cases as well as multi-contexts cases. Known options for reducing this type of interference:
Using [cuDLA for DLA inference](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41516/): See [here](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/cuDLAStandaloneMode) for a sample in standalone mode (without TensorRT – requires extracting a DLA loadable from a TensorRT engine). Starting from TensorRT 8.5, DLA inference through the TensorRT interface uses cuDLA by default.
Increasing [the preferred number of compute and copy engine concurrent connections](https://docs.nvidia.com/deploy/mps/index.html#topic_5_2_4) (work queues): Can be achieved by setting the Linux environment variable `CUDA_DEVICE_MAX_CONNECTIONS=32`.

### What precision formats are supported on DLA?
DLA on Orin and Xavier supports the optimal inference precision formats - FP16 and INT8.
DLA on Orin specifically has been optimized for INT8 given the optimality of this precision for AI inference by trading off FP16 performance when compared to DLA on Xavier. 
The option of FP16 and INT8 hybrid precision in the same model allows you to find the sweet spot between accuracy and low resource consumption.

### How does FP16 performance compare to int8?
NVIDIA has designed its Deep Learning Accelerator with an emphasis on INT8 for AI inferencing since inference is the key value proposition of the Jetson or DRIVE modules. Training happens on bigger NVIDIA GPUs and systems. 
Orin DLA was designed for 9x increase in INT8 compute and higher power efficiency in exchange for lower FP16 convolution compute, compared to previous generation Xavier architecture. DLA is designed for well-understood AI inference models and running at a lower power and lower area overhead. As a result, it offers a highly optimized INT8 DL inference engine.

### How do you quantize the network to INT8 for DLA?
To quantize a network for DLA, you need to know the dynamic range of the intermediate tensors to help map FP32/FP16 (wide representation) to INT8 (narrow representation). For DLA, this can be achieved through TensorRT’s Post-Training Quantization (PTQ) toolchain.
The TensorRT PTQ workflow utilizes a representative calibration dataset after a model has been trained. This provides the scaling factors for the output and input tensors of layers that you want to run in int8. If you do not have scaling factors, those layers are run in fp16.
Here is a sample you can reference to get started: https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleINT8 
Note the presence of the `--useDLACore=N` option.

### Is a calibration file always necessary to convert a model to int8?
You can either use a TensorRT calibration file or use the [ITensor TensorRT APIs](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_tensor.html#a04d2107c0a3a5058d485863339f45872) to set a network tensor’s scaling factors. If you do not have scaling factors for individual tensors, you can also run the affected layers in fp16 precision.

How would I go about using DLA without TensorRT?
TensorRT Builder is the only user interface to invoke the DLA compiler and it provides a consistent interface to parse your network and build an inference engine both for GPU as well as DLA (DLA loadable). 
TensorRT Runtime provides the runtime APIs to invoke the DLA Runtime stack with little user intervention but for advanced users, you can also cuDLA runtime APIs to run your DLA loadable on DLA HW.

### Is ONNX the recommended way to go from PyTorch to TensorRT?
DLA depends on TensorRT for parsing the network before the DLA compiler compiles it into a loadable. ONNX is the preferred TensorRT path to go from various frameworks including Pytorch to TRT as of today. However, you can also use [Torch-TensorRT](https://github.com/pytorch/TensorRT/blob/d9baa3668a9f2c4a1447c53c3a686e1b316613b8/docs/v0.3.0/_sources/tutorials/using_dla.rst.txt) with DLA for example as well as building your own translator using TensorRT’s API.

### What's the difference between Dense TOPs and Sparse TOPS?
Structured Sparsity is an optimization to leverage a certain pattern in the sparse computation within a network because of zero weights. Structured sparsity is a new feature of the Ampere GPU and the DLAs in Orin. It allows you to increase your compute throughput by providing convolution weights with 2:4 “sparse” weight patterns. This means that for a KCRS convolution weight blob, for every 4 weights along the C dimension, at least 2 are zeros. For DLA, you will need C > 64 in order to enable structured sparsity. Refer to this [detailed article on how structured sparsity is leveraged](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) both in HW and SW on NVIDIA platforms.

### Can we scale the clock frequency on DLA?
Yes, you can set the clock frequency on DLA. The [reference here](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#supported-modes-and-power-efficiency) walks you through the steps for setting power & clock profiles on Jetson boards. For Automotive platforms, it is typically required to change the clocks at flashing/OS setup time.

### Can the DLA directly access the data in the DRAM system memory?
Yes, it is connected to system DRAM, just like GPU or CPU.

### How much internal SRAM does the DLA have? 
On Xavier, 2 DLAs and 2 Programmable Vision Accelerators (PVAs) share a total of 4 Megabyte of SRAM. On Orin, each DLA has exclusive access to 1 Megabyte of internal SRAM.

### How does the inference latency on DLA compare to that of GPU?
Latency for a single workload on one DLA will be higher when compared to GPU because each DLA instance has lower theoretical math throughput (TOPs) than the GPU. There are 2 DLAs on Xavier and Orin, and at default clocks, each DLA instance has lower theoretical math throughput (TOPs) than the GPU. 

However, when you look at it from an application perspective, you can reduce your total or overall latency by distributing the deep learning and non-deep learning workloads across DLA & GPU. For certain applications where there are requirements on consistency of workload latency, DLA is especially suited.

If you want to evaluate device performance for a certain workload, we recommend starting with the calculation of math utilization for GPU and DLA. The math utilization is calculated by dividing the achieved throughput (number of images per second times FLOPs per image) by the theoretical peak throughput of the accelerator at the used precision (number of operations per second, often indicated in Tera operations per seconds or TOPs).


### Is it possible to run multiple networks concurrently on two DLA cores?
Yes, it is possible to run multiple networks on both DLAs concurrently (so model A runs on the first instance, and model B runs on the second instance). You can do the same with two instances of the same network. In fact, you can concurrently run multiple networks on GPU and the DLA cores.
One way to quickly get started is by running two concurrent trtexec processes, where the first one is run with `--useDLACore=0`, and the second one with `--useDLACore=1`. If you want to use 2 DLA Cores in the same process, use [this TensorRT API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_builder_config.html#a9bf5696456d7c54203075006ac5fc9f5). 

### Can multiple models run on a single DLA?
Yes, you can run multiple models on a single DLA core in sequential order.

### What tools and utilities are available for profiling and debugging DLA workloads?
The easiest way to measure DLA performance is using [trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec). For example, you can run a ResNet-50 network on DLA using trtexec in INT8 mode with this command:
```sh
./trtexec --onnx=rn50.onnx --int8 --useDLACore=0 --dumpProfile --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 
```
Add `--allowGPUFallback` if model cannot fully run on DLA

Nsight System is the best tool to profile your application not just for DLA but for the entire SoC. You can get a detailed profile of DLA runtime of each subgraph and which core is used during runtime and we continue to add more profiling functionality into Nsight systems. See [here](https://developer.nvidia.com/docs/drive/drive-os/latest/tensorrt/developer-guide/index.html#dla-profiling) for more details.
For (layer-wise) debugging for accuracy, it is recommended to use tools like [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/master/tools/Polygraphy).

### Will DLA help in reducing power consumption?
Yes, DLA is optimized for high performance per Watt. If you want to reduce the power consumption of your Jetson or DRIVE Module, the best approach is to map as much of your deep learning workload to the DLA as possible.

### Where can we learn more about how DLA is leveraged in ISAAC reference applications?
ISAAC SDK has a reference application for proximity segmentation using stereo data. The application has a requirement of having two independent paths and the team designed the independent pipelines to leverage both GPU and DLA optimally. You can find the code and their network architecture in the link here: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation
