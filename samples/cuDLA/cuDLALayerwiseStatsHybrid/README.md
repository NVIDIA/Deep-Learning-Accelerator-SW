# cuDLALayerwiseStatsHybrid - cuDLA Layerwise statistics HybridMode

## Description

This sample is used to provide layerwise statistics to the application in the [cuDLA hybrid mode](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-model) wherein DLA is programmed using CUDA.
* Sample output:
        <pre><code>./cudla/cuDLALayerwiseStatsHybrid model_loadable.bin 1152 1 hybrid1
        cat hybrid11.csv
        layer_name,start_ms,end_ms
        'Conv_219 + BatchNormalization_220+Relu_221+MaxPool_222',0.004556,0.708967
        'MaxPool_222',0.709676,1.290478
        'Conv_223 + BatchNormalization_224+Relu_225',1.291201,1.692339
        'Conv_226 + BatchNormalization_227+Add_228+Relu_229',1.693057,2.109201
        'Conv_230 + BatchNormalization_231+Relu_232',2.109914,2.513233
        'Conv_233 + BatchNormalization_234+Add_235+Relu_236',2.513964,2.930510
        'Conv_237 + BatchNormalization_238+Relu_239',2.931229,3.328956
        'Conv_240 + BatchNormalization_241+Add_242+Relu_243',3.329661,3.745779
        'Conv_244 + BatchNormalization_245+Relu_246',3.746490,4.018651</pre></code>
## Key Concepts

cuDLA, Data Parallel Algorithms, Image Processing

## Supported SM Architectures

[SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, QNX

 ## Minimum system config required by layerwise profiling sample
| Hardware platform | OS | DLA SW version |
| -------- | -------- | -------- |
| DRIVE Orin (Automotive) | DRIVE OS 6.0.8.0 (not supported QNX Safety)| DLA 3.14.0 |
| Jetson Orin (Embedded) | JetPack 6.0 (unreleased) | DLA 3.14.0 |


## Supported CPU Architecture

aarch64

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaStreamCreateWithFlags, cudaStreamDestroy, cudaFree, cudaGetErrorName, cudaSetDevice, cudaStreamSynchronize, cudaMalloc, cudaMemsetAsync, cudaMemcpyAsync
## Prerequisites

Download and install the [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are aarch64.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=aarch64` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
    ```
    $ make HOST_COMPILER=g++
    ```
* **Building the DLA Loadable**
     1) Flash the board with DriveOS 6.0.8.0 PDK.
     2) On the target ``trtexec`` bin can be found in ``/usr/src/tensorrt/bin/`` path.
     3) DLA loadable can be generated with *trtexec* bin with the command
        ``` ./trtexec <Model options> --saveEngine=<Name for output DLA file> --useDLACore=<Index of DLA core> --inputIOFormats=<Input format> --outputIOFormats=<Output format> --buildDLAStandalone ```
   4) Run ``` ./trtexec --help ``` command to find out more info on all the flags supported by *trtexec*  Eg: Model options, Build options, Inference options etc.  One such important flag is *--profilingVerbosity* (in the current TRT version (8.6.11) this flag isn't supported and the flag is always set to *layer_names_only* which cannot be modified. *detailed* and *none* options of the flag will be supported in the future TRT versions). This flag allows the users to specify:
         * layer_names_only : Print only the layer names. This is the default setting.
         * detailed : Print detailed layer information including layer names and layer parameters.
         * none : Do not print any layer information.
  5) Refer [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec) for more info.
*  **COMMAND TO RUN THE SAMPLE**
       ./cuDLALayerwiseStatsHybrid <loadable> <freqMHZ> <statSupport> <filenamePrefix>
* **Input args to run the sample**
   1) **Loadable** - Give path to the DLA loadable file (can be generated by following                          `Building the DLA Loadable` section above)
        2) **Setting DLA Clk Freq** - This is an optional step, if the platform where the test runs has the following option(s) to execute, then perform these steps. Else, just read the clocks and use the same clock while running the sample
DLA clock frequency can be queried using the command:
               <pre><code>sudo cat /sys/kernel/debug/bpmp/debug/clk/nafll_dla0_core/rate</pre></code> The DLA_CLK can also be set with a desired freq value using <pre><code>echo ${DLA_CLK} > /sys/kernel/debug/bpmp/debug/clk/nafll_dla0_core/rate</pre></code> Make sure the input param freqMHZ matches the DLA clock value otherwise the #cycle → wall time translation will be off.
               **Note**: For `D6Q-standard` platform the above path for `rate` would be `/dev/nvbpmpdebugfs/bpmp_debug/clk/nafll_dla0_core/rate`
    3) **statSupport** - This param needs to be set to '1' when a user wants to request for 'Layerwise profiling'.
    4) **filename prefix** - The layerwise statistics values are written to output file(s), these output csv file(s) take this \<filename prefix> parameter as the prefix for the name. A typical output csv file looks like ``<filename prefix><task statistics index>.csv``

* **Working example to build a loadable with RetinaNet-ResNet-34 models:**
    1) Download the ONNX network model files present in [Untrained ResNet-34 with and without RetinaNet head](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/tree/main/scripts/prepare_models#retinanet-resnet-34))
     2) Run the following command
                        <pre><code>./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=backbone_rn34_1280x768_dummy.onnx --saveEngine=model_loadable.bin --buildDLAStandalone</code></pre>
        * Sample output for a successful build:
          <pre><code>[07/24/2023-20:00:01] [I] Engine built in 28.6839 sec.
          [07/24/2023-20:00:01] [I] Skipped inference phase since --skipInference is added.
          &&&& PASSED TensorRT.trtexec [TensorRT v8611] # ./trtexec --onnx=backbone_rn34_1280x768_dummy.onnx --saveEngine=model_loadable.bin --useDLACore=0 --fp16 --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --skipInference --buildDLAStandalone</pre></code>
  3) Loadable will be generated with the name *model_loadable.bin* which can be used to run the sample.
  4) Setting the DLA_CLK [Optional]:
      `DLA_CLK = 1152000000`
      <pre><code>echo ${DLA_CLK} > /sys/kernel/debug/bpmp/debug/clk/nafll_dla0_core/rate</pre></code>
		Refer `Setting DLA Clk Freq` section for more detail.
        **Note:** Here we are setting the value in `Hz`, while passing this parameter to run cuDLA layerwise sample the value is passed in `MHz`.
  5) Run the sample<pre><code>./cuDLALayerwiseStatsHybrid model_loadable.bin 1152 1 hybrid1</pre></code> Layerwise statistics profiling data will be populated in the file with name *hybrid11.csv*

## Interpretation of results
With the latest available TRT version (8.6.11) only layer_name, start_ms and end_ms columns are supported and populated in the output file(s). In future versions of TRT, more detailed profiling data will be available in the output file(s).

__layer_name__: Layer name after all fusions inside the DLA Compiler.

__start_ms__: The beginning of the fused layers to the left in milliseconds.

__end_ms__: The end of the fused layers to the left in milliseconds.
* Note that the execution of a layer can overlap with the execution of other layers - this overlap can for example be observed in models with several independent branches.

*__latency_ms__: This section is not existing in the original CSV output. This can be added by post-processing actions – the wall clock latency of beginning to completing of the fused layers.*

__net_duration_cpipes_ms__, __net_duration_rubik_ms__, __net_duration_bdma_ms__: These 3 columns list the total latency on Orin DLA's 3 fundamental hardware engines in milliseconds (they can run in parallel):
1) CPIPEs or Convolution Pipes (there are 2 of them on Orin) – used for INT8 Convolutions, INT8 & FP16 Pooling ops, INT8 & FP16 LRN ops, and all kinds of INT8 & FP16 point-wise ops such as Scale, Bias, Activations, precision conversion.
2) Rubik – used for FP16 Convolutions, all kinds of input/output format conversions ops from/to planar format, for Reshape+Slice ops involving the channel dimension, for Transpose ops, for Resize ops.
3) BDMA – used for data copies, for instance triggered by Reshape, Concat, Slice, Transpose ops.

__src_precisions__: The input tensor precisions to the fused layers. Note that in this release, all data transfers with the BDMA engine and non-Conv Rubik ops report "BYTE" as input, compute & output precisions.

__compute_precisions__: The internal compute precisions of the fused layers. Note that a layer can consist of several internal nodes and engines, and that the internal precision of the CPIPEs is FP19. Note that in this release, all data transfers with the BDMA engine and non-Conv Rubik ops report "BYTE" as input, compute & output precisions.

__dst_precisions__: The output tensor precisions from the fused layers. Note that in this release, all data transfers with the BDMA engine and non-Conv Rubik ops report "BYTE" as input, compute & output precisions.

__uses_sram__: Shows whether internal SRAM (also called UBUF on Orin's DLA) is used for these fused layers.

__uses_fp16_conv__: Shows whether the FP16 Convolution engine is used for these fused layers.

__uses_int8_conv__: Shows whether the INT8 Convolution engine is used for these fused layers.

__max_num_cpipes__: Shows the maximum number of CPIPEs used for these layers. There are 2 CPIPEs available on Orin.

__uses_weight_compression__: Shows whether there is weight compression used in any INT8 Convolutions for these fused layers.

__uses_structured_sparsity__: Shows whether there is structured sparsity used in any INT8 Convolutions for these fused layers.

__uses_sdp__: Shows whether there are point-wise ops used for these fused layers. SDP = Single Data Processor part of the CPIPE engines.

__uses_pool__: Shows whether there are pooling ops used for these fused layers. Pooling is performed on the CPIPE engines, and is part of the SDP engine.

__uses_square_sum_unit__: Shows whether there are Square Sum Unit ops used for these fused layers (used for LRN for instance). The Square Sum Unit is part of the CPIPE engines.
