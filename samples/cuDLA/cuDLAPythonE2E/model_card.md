# ResNet-50-v1-12 Overview

## Description:  
ResNet-50-v1-12 classifies an input image into an ImageNet object category using a 50-layer residual convolutional network. The sample does not redistribute the model or compiled artifacts. Users download the original from the ONNX Model Zoo, then run `model_prep/prepare_resnet50_dla.py` to split the graph into a GPU stem, a Deep Learning Accelerator backbone, and a GPU head. Preparation converts the backbone tensors and weights to FP16 while preserving the GPU stem/head in FP32. TensorRT `trtexec` compiles the backbone on the target Jetson to a standalone FP16 CHW16 loadable so the sample can run the heavy layers on the Deep Learning Accelerator from Python.
ResNet-50-v1-12 was developed by Microsoft Research as a part of ResNet.  
_This model is ready for commercial or non-commercial use._  

## Third-Party Community Consideration  
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [Microsoft Research Model Card](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet).  

### License/Terms of Use:
GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The sample code is also licensed under [Apache-2.0](LICENSE). Notices for the bundled images and `assets/synset.txt` are preserved in this sample's [LICENSE](LICENSE).

### Deployment Geography:  
Global

### Use Case:  
Jetson developers and embedded computer-vision engineers may use this model as a standard ImageNet image classifier. In this release, it is run locally on Jetson AGX Orin to identify the main object in an image (for example, a dog breed) while the heavy layers execute on the Deep Learning Accelerator. Typical applications are on-device classification demos, cuDLA integration samples, and as a baseline for other vision workloads on Jetson.  
  

### [Expected] Release Date:
GitHub: 10/08/2026 via https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/tree/main/samples/cuDLA/cuDLAPythonE2E

  

## Reference(s):
[ResNet-50-v1-12 from the ONNX Model Zoo](https://github.com/onnx/models)  
[ONNX Model Zoo ILSVRC-2012 synset](https://github.com/onnx/models)  

## Model Architecture:   
**Architecture Type:** Convolution Neural Network (CNN)   
**Network Architecture:** ResNet50  
**This model was developed based on:** Not applicable (trained from scratch on ImageNet; not fine-tuned from another model).  
**Number of model parameters:** 25.6M  

<!--
Computational Load (Internal Only: For NVIDIA Models Only)
Cumulative Compute: N/A (third-party model; NVIDIA did not train it)
Estimated Energy and Emissions for Model Training: N/A (third-party model; NVIDIA did not train it)
-->  

  
 
## Input:  
**Input Type(s):** Image  
**Input Format(s):** RGB (Red, Green, Blue)  
**Input Parameters:** Two-Dimensional (2D)  
**Other Properties Related to Input:** Resolution: 224 by 224 RGB. Preprocessing: convert to RGB; resize so the shorter side is 256 pixels; center-crop to 224 by 224; scale pixels to 0–1; normalize with ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225); layout is NCHW float32, batch size 1.  
  

## Output:  
**Output Type(s):** Text  
**Output Format:** String  
**Output Parameters:** One-Dimensional (1D)  
**Other Properties Related to Output:** The output text is the ImageNet class name of the primary object in the input image, with a confidence score. For example, if the input is a photo of a dog, the output is the predicted breed such as "golden retriever" and its probability. The model produces 1000 class logits; post-processing applies softmax, ranks the top classes, and maps indices to names from synset.txt. Inference is one image in, one classification result out.  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** NVIDIA CUDA Deep Learning Accelerator (cuDLA); ONNX Runtime (CUDA Execution Provider)  
**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere architecture-based Jetson AGX Orin  
**Supported Operating System(s):** Linux for Tegra  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  


## Model Version(s): 
ResNet-50-v1-12  

To integrate the model into an AI system, clone the sample repository and install the setup and runtime packages from `requirements.txt` (including `onnx` and `onnx-graphsurgeon` for preparation). On the target Jetson AGX Orin with JetPack, TensorRT `trtexec`, and `curl` installed, run `bash setup_models.sh`. This downloads [resnet50-v1-12.onnx from the ONNX Model Zoo](https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx), runs `prepare_resnet50_dla.py`, and builds the standalone loadable with `trtexec`. The ONNX Model Zoo is the source of truth; `models/` contains only local, ignored downloads and generated artifacts. No Git LFS setup is needed. See [README Setup](README.md#setup) for environment and TensorRT path configuration. Then run `./venv/bin/python run.py`; the sample preprocesses the image with NumPy, runs the stem and head with ONNX Runtime on the GPU, and runs the backbone with cuDLA on the Deep Learning Accelerator.
 
## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:** Image  

**Image Training Data Size:** 1 Million to 1 Billion Images  
**Data Collection Method by dataset:** Automated  
**Labeling Method by dataset:** Manually-Labelled  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Training used ILSVRC-2012 ImageNet: about 1.28 million labeled photographs across 1,000 everyday object classes (animals, vehicles, household items, and similar). Images were collected from the public web around 2009–2012 and annotated by people, so the domain is natural RGB photography, not medical, satellite, or synthetic data.

### Testing Dataset:

**Data Collection Method by dataset:** Automated  
**Labeling Method by dataset:** Manually-Labelled  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Original evaluation used the ILSVRC-2012 ImageNet validation set (50,000 images, 1,000 classes). The ONNX Model Zoo reports about 74.97% top-1 and 92.33% top-5 accuracy for ResNet-50-v1-12.

### Evaluation Dataset:
**Data Collection Method by dataset:** Automated  
**Labeling Method by dataset:** Manually-Labelled  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Evaluation follows the ILSVRC-2012 ImageNet classification benchmark: 50,000 validation photographs over 1,000 object classes in everyday natural-image domains. The usual metric is top-1 and top-5 accuracy with a single 224 by 224 center crop after standard ImageNet preprocessing. ONNX Model Zoo lists about 74.97% top-1 and 92.33% top-5 for ResNet-50-v1-12.



## Inference:
**Acceleration Engine:** NVIDIA CUDA Deep Learning Accelerator (cuDLA) for the backbone; ONNX Runtime with the CUDA Execution Provider for the stem and head  
**Hardware Requirements (GPU Architecture, Model):** NVIDIA Ampere architecture-based Jetson AGX Orin

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.  
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

**Generated by NVIDIA Model Card Generator Toolkit.**
