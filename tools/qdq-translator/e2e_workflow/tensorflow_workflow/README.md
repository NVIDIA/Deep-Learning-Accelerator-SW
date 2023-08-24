# End-to-end workflow for ResNet-50v1 using NVIDIA's tensorflow-quantization toolkit

## About
1. Obtain QAT ResNet-50v1 ONNX model with [NVIDIA's tensorflow-quantization toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization)
2. Convert QAT model to Stripped model + scales
3. Deploy and evaluate TensorRT engine. 

## Requirements
1. Install `tensorflow-quantization` toolkit:
```
./install.sh
```

2. Manually install TensorRT python wheel. This should be the same as your `trtexec` version.
```
pip install TensorRT-8.6EA/python/tensorrt-8.6.0-cp38-none-linux_x86_64.whl 
```

## Workflow

### 1. Generate QAT Model
Obtain QAT ResNet-50v1 ONNX model with [NVIDIA's tensorflow-quantization toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization) and copy it to `../../models/resnet_50v1.onnx`.
 Prepare the data and follow the steps in [ResNet's README file](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization/examples/resnet#requirements).
 To reproduce the results in the linked README, use `bs=64, ep=10, lr=0.001` as hyper-parameters.

### 2. Evaluate QAT Model
Calculate the QAT model's accuracy (baseline):
```
./scripts/accuracy_eval_qat.sh
```

### 3. Convert QAT Model
```
cd ../.. && python3 qdq_translator.py -i ./models/resnet_50v1.onnx -o ./translated && cd -
```

### For running on DLA (not compatible with GPU)
 Add `--add_unary_ew_scales_for_dla` (see [Why is `--add_unary_ew_scales_for_dla` needed?](../README.md#why-is---add_unary_ew_scales_for_dla-needed) for details):
```
cd ../.. && python3 qdq_translator.py -i ./models/resnet_50v1.onnx -o ./translated --add_unary_ew_scales_for_dla && cd -
```

### 4. Evaluate Converted Model 
Calculate the Converted models' accuracy:
```
./scripts/accuracy_eval.sh
```

### 5. Run model on DLA without GPU fallback
Follow the steps in [ResNet-50 with QDQ Translator](../../../scripts/prepare_models/README.md#resnet-50-with-qdq-translator) (requires `--add_unary_ew_scales_for_dla`).

## Results
### ImageNet Top-1 accuracy with TensorRT 8.6-EA (on NVIDIA GeForce RTX™ 3090)

| Model                                  | Original Model<br>*(Explicit Quantization, with Q/DQ nodes)* | Model converted by QDQ Translator<br>*(Implicit Quantization with calibration table, without Q/DQ nodes)* |
|----------------------------------------|---------------------------------|----------------------------------|
| ResNet-50v1 | 75.18%                           | 75.22%                            |

Evaluated on x86 training host. Results on Orin's DLA and GPU expected to be similar.
