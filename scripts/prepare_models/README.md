# Instructions to Download, Prepare & Run DLA Reference Models

This page provides instructions for optimizing the performance of models running on DLA.
The Python steps are expected to run on host: Make sure you install the Python dependencies from `requirements.txt`.

Below steps involving `trtexec` were run with TensorRT 8.5 on an Orin L4T platform where the application was present in `/usr/local/tensorrt/bin/trtexec`.

## ResNet-50

### Download

Original location: https://zenodo.org/record/4735647/files/resnet50_v1.onnx

Mirror: https://web.archive.org/web/20221201211434/https://zenodo.org/record/4735647/files/resnet50_v1.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run

1. Download `resnet50_v1.onnx` from above links
2. Run `python3 resnet50.py` on host
3. Copy the generated `resnet50_v1_prepared.onnx` to your Orin target
4. Run the following command on your Orin target:
```bash
./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=resnet50_v1_prepared.onnx --shapes=input_tensor:0:2x3x224x224
```

## ResNet-50 with QDQ Translator

### Prepare & Run

1. Follow steps in [../../tools/qdq-translator/e2e_workflow/tensorflow_workflow/](../../tools/qdq-translator/e2e_workflow/tensorflow_workflow/README.md) – make sure to include the `--add_unary_ew_scales_for_dla` arg and that your DLA SW version is at least 3.13.0 (as detailed in [QDQ Translator](../../tools/qdq-translator/README.md)).
2. Run `python3 scripts/prepare_models/resnet50_noqdq.py` from [repo top dir](../..)
3. Copy the resulting ``translated/`` dir (now containing `resnet_50v1_noqdq_prepared.onnx`) to your Orin target
4. Run the following command on your Orin target:
```bash
trtexec --onnx=translated/resnet_50v1_noqdq_prepared.onnx \
        --calib=translated/resnet_50v1_precision_config_calib.cache \
        --useDLACore=0 \
        --int8 \
        --fp16 \
        --precisionConstraints=prefer \
        --memPoolSize=dlaSRAM:1 \
        --shapes=StatefulPartitionedCall/resnet50/quant_conv1_bn/FusedBatchNormV3:0:2x3x224x224 \
        --layerPrecisions=$(cat translated/resnet_50v1_precision_config_layer_arg.txt) \
        --inputIOFormats=int8:hwc4 --outputIOFormats=fp16:chw16
```
5. Optional - you can compare the pure int8 latency without the calibration cache (and using dummy scales instead) by running the following – the latency is expected to be similar:
```bash
trtexec --onnx=translated/resnet_50v1_noqdq_prepared.onnx \
        --useDLACore=0 \
        --int8 \
        --memPoolSize=dlaSRAM:1 \
        --shapes=StatefulPartitionedCall/resnet50/quant_conv1_bn/FusedBatchNormV3:0:2x3x224x224 \
        --inputIOFormats=int8:hwc4 --outputIOFormats=int8:chw32
```

## SSD-ResNet-34

### Download

Original location: https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

Mirror: https://web.archive.org/web/20221201213043/https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run

1. Download `resnet34-ssd1200.onnx` from above links
2. Run `python3 ssd_resnet34.py` on host
3. Copy the generated `resnet34-ssd1200_prepared.onnx` to your Orin target
4. Run the following command on your Orin target:
```bash
./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=resnet34-ssd1200_prepared.onnx
```

## SSD-MobileNetV1

### Download

Original location: https://zenodo.org/record/4735652/files/ssd_mobilenet_v1_coco_2018_01_28.onnx

Mirror: https://web.archive.org/web/20221201212531/https://zenodo.org/record/4735652/files/ssd_mobilenet_v1_coco_2018_01_28.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run

1. Download `ssd_mobilenet_v1_coco_2018_01_28.onnx` from above links
2. Run `python3 ssd_mobilenetv1.py` on host
3. Copy the generated `ssd_mobilenet_v1_coco_2018_01_28_prepared.onnx` to your Orin target
4. Run the following command on your Orin target:
```bash
./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=ssd_mobilenet_v1_coco_2018_01_28_prepared.onnx --shapes=Preprocessor/sub:0:2x3x300x300
```

## RetinaNet ResNeXt-50

### Download

Original location: https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run

1. Download `resnext50_32x4d_fpn.onnx` from above links
2. Run `polygraphy surgeon sanitize resnext50_32x4d_fpn.onnx -o resnext50_32x4d_fpn_sanitized.onnx --fold-constants` on host
2. Run `python3 retinanet_resnext50.py` on host
3. Copy the generated `resnext50_32x4d_fpn_sanitized_prepared.onnx` to your Orin target
4. Run the following command on your Orin target:
```bash
./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=resnext50_32x4d_fpn_prepared.onnx
```

## RetinaNet ResNet-34

### Download

- Original locations: 
    - Full model: https://zenodo.org/record/8144349/files/retinanet_rn34_1280x768_dummy.onnx
    - Backbone without RetinaNet head: https://zenodo.org/record/8144349/files/backbone_rn34_1280x768_dummy.onnx
- Source: See "Structured Sparsity Case Study: Object Detection Accuracy with RetinaNet ResNet-34" in `../../README.md`. Note that the parameters are random due to dataset licensing restrictions (hence the `_dummy` appendix).

### Prepare & Run

For `MODEL=retinanet_rn34_1280x768_dummy.onnx` or `backbone_rn34_1280x768_dummy.onnx`:
1. Download `${MODEL}` from above links
2. Copy `${MODEL}` to your Orin target
3. Run the following command on your Orin target:
```bash
./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1 --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=${MODEL}.onnx
```
4. For sparse operation, add `--sparsity=force`.
