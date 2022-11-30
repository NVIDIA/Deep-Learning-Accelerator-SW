# Instructions to Download, Prepare & Run DLA Reference Models
This page provides instructions for optimizing the performance of models running on DLA. 
Make sure you install the Python dependencies from `requirements.txt`.

Below steps were run with TensorRT 8.5 on an Orin L4T platform where `trtexec` was present in `/usr/local/tensorrt/bin/trtexec`.

## ResNet-50
### Download
Original location: https://zenodo.org/record/4735647/files/resnet50_v1.onnx

Mirror: https://web.archive.org/web/20221201211434/https://zenodo.org/record/4735647/files/resnet50_v1.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run
1. Download `resnet50_v1.onnx` from above links
2. Run `python3 resnet50.py` on any platform (host or Orin target)
3. Copy the generated `resnet50_v1_prepared.onnx` to your Orin target
4. Run the following command: `./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1MiB --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=resnet50_v1_prepared.onnx --shapes=input_tensor:0:2x3x224x224`

## SSD-ResNet-34

### Download
Original location: https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

Mirror: https://web.archive.org/web/20221201213043/https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run
1. Download `resnet34-ssd1200.onnx` from above links
2. Run `python3 ssd_resnet34.py` on any platform (host or Orin target)
3. Copy the generated `resnet34-ssd1200_prepared.onnx` to your Orin target
4. Run the following command on your Orin target: `./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1MiB --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=dla/regression/testdb/resnet34-ssd1200_prepared.onnx`


## SSD-MobileNetV1

### Download
Original location: https://zenodo.org/record/4735652/files/ssd_mobilenet_v1_coco_2018_01_28.onnx

Mirror: https://web.archive.org/web/20221201212531/https://zenodo.org/record/4735652/files/ssd_mobilenet_v1_coco_2018_01_28.onnx

Source: https://github.com/mlcommons/inference/blob/2acca351fbdf7821a5d8fa1d9a9799d50ab270f1/vision/classification_and_detection/README.md

### Prepare & Run
1. Download `ssd_mobilenet_v1_coco_2018_01_28.onnx` from above links
2. Run `python3 ssd_mobilenetv1.py` on any platform (host or Orin target)
3. Copy the generated `ssd_mobilenet_v1_coco_2018_01_28_prepared.onnx` to your Orin target

4. Run the following command on your Orin target: `./trtexec --useDLACore=0 --int8 --memPoolSize=dlaSRAM:1MiB --inputIOFormats=int8:dla_hwc4 --outputIOFormats=int8:chw32 --onnx=ssd_mobilenet_v1_coco_2018_01_28_prepared.onnx --shapes=Preprocessor/sub:0:2x3x300x300`
