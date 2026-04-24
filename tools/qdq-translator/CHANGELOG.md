# NVIDIA QDQ Translator change log

Dates are in YYYY-MM-DD format.

## Unreleased

- Added support for translating models with a standalone DequantizeLinear connected to an int8/uint8 graph input.
- Skip the onnxoptimizer ``fuse_bn_into_conv`` pass when any BatchNormalization parameter is fp16, to avoid graph decomposition (see https://github.com/onnx/optimizer/blob/master/onnxoptimizer/passes/fuse_bn_into_conv.h#L40).

## v0.2.0 (2023-08-09)

- Added "infer_mul_scales" arg for handling Mul op.
- Added "calibration_type" arg for handling errors in weight scale confirmation when calibration method is not max.
- Made compatibility adjustments for cases where quantize_node.inputs[1] of certain nodes in the yolov5 model are of type gs.Constant.

## v0.1.0 (2022-11-23)

- Initial release.
