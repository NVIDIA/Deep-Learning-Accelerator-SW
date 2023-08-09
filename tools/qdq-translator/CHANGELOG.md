# NVIDIA QDQ Translator change log

Dates are in YYYY-MM-DD format.
## v0.2.0 (2023-08-09)

- Added "infer_mul_scales" arg for handling Mul op.
- Added "calibration_type" arg for handling errors in weight scale confirmation when calibration method is not max.
- Made compatibility adjustments for cases where quantize_node.inputs[1] of certain nodes in the yolov5 model are of type gs.Constant.

## v0.1.0 (2022-11-23)

- Initial release.
