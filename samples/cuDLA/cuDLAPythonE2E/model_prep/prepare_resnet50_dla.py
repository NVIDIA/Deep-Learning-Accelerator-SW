# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare resnet50-v1-12.onnx for DLA compilation on Orin.

Usage:
    python3 prepare_resnet50_dla.py [--input resnet50-v1-12.onnx] \
                                    [--output resnet50_v1_prepared.onnx] \
                                    [--stem   resnet50_stem.onnx] \
                                    [--head   resnet50_head.onnx] \
                                    [--batch N] [--fp16]

The script produces the **three** artifacts the hybrid GPU+DLA pipeline needs:

    ┌ stem     data -> resnetv17_pool0_fwd            [N,   3,224,224] -> [N, 64,56,56]   (GPU)
    ├ backbone resnetv17_pool0_fwd -> ..stage4_act2   [N,  64, 56, 56] -> [N,2048, 7, 7]  (DLA)
    └ head     ..stage4_activation2 -> ..dense0_fwd   [N,2048,  7,  7] -> [N,1000]        (GPU)

* **backbone** (``--output``) is trimmed to the layers the Orin DLA can accept
  and is what gets compiled to a DLA loadable (e.g. via
  ``trtexec --buildDLAStandalone``). BatchNormalization nodes are preserved.
  Pass ``--fp16`` to convert the backbone's FP32 tensors and weights to FP16,
  as required by strongly typed TensorRT 11 DLA builds. The GPU stem/head
  are still extracted unchanged from the original FP32 model.
* **stem** (``--stem``) and **head** (``--head``) are the GPU-side prefix/suffix,
  carved straight out of the original graph at the same cut-point tensors
  (``onnx.utils.extract_model``). They run on the GPU (e.g. ONNX Runtime) around
  the DLA stage. Pass ``--no-subgraphs`` to emit only the backbone.

The backbone graph has a concrete batch dimension (default 1). Pass ``--batch 0``
to leave it symbolic (``N``) if the downstream tool resolves batch at
engine-build time (for example ``trtexec --shapes=...``). Prefer a concrete
batch for DLA standalone builds; the stem/head keep the original model's
(symbolic ``N``) batch dimension, since they run on the GPU where dynamic
batch is fine.

Pipeline
--------
1. Trim the stem MaxPool and the classification head (see "DLA-incompatible
   layers removed" below) so the backbone graph contains only ResNet-50 stages
   1-4 (Conv + BatchNormalization + ReLU + residual Add).
2. Extract the stem (original input -> backbone input) and head (backbone output
   -> original output) subgraphs from the original model.

DLA-incompatible layers removed (they become the stem/head instead)
-------------------------------------------------------------------
1. Stem MaxPool  resnetv17_pool0_fwd  (MaxPool 3x3/2 on 112x112x64)
   DLA compiler error code 2 on this layer with TRT 10.x.

2. Classification head:
   - GlobalAveragePool  resnetv17_pool1_fwd  (2048x7x7 -> 2048x1x1)
     TRT 10.x maps GlobalAveragePool to an internal Reduce op which DLA
     does not support.  Manually converting it to AveragePool(kernel=7x7)
     also fails with error code 2 (DLA channel*kernel constraint).
   - Flatten  flatten_473
   - Gemm/FC  resnetv17_dense0_fwd  (1000 classes)

Resulting backbone subgraph
---------------------------
  input:  resnetv17_pool0_fwd            [N,   64, 56, 56]
  output: resnetv17_stage4_activation2   [N, 2048,  7,  7]

  Op counts after the default pipeline (trim only):
    Conv 52   BatchNormalization 52   Relu 48   Add 16

All ResNet-50 stages 1-4 (Conv+BatchNormalization+ReLU+Residual shortcuts)
compile successfully on the Orin DLA with ``trtexec --buildDLAStandalone``.

Requirements: onnx, onnx-graphsurgeon, numpy
    pip install onnx onnx-graphsurgeon numpy
"""

import argparse
import os
from collections import Counter

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference


_INPUT_NAME  = "resnetv17_pool0_fwd"
_INPUT_CHW   = (64, 56, 56)
_OUTPUT_NAME = "resnetv17_stage4_activation2"
_OUTPUT_CHW  = (2048, 7, 7)
_DEFAULT_OUTPUT = "resnet50_v1_prepared.onnx"
_DEFAULT_STEM   = "resnet50_stem.onnx"
_DEFAULT_HEAD   = "resnet50_head.onnx"


def extract_subgraph(graph, input_names, output_names, input_shape):
    tensors = graph.tensors()
    for tensor in tensors.values():
        if tensor.shape is None or isinstance(tensor, gs.Constant):
            continue
        if len(tensor.shape) > 0:
            tensor.shape[0] = input_shape[0]

    graph.inputs = [
        tensors[name].to_variable(dtype=np.float32, shape=input_shape)
        for name in input_names
    ]
    graph.outputs = [tensors[name].to_variable(dtype=np.float32) for name in output_names]
    graph.cleanup()

    for node in graph.nodes:
        for out in node.outputs:
            out.shape = None


def graph_io_names(model_path):
    """(inputs, outputs) of an ONNX graph, excluding initializer tensors."""
    m = onnx.load(model_path)
    init = {i.name for i in m.graph.initializer}
    ins = [i.name for i in m.graph.input if i.name not in init]
    outs = [o.name for o in m.graph.output]
    return ins, outs


def convert_backbone_to_fp16(graph):
    """Convert the trimmed ResNet Conv/BatchNorm/Relu/Add graph, without casts."""
    for tensor in graph.tensors().values():
        if tensor.dtype == np.float32:
            if isinstance(tensor, gs.Constant):
                tensor.values = tensor.values.astype(np.float16)
            else:
                tensor.dtype = np.float16


def extract_stem_head(original_path, input_name, output_name, stem_path, head_path):
    """Carve the GPU stem/head out of the ORIGINAL model at the DLA cut points.

    stem: original input(s) -> [input_name]      (data -> backbone input)
    head: [output_name] -> original output(s)    (backbone output -> logits)

    Uses onnx.utils.extract_model, which preserves the original tensor shapes
    (so the batch dimension stays symbolic N, fine for GPU execution).
    """
    orig_in, orig_out = graph_io_names(original_path)
    onnx.utils.extract_model(original_path, stem_path, orig_in, [input_name])
    onnx.utils.extract_model(original_path, head_path, [output_name], orig_out)
    return orig_in, orig_out


def op_counts(model):
    return Counter(n.op_type for n in model.graph.node)


def _describe(path, tag):
    """Print the I/O tensors (name + shape) and op counts of a written subgraph."""
    g = onnx.load(path).graph

    def shp(t):
        return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]

    counts = Counter(n.op_type for n in g.node)
    print(f"  {tag} -> {path}")
    print(f"    in : {[(i.name, shp(i)) for i in g.input]}")
    print(f"    out: {[(o.name, shp(o)) for o in g.output]}")
    print(f"    nodes: {sum(counts.values())}  "
          + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="resnet50-v1-12.onnx", help="Path to the original ONNX model")
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Path for the prepared DLA backbone ONNX. Default: '{_DEFAULT_OUTPUT}'.",
    )
    parser.add_argument(
        "--stem",
        default=_DEFAULT_STEM,
        help=f"Path for the GPU stem (prefix) subgraph. Default: '{_DEFAULT_STEM}'.",
    )
    parser.add_argument(
        "--head",
        default=_DEFAULT_HEAD,
        help=f"Path for the GPU head (suffix) subgraph. Default: '{_DEFAULT_HEAD}'.",
    )
    parser.add_argument(
        "--no-subgraphs",
        action="store_true",
        help="Emit only the DLA backbone; skip writing the stem/head subgraphs.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert backbone tensors and weights to FP16 for TensorRT 11 DLA. "
             "The GPU stem/head retain their original types.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Concrete batch dimension to bake into the prepared backbone ONNX. "
             "Pass 0 (or any value <= 0) to leave it symbolic (N) for tools that "
             "resolve batch at engine-build time. Default: 1.",
    )
    parser.add_argument(
        "--input-name",
        default=_INPUT_NAME,
        help="Graph tensor to trim to as the new input boundary. Default is the "
             f"float ResNet-50 stem name '{_INPUT_NAME}'. For a QDQ (QuantizeLinear/"
             "DequantizeLinear) export the same boundary is named "
             "'resnetv17_pool0_fwd_QuantizeInput' (the MaxPool output that feeds the "
             "first stage-1 QuantizeLinear).",
    )
    parser.add_argument(
        "--output-name",
        default=_OUTPUT_NAME,
        help="Graph tensor to trim to as the new output boundary. Default "
             f"'{_OUTPUT_NAME}' is shared by both the float and QDQ ResNet-50 exports.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Input ONNX not found: {args.input}\n"
            "Download from https://github.com/onnx/models "
            "(validated/vision/classification/resnet/model/resnet50-v1-12.onnx -- "
            "the raw LFS file, not the GitHub /blob/ HTML landing page)."
        )

    graph = gs.import_onnx(onnx.load(args.input))

    batch_dim = args.batch if args.batch > 0 else -1
    in_c, in_h, in_w = _INPUT_CHW
    out_c, out_h, out_w = _OUTPUT_CHW

    extract_subgraph(
        graph,
        input_names=[args.input_name],
        output_names=[args.output_name],
        input_shape=(batch_dim, in_c, in_h, in_w),
    )
    if batch_dim > 0:
        print(f"Pinned batch dimension to {batch_dim} (concrete).")
    else:
        print("Left batch dimension symbolic (N).")

    if args.fp16:
        convert_backbone_to_fp16(graph)
        print("Converted backbone tensors and weights to FP16 (GPU stem/head unchanged).")

    print(f"Saving prepared backbone to {args.output} ...")
    model = gs.export_onnx(graph)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)

    counts = op_counts(model)
    batch_str = str(batch_dim) if batch_dim > 0 else "N"
    print("Done (backbone).")
    print(f"  input : {args.input_name:34s} [{batch_str:>2s}, {in_c:>4d}, {in_h:>2d}, {in_w:>2d}]")
    print(f"  output: {args.output_name:34s} [{batch_str:>2s}, {out_c:>4d}, {out_h:>2d}, {out_w:>2d}]")
    print(f"  total nodes: {sum(counts.values())}")
    for op_type, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {op_type:25s} {n}")

    if args.no_subgraphs:
        return

    print("\nExtracting GPU stem/head subgraphs from the original model ...")
    extract_stem_head(args.input, args.input_name, args.output_name,
                      args.stem, args.head)
    _describe(args.stem, "stem (GPU prefix)")
    _describe(args.head, "head (GPU suffix)")


if __name__ == "__main__":
    main()
