# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for FP16 preparation and TensorRT 11 trtexec arguments."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from model_prep.prepare_resnet50_dla import convert_backbone_to_fp16


class ModelSetupTests(unittest.TestCase):
    def test_fp16_conversion(self):
        x = gs.Variable("x", dtype=np.float32, shape=[1, 2])
        weights = gs.Constant("weights", values=np.array([[1.1, -2.2]], dtype=np.float32))
        mid = gs.Variable("mid", dtype=np.float32, shape=[1, 2])
        y = gs.Variable("y", dtype=np.float32, shape=[1, 2])
        integers = gs.Constant("integers", values=np.array([3], dtype=np.int64))
        integer_out = gs.Variable("integer_out", dtype=np.int64, shape=[1])
        graph = gs.Graph(
            nodes=[gs.Node("Add", inputs=[x, weights], outputs=[mid]),
                   gs.Node("Relu", inputs=[mid], outputs=[y]),
                   gs.Node("Identity", inputs=[integers], outputs=[integer_out])],
            inputs=[x], outputs=[y, integer_out], opset=12,
        )

        convert_backbone_to_fp16(graph)

        for tensor in (x, weights, mid, y):
            self.assertEqual(tensor.dtype, np.float16)
        np.testing.assert_array_equal(weights.values, np.array([[1.1, -2.2]], dtype=np.float16))
        self.assertEqual(integers.dtype, np.int64)
        self.assertEqual(integer_out.dtype, np.int64)
        self.assertEqual([node.op for node in graph.nodes], ["Add", "Relu", "Identity"])
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(gs.export_onnx(graph)))

    def test_trtexec11_flags(self):
        source = Path(__file__).resolve().parents[1] / "setup_models.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(source, root / source.name)
            (root / "models").mkdir()
            (root / "models/resnet50-v1-12.onnx").write_text("existing download")
            log = root / "calls.jsonl"
            stub = root / "tool"
            stub.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['TEST_LOG'], 'a') as f:\n"
                "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=str(stub), TRTEXEC=str(stub), TEST_LOG=str(log))
            result = subprocess.run(["bash", str(root / source.name)], cwd="/tmp",
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 3)  # dependency check, preparation, build
            self.assertFalse(any("--help" in args for args in calls))
            preparation = next(args for args in calls if "--input" in args)
            self.assertIn("--fp16", preparation)
            build = next(args for args in calls if "--buildDLAStandalone" in args)
            self.assertNotIn("--fp16", build)
            self.assertIn("--inputIOFormats=chw16", build)
            self.assertIn("--outputIOFormats=chw16", build)
            self.assertIn("--useDLACore=0", build)
            self.assertIn("--memPoolSize=dlaSRAM:1", build)
            self.assertIn("--saveEngine=models/resnet50_v1_prepared.loadable", build)


if __name__ == "__main__":
    unittest.main()
