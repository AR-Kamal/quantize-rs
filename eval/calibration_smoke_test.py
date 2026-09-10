#!/usr/bin/env python3
"""Validate static Conv calibration against held-out synthetic inputs in ORT.

Requires numpy, onnx and onnxruntime. --python also tests an installed wheel.
This is a deterministic correctness regression, not a real-dataset accuracy or
latency benchmark. Calibration and held-out samples share the same distribution.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def build_model(path):
    rng = np.random.default_rng(12)
    # Chain exercises reuse of a quantized upstream Conv output and preservation
    # of public graph outputs. Node names deliberately differ from tensor names.
    graph = helper.make_graph(
        [helper.make_node("Conv", ["X", "first.weight"], ["hidden"], name="first", kernel_shape=[1, 1]),
         helper.make_node("Conv", ["hidden", "second.weight"], ["Y"], name="second", kernel_shape=[1, 1])],
        "static_conv_smoke",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 8, 8])],
        [numpy_helper.from_array(rng.normal(0, 0.2, (4, 3, 1, 1)).astype(np.float32), "first.weight"),
         numpy_helper.from_array(rng.normal(0, 0.2, (2, 4, 1, 1)).astype(np.float32), "second.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run(binary, *args, ok=True):
    p = subprocess.run([binary, *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if ok and p.returncode:
        raise AssertionError(p.stdout + p.stderr)
    if not ok and p.returncode == 0:
        raise AssertionError("invalid configuration unexpectedly succeeded")
    return p


def check_accuracy(src, dst, samples):
    onnx.checker.check_model(str(dst))
    ref = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    unfused = ort.InferenceSession(str(dst), options, providers=["CPUExecutionProvider"])
    fused = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    for sample in samples:
        inputs = {"X": sample[None]}
        expected = ref.run(None, inputs)[0]
        a = unfused.run(None, inputs)[0]
        b = fused.run(None, inputs)[0]
        assert np.isfinite(a).all() and np.isfinite(b).all()
        for actual in [a, b]:
            relative_rmse = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
            assert relative_rmse < 0.05, relative_rmse
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--python", action="store_true", dest="python_api")
    args = parser.parse_args()
    args.binary = str(Path(args.binary).resolve(strict=True))
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src, data = root / "model.onnx", root / "samples.npy"
        build_model(src)
        rng = np.random.default_rng(42)
        np.save(data, rng.uniform(-1, 1, (24, 3, 8, 8)).astype(np.float32))
        held_out = rng.uniform(-1, 1, (5, 3, 8, 8)).astype(np.float32)
        for method in ["minmax", "percentile:99.9", "entropy", "mse"]:
            dst = root / f"{method.replace(':', '_')}.onnx"
            run(args.binary, "calibrate", src, "--data", data, "-o", dst, "--method", method, "--per-channel", "--symmetric")
            graph = onnx.load(dst).graph
            assert sum(n.op_type == "QuantizeLinear" for n in graph.node) == 3
            check_accuracy(src, dst, held_out)
            assert json.loads(run(args.binary, "validate", src, dst, "--format", "json").stdout)["validation_passed"]
            assert json.loads(run(args.binary, "benchmark", src, dst, "--format", "json").stdout)["structure_preserved"]
            print(f"[ok] {method}: activation QDQ, ORT accuracy and optimization parity")
        dst = root / "excluded.onnx"
        run(args.binary, "calibrate", src, "--data", data, "-o", dst, "--exclude", "second.weight", "--method", "minmax")
        g = onnx.load(dst).graph
        assert any(t.name == "second.weight" for t in g.initializer)
        assert sum(n.op_type == "QuantizeLinear" for n in g.node) == 2
        check_accuracy(src, dst, held_out)
        for flags in [["--bits", "4"], ["--native-int4"], ["--min-elements", "99999"], ["--layer-bits", "first.weight=4"]]:
            before = dst.read_bytes()
            run(args.binary, "calibrate", src, "--data", data, "-o", dst, *flags, ok=False)
            assert dst.read_bytes() == before
        run(args.binary, "calibrate", src, "--data", root / "missing.data", "-o", dst, ok=False)
        if args.python_api:
            import quantize_rs
            quantize_rs.quantize_with_calibration(str(src), str(dst), calibration_data=str(data), per_channel=True, symmetric=True,
                                                 excluded_layers=["second.weight"], min_elements=1, layer_bits={"first.weight": 8})
            g = onnx.load(dst).graph
            assert any(t.name == "second.weight" for t in g.initializer)
            assert sum(n.op_type == "QuantizeLinear" for n in g.node) == 2
            check_accuracy(src, dst, held_out)
            for kwargs in [{}, {"calibration_data": str(data), "bits": 4}, {"calibration_data": str(data), "min_elements": 99999},
                           {"calibration_data": str(data), "layer_bits": {"first.weight": 4}}]:
                before = dst.read_bytes()
                try:
                    quantize_rs.quantize_with_calibration(str(src), str(dst), **kwargs)
                except (ValueError, RuntimeError):
                    pass
                else:
                    raise AssertionError(f"invalid Python calibration accepted: {kwargs}")
                assert dst.read_bytes() == before
            print("[ok] Python calibration selection, errors and ORT execution")
    print("STATIC CALIBRATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
