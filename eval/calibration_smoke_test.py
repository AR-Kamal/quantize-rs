#!/usr/bin/env python3
"""Validate static Conv calibration against held-out synthetic inputs in ORT.

Requires numpy, onnx and onnxruntime. --python also tests an installed wheel.
This is a deterministic correctness regression, not a real-dataset accuracy or
latency benchmark. Calibration and held-out samples share the same distribution.
"""
import argparse
from contextlib import nullcontext
import platform
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ort_utils import cpu_session_options
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
    options = cpu_session_options()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.optimized_model_filepath = str(dst.with_suffix(".unoptimized.onnx"))
    unfused = ort.InferenceSession(str(dst), options, providers=["CPUExecutionProvider"])
    optimized_options = cpu_session_options()
    optimized_options.log_severity_level = 3
    optimized_options.optimized_model_filepath = str(dst.with_suffix(".optimized.onnx"))
    fused = ort.InferenceSession(str(dst), optimized_options, providers=["CPUExecutionProvider"])
    for sample_index, sample in enumerate(samples):
        inputs = {"X": sample[None]}
        expected = ref.run(None, inputs)[0]
        a = unfused.run(None, inputs)[0]
        b = fused.run(None, inputs)[0]
        context = f"{dst.name}, ORT {ort.__version__}, sample={sample_index}"
        errors = {}
        for mode, actual in [("unoptimized", a), ("optimized", b)]:
            assert actual.shape == expected.shape, f"{context}: {mode} output shape"
            assert np.isfinite(actual).all(), f"{context}: {mode} nonfinite output"
            errors[mode] = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-8))
        if any(error >= 0.05 for error in errors.values()):
            np.savez(dst.with_suffix(".failure.npz"), input=sample[None], expected=expected,
                     unoptimized=a, optimized=b)
        assert all(error < 0.05 for error in errors.values()), f"{context}: relative RMSE {errors}, limit=0.05"
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4, err_msg=context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--python", action="store_true", dest="python_api")
    parser.add_argument("--artifacts", type=Path, help="Keep generated models and data in this directory for diagnosis")
    args = parser.parse_args()
    args.binary = str(Path(args.binary).resolve(strict=True))
    print(f"Platform {platform.platform()}, ONNX Runtime {ort.__version__}, ONNX {onnx.__version__}, NumPy {np.__version__}, x64quantprecision=1", flush=True)
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=True)
    directory = nullcontext(args.artifacts) if args.artifacts else tempfile.TemporaryDirectory()
    with directory as td:
        root = Path(td)
        src, data = root / "model.onnx", root / "samples.npy"
        build_model(src)
        rng = np.random.default_rng(42)
        np.save(data, rng.uniform(-1, 1, (24, 3, 8, 8)).astype(np.float32))
        held_out = rng.uniform(-1, 1, (5, 3, 8, 8)).astype(np.float32)
        np.save(root / "held_out.npy", held_out)
        for method in ["minmax", "percentile:99.9", "entropy", "mse"]:
            print(f"[check] {method}: CLI Conv calibration", flush=True)
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
            print("[check] Python Conv calibration", flush=True)
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
