#!/usr/bin/env python3
"""Generated matrix activation tests: real calibration, accuracy and integer kernels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from ort_utils import CPU_SESSION_CONFIG, cpu_session_options
from onnx import helper as h, numpy_helper as nh, TensorProto


def build(path, kind, trans_a=0, trans_b=0):
    rng = np.random.default_rng(917)
    k, n = 8, 5
    shape = (1 if trans_a else k, n)
    w = rng.uniform(-.6, .6, shape).astype(np.float32)
    if trans_b:
        w = w.T.copy()
    tensors = [nh.from_array(w, "W")]
    expected_axes = {"W": 0 if trans_b else 1}
    rows = k if trans_a else 1
    protected = []
    if kind.startswith("gemm"):
        inputs = ["X", "W"]
        if kind != "gemm_no_bias":
            bias_shape = (rows, n) if kind == "gemm_matrix_bias" else (n,)
            tensors.append(nh.from_array(rng.uniform(-.1, .1, bias_shape).astype(np.float32), "bias"))
            inputs.append("bias")
            protected.append("bias")
        nodes = [h.make_node("Gemm", inputs, ["Y"], transA=trans_a, transB=trans_b)]
    else:
        nodes = [h.make_node("MatMul", ["X", "W"], ["Y"])]
    output_shapes = {"Y": [rows, n]}
    if kind in ("chain", "adjacent"):
        nodes[0].output[0] = "H"
        tensors += [nh.from_array(rng.uniform(-.5, .5, (n, 3)).astype(np.float32), "V"),
                    nh.from_array(np.zeros(n, np.float32), "bias")]
        protected.append("bias")
        if kind == "chain":
            nodes += [h.make_node("Add", ["H", "bias"], ["B"]), h.make_node("Relu", ["B"], ["R"])]
        nodes.append(h.make_node("Gemm", ["R" if kind == "chain" else "H", "V"], ["Y"]))
        output_shapes = {"Y": [1, 3]}
        expected_axes["V"] = 1
    elif kind in ("shared", "fanout"):
        if kind == "fanout":
            tensors.append(nh.from_array(w * .7, "V"))
            expected_axes["V"] = 1
        nodes.append(h.make_node("Gemm", ["X", "V" if kind == "fanout" else "W"], ["Z"]))
        output_shapes["Z"] = [1, n]
    graph = h.make_graph(nodes, "matrix_activation", [h.make_tensor_value_info("X", TensorProto.FLOAT, [1, k])],
                         [h.make_tensor_value_info(name, TensorProto.FLOAT, dims) for name, dims in output_shapes.items()], tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    calibration = rng.uniform(-2, 2, (256, k)).astype(np.float32)
    held_out = rng.uniform(-1, 1, (32, k)).astype(np.float32)
    count = None if kind in ("shared", "fanout") else (2 if kind in ("chain", "adjacent") else 1)
    return calibration, held_out, expected_axes, protected, count


def session(path, optimize, profile=None, precision=True):
    # ORT 1.30's precision conversion cannot initialize shared/fan-out QDQ
    # graphs on AVX2 (duplicate converted initializers). Those cases explicitly
    # exercise default runtime execution; they do not claim integer fusion.
    opts = cpu_session_options() if precision else ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if profile:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile)
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def run(binary, src, data, dst, method="minmax", extra=(), success=True):
    result = subprocess.run([str(binary), "calibrate-matrix", str(src), "--data", str(data), "-o", str(dst), "--method", method, *extra],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if (result.returncode == 0) != success:
        raise AssertionError(result.stdout + result.stderr)


def check(src, dst, inputs, axes, protected, integer_count):
    onnx.checker.check_model(str(dst))
    a, b = onnx.load(src), onnx.load(dst)
    before = {t.name: t for t in a.graph.initializer}
    after = {t.name: t for t in b.graph.initializer}
    assert [v.SerializeToString() for v in a.graph.output] == [v.SerializeToString() for v in b.graph.output]
    assert [v.SerializeToString() for v in a.graph.input] == [v.SerializeToString() for v in b.graph.input]
    for name in protected:
        assert before[name].SerializeToString() == after[name].SerializeToString()
    for name, axis in axes.items():
        dq = next(n for n in b.graph.node if n.op_type == "DequantizeLinear" and n.output[0] == name)
        assert h.get_attribute_value(next(a for a in dq.attribute if a.name == "axis")) == axis
        assert nh.to_array(after[dq.input[1]]).size == before[name].dims[axis]
        assert np.all(nh.to_array(after[dq.input[2]]) == 0)
        assert after[dq.input[0]].data_type == TensorProto.INT8
    precision = integer_count is not None
    reference = session(src, False, precision=precision)
    plain = session(dst, False, precision=precision)
    optimized = session(dst, True, precision=precision)
    output_scales = []
    for value in b.graph.output:
        dq = next(n for n in b.graph.node if n.op_type == "DequantizeLinear" and n.output[0] == value.name)
        output_scales.append(float(nh.to_array(after[dq.input[1]])))
    expected_all, actual_all = [], []
    max_parity_steps = 0.
    for sample in inputs:
        feed = {"X": sample[None]}
        for expected, x, y, step in zip(reference.run(None, feed), plain.run(None, feed), optimized.run(None, feed), output_scales):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            # Integer requantization can round a tie one output quantum differently.
            difference = float(np.max(np.abs(x - y)))
            assert difference <= step * 1.01 + 1e-6, (difference, step)
            max_parity_steps = max(max_parity_steps, difference / step)
            expected_all.append(expected.ravel())
            actual_all.append(y.ravel())
    expected, actual = np.concatenate(expected_all), np.concatenate(actual_all)
    relative_rmse = float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-8))
    assert relative_rmse < .05, relative_rmse
    profiled = session(dst, True, dst.with_suffix(".profile"), precision=precision)
    profiled.run(None, {"X": inputs[0][None]})
    events = json.loads(Path(profiled.end_profiling()).read_text())
    kernels = [e for e in events if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")]
    integer = [e for e in kernels if e.get("args", {}).get("op_name") in ("QLinearMatMul", "QGemm")]
    floating = [e for e in kernels if e.get("args", {}).get("op_name") in ("MatMul", "Gemm")]
    if integer_count is not None:
        assert len(integer) == integer_count and not floating, [e.get("args", {}).get("op_name") for e in kernels]
    else:
        # Shared QDQ edges can prevent ORT fusion. Check the numerical contract
        # and report actual execution without claiming these fan-out cases fuse.
        assert len(integer) + len(floating) == 2
    for event in integer:
        types = event["args"]["input_type_shape"]
        # ORT may convert signed activations to UINT8 and omit prepacked weight
        # inputs from the profile. The executed Q operator and quantized data
        # input establish integer execution; serialized weight types are checked above.
        assert "int8" in types[0] or "uint8" in types[0], types
    return dict(session_config=CPU_SESSION_CONFIG if precision else {}, relative_rmse=relative_rmse, maximum_parity_error_in_output_steps=max_parity_steps,
                executed_integer_kernels=dict(Counter(e["args"]["op_name"] for e in integer)),
                executed_float_matrix_kernels=dict(Counter(e["args"]["op_name"] for e in floating)),
                fusion_required=integer_count is not None,
                profiled_activation_types=[list(e["args"]["input_type_shape"][0]) for e in integer],
                source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), output_sha256=hashlib.sha256(dst.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        src, dst, data = root / "source.onnx", root / "output.onnx", root / "data.npy"
        cases = [(name, 0, 0) for name in ("matmul", "chain", "adjacent", "shared", "fanout", "gemm_no_bias")]
        cases += [("gemm_bias", a, b) for a in (0, 1) for b in (0, 1)]
        for kind, trans_a, trans_b in cases:
            calibration, held_out, axes, protected, count = build(src, kind, trans_a, trans_b)
            np.save(data, calibration)
            methods = ("minmax", "percentile:99.9", "entropy", "mse") if kind == "matmul" else ("minmax",)
            for method in methods:
                run(binary, src, data, dst, method)
                key = f"{kind}/transA={trans_a}/transB={trans_b}/{method}"
                results[key] = check(src, dst, held_out, axes, protected, count)
                result = results[key]
                print(f"[ok] {key}: relative RMSE={result['relative_rmse']:.4f}; integer={result['executed_integer_kernels']}; float={result['executed_float_matrix_kernels']}", flush=True)
        calibration, _, _, _, _ = build(src, "matmul")
        np.save(data, calibration)
        original = onnx.load(src)
        for case in ("alpha", "int64_input", "dynamic_input", "indirect", "excluded", "shared_conflict", "matrix_bias"):
            model = onnx.ModelProto(); model.CopyFrom(original)
            extra = ()
            if case == "alpha":
                model.graph.node[0].op_type = "Gemm"
                model.graph.node[0].attribute.append(h.make_attribute("alpha", .5))
            elif case == "int64_input":
                model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT64
            elif case == "dynamic_input":
                model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "K"
            elif case == "indirect":
                model.graph.node[0].input[1] = "Wt"
                model.graph.node.insert(0, h.make_node("Identity", ["W"], ["Wt"]))
            elif case == "excluded":
                extra = ("--exclude", "W")
            elif case == "matrix_bias":
                model.graph.node[0].op_type = "Gemm"
                model.graph.initializer.append(nh.from_array(np.zeros((1, 5), np.float32), "bias"))
                model.graph.node[0].input.append("bias")
            else:
                model.graph.node.append(h.make_node("Add", ["W", "W"], ["unused"]))
            onnx.save(model, src)
            dst.write_bytes(b"preserve existing output")
            run(binary, src, data, dst, extra=extra, success=False)
            assert dst.read_bytes() == b"preserve existing output"
        print("[ok] unsupported configurations preserve existing files", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__,
                                               provider="CPUExecutionProvider", binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                                               seed=917, calibration_samples=256, held_out_samples=32,
                                               maximum_relative_rmse=.05, cases=results), indent=2) + "\n", encoding="utf-8")
    print("MATRIX ACTIVATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
