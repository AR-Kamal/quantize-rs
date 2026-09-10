#!/usr/bin/env python3
"""Check graph-aware MatMul/Gemm selection and axes with ONNX Runtime.

Generated non-square models; no downloads. Requires numpy, onnx, onnxruntime.
Pass --python to exercise the installed Python extension as well as the CLI.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def build(path, kind, trans_a=0, trans_b=0):
    rng = np.random.default_rng(734)
    w = rng.normal(size=(3, 5)).astype(np.float32) * np.array([0.1, 0.3, 1, 3, 10], np.float32)
    bias = np.linspace(-0.1, 0.1, 5, dtype=np.float32)[None]
    initializers = [numpy_helper.from_array(bias, "bias")]
    protected = ["bias"]
    nodes, outputs = [], [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 5])]
    input_shape, axis = [2, 3], 1
    if kind == "gemm":
        input_shape = [3, 2] if trans_a else [2, 3]
        if trans_b:
            w = w.T.copy()
            axis = 0
        nodes = [helper.make_node("Gemm", ["X", "W", "bias"], ["Y"],
                                  transA=trans_a, transB=trans_b, alpha=0.7, beta=0.25)]
    elif kind in ["batched", "broadcast"]:
        w = np.stack([w, w * 0.3]) if kind == "batched" else w[None]
        input_shape, axis = [2, 4, 3], 2
        nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 5])]
    elif kind == "vector":
        input_shape = [3]
        nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])]
    elif kind == "shared":
        nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"]),
                 helper.make_node("Gemm", ["X", "W", "bias"], ["Z"])]
        outputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 5]))
    elif kind == "embedding":
        initializers += [numpy_helper.from_array(rng.normal(size=(8, 3)).astype(np.float32), "embedding"),
                         numpy_helper.from_array(np.array([1, 5], np.int64), "indices")]
        protected += ["embedding", "indices"]
        nodes = [helper.make_node("Gather", ["embedding", "indices"], ["lookup"]),
                 helper.make_node("Add", ["X", "lookup"], ["hidden"]),
                 helper.make_node("MatMul", ["hidden", "W"], ["Y"])]
    else:
        nodes = [helper.make_node("MatMul", ["X", "W"], ["Y"])]
    initializers.append(numpy_helper.from_array(w, "W"))
    graph = helper.make_graph(nodes, "operator_axes", [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)], outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return rng.normal(size=(5, *input_shape)).astype(np.float32), axis, protected


def session(path, optimize):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    opts.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize
                                    else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def check(src, dst, samples, axis, protected, bits=8, per_channel=True):
    onnx.checker.check_model(str(dst))
    a, b = onnx.load(src), onnx.load(dst)
    before = {t.name: t for t in a.graph.initializer}
    after = {t.name: t for t in b.graph.initializer}
    for name in protected:
        assert before[name].SerializeToString() == after[name].SerializeToString(), name
    dq = [n for n in b.graph.node if n.op_type == "DequantizeLinear"]
    assert len(dq) == 1 and dq[0].output == ["W"]
    attrs = {a.name: helper.get_attribute_value(a) for a in dq[0].attribute}
    assert attrs.get("axis") == (axis if per_channel else None), attrs
    assert numpy_helper.to_array(after["W_scale"]).size == (before["W"].dims[axis] if per_channel else 1)
    assert [v.name for v in a.graph.output] == [v.name for v in b.graph.output]
    reference, plain, optimized = session(src, False), session(dst, False), session(dst, True)
    for sample in samples:
        feed = {"X": sample}
        for expected, x, y in zip(reference.run(None, feed), plain.run(None, feed), optimized.run(None, feed)):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            np.testing.assert_allclose(x, y, atol=1e-5, rtol=1e-4)
            relative_rmse = np.linalg.norm(x - expected) / max(np.linalg.norm(expected), 1e-8)
            assert relative_rmse < (0.03 if bits == 8 else 0.35), relative_rmse


def run(binary, *args, ok=True):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    if (result.returncode == 0) != ok:
        raise AssertionError(result.stdout + result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python", action="store_true", dest="python_api")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    if args.python_api:
        import quantize_rs
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cases = [(name, 0, 0) for name in ["matmul", "batched", "broadcast", "vector", "shared", "embedding"]]
        cases += [("gemm", a, b) for a in [0, 1] for b in [0, 1]]
        for kind, a, b in cases:
            src, dst = root / "source.onnx", root / "quantized.onnx"
            samples, axis, protected = build(src, kind, a, b)
            for flags, bits, pc in [(["--per-channel", "--symmetric"], 8, True),
                                    (["--per-channel"], 8, True),
                                    ([], 8, False),
                                    (["--bits", "4", "--per-channel", "--symmetric"], 4, True),
                                    (["--per-channel", "--symmetric", "--layer-bits", "W=4", "--native-int4"], 4, True)]:
                run(binary, "quantize", src, "-o", dst, *flags)
                check(src, dst, samples, axis, protected, bits, pc)
            if args.python_api:
                quantize_rs.quantize(str(src), str(dst), per_channel=True, symmetric=True)
                check(src, dst, samples, axis, protected)
            print(f"[ok] {kind} transA={a} transB={b}: selection, axes, storage and ORT parity", flush=True)
        # Valid ONNX graphs with unsafe shared-weight uses must fail before save.
        src, dst = root / "conflict.onnx", root / "preserved.onnx"
        samples, _, _ = build(src, "matmul")
        base = onnx.load(src)
        for conflict in ["transpose", "gather", "custom", "output"]:
            model = onnx.ModelProto(); model.CopyFrom(base)
            if conflict == "transpose":
                model.graph.node.append(helper.make_node("Gemm", ["A", "W"], ["Z"], transB=1))
                model.graph.input.append(helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 5]))
            elif conflict == "gather":
                model.graph.initializer.append(numpy_helper.from_array(np.array([0], np.int64), "idx"))
                model.graph.node.append(helper.make_node("Gather", ["W", "idx"], ["Z"]))
            elif conflict == "custom":
                model.opset_import.append(helper.make_opsetid("example", 1))
                model.graph.node.append(helper.make_node("Use", ["W"], ["Z"], domain="example"))
            else:
                model.graph.output.append(helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 5]))
            onnx.checker.check_model(model)
            onnx.save(model, src)
            dst.write_bytes(b"existing output must survive")
            run(binary, "quantize", src, "-o", dst, "--per-channel", ok=False)
            assert dst.read_bytes() == b"existing output must survive"
            if args.python_api:
                try:
                    quantize_rs.quantize(str(src), str(dst), per_channel=True)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Python accepted an unsafe shared initializer")
                assert dst.read_bytes() == b"existing output must survive"
        print("[ok] unsupported shared uses preserve output files in enabled APIs")
    print("OPERATOR AXES SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
