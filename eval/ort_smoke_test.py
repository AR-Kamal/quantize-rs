#!/usr/bin/env python3
"""Self-contained ONNX Runtime execution smoke test for quantize-rs.

Unlike `validate_models.py` (which downloads real models), this builds a tiny
Conv model in-process, quantizes it with the quantize-rs CLI, and then **loads
and runs the quantized model in ONNX Runtime**.  It closes the gap that the Rust
test suite leaves open: those tests verify the QDQ graph is structurally
connected, but never confirm ORT will actually execute it.

It exercises three code paths:
  * per-tensor INT8   → DequantizeLinear with a scalar scale
  * per-channel INT8  → DequantizeLinear with an axis-0 scale vector (opset 13)
  * INT4 (widened)    → INT8-stored INT4 values

Usage
-----
    python eval/ort_smoke_test.py
    python eval/ort_smoke_test.py --binary ./target/release/quantize-rs

Requirements
------------
    pip install onnx onnxruntime numpy

Exits non-zero if any model fails to build, quantize, load, or run, or if an
INT8 output diverges from the FP32 baseline (cosine similarity < 0.99).
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("ERROR: onnx is required.  pip install onnx")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required.  pip install onnxruntime")
    sys.exit(1)


INPUT_SHAPE = (1, 3, 8, 8)  # N, C, H, W


def build_conv_model(path: Path) -> None:
    """Write a minimal single-Conv FP32 model (opset 13) to `path`."""
    rng = np.random.default_rng(0)
    # Conv weight: [out_channels=4, in_channels=3, kh=3, kw=3]
    weight = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(INPUT_SHAPE))
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 6, 6])
    w_init = numpy_helper.from_array(weight, name="conv.weight")

    node = helper.make_node("Conv", ["X", "conv.weight"], ["Y"], name="conv0")

    graph = helper.make_graph([node], "smoke", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # conservative; supported by ORT >= 1.16
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def run_model(path: Path, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"X": x})[0]
    return np.asarray(out, dtype=np.float64).ravel()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def quantize(binary: str, src: Path, dst: Path, extra: list[str]) -> None:
    cmd = [binary, "quantize", str(src), "-o", str(dst), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"quantize-rs failed ({' '.join(cmd)}):\n{proc.stdout}\n{proc.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_bin = "./target/release/quantize-rs" + (
        ".exe" if platform.system() == "Windows" else ""
    )
    parser.add_argument("--binary", default=default_bin, help="path to quantize-rs binary")
    args = parser.parse_args()

    if not Path(args.binary).exists():
        print(f"ERROR: binary not found: {args.binary}")
        print("Build it first:  cargo build --release --bin quantize-rs")
        return 1

    rng = np.random.default_rng(1)
    x = rng.standard_normal(INPUT_SHAPE).astype(np.float32)

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "fp32.onnx"
        build_conv_model(src)
        baseline = run_model(src, x)
        print(f"[ok] FP32 baseline runs in ORT ({baseline.size} output elements)")

        # (label, CLI flags, cosine floor or None for finite-only)
        cases = [
            ("per-tensor INT8", ["--bits", "8"], 0.99),
            ("per-channel INT8", ["--bits", "8", "--per-channel"], 0.99),
            ("INT4 (widened)", ["--bits", "4"], None),
            ("native INT4 (opset 21)", ["--bits", "4", "--native-int4"], None),
        ]

        for i, (label, flags, floor) in enumerate(cases):
            dst = tmp / f"q_{i}.onnx"
            try:
                quantize(args.binary, src, dst, flags)
                out = run_model(dst, x)
            except Exception as exc:  # noqa: BLE001 - smoke test, report and continue
                failures.append(f"{label}: {exc}")
                print(f"[FAIL] {label}: {exc}")
                continue

            if not np.all(np.isfinite(out)):
                failures.append(f"{label}: output contains non-finite values")
                print(f"[FAIL] {label}: non-finite output")
                continue

            if floor is None:
                print(f"[ok] {label}: runs in ORT, output finite")
            else:
                cos = cosine(baseline, out)
                if cos >= floor:
                    print(f"[ok] {label}: runs in ORT, cosine={cos:.4f} (>= {floor})")
                else:
                    failures.append(f"{label}: cosine {cos:.4f} < {floor}")
                    print(f"[FAIL] {label}: cosine {cos:.4f} < {floor}")

    print()
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE TEST PASSED - quantized models load and run in ONNX Runtime.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
