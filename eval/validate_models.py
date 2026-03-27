#!/usr/bin/env python3
"""
Real-world model validation for quantize-rs.

Downloads standard ONNX models, quantizes them with the quantize-rs CLI binary,
loads both original and quantized models in ONNX Runtime, runs inference with
the same random input, and compares outputs.

Usage
-----
    # Run all validations (downloads models on first run, ~200 MB total)
    python eval/validate_models.py

    # Run a specific model only
    python eval/validate_models.py --model resnet18

    # Skip download (if models are already cached)
    python eval/validate_models.py --no-download

    # Use a custom quantize-rs binary path
    python eval/validate_models.py --binary ./target/release/quantize-rs

Requirements
------------
    pip install onnxruntime numpy

Run from the quantize-rs project root so the binary path resolves correctly.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required.  pip install onnxruntime")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification for a test model."""
    name: str
    url: str
    input_shape: dict[str, tuple[int, ...]]
    # Minimum cosine similarity between FP32 and quantized outputs.
    # INT4 threshold is intentionally low — naive per-tensor INT4 without
    # calibration data is inherently very lossy on large models.
    min_cosine_int8: float = 0.95
    min_cosine_int4: float = 0.30


MODELS: dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        name="ResNet-18",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx",
        input_shape={"data": (1, 3, 224, 224)},
    ),
    "mobilenetv2": ModelSpec(
        name="MobileNetV2",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        input_shape={"input": (1, 3, 224, 224)},
    ),
    "squeezenet": ModelSpec(
        name="SqueezeNet-1.0",
        url="https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx",
        input_shape={"data_0": (1, 3, 224, 224)},
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PROJECT_ROOT / "eval" / "models"


def default_binary() -> str:
    """Locate the quantize-rs binary."""
    ext = ".exe" if platform.system() == "Windows" else ""
    for profile in ("release", "debug"):
        p = PROJECT_ROOT / "target" / profile / f"quantize-rs{ext}"
        if p.exists():
            return str(p)
    return f"quantize-rs{ext}"


def download_model(spec: ModelSpec) -> Path:
    """Download a model to the cache directory, skip if already present."""
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    filename = spec.url.split("/")[-1]
    dest = MODEL_CACHE / filename
    if dest.exists():
        print(f"  [cached] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"  Downloading {spec.name}...")
    urllib.request.urlretrieve(spec.url, dest)
    print(f"  Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def quantize_model(
    binary: str, input_path: Path, output_path: Path,
    bits: int = 8, per_channel: bool = False, min_elements: int = 0,
) -> bool:
    """Run quantize-rs CLI to produce a quantized model. Returns True on success."""
    cmd = [
        binary, "quantize",
        str(input_path),
        "-o", str(output_path),
        "--bits", str(bits),
    ]
    if per_channel:
        cmd.append("--per-channel")
    if min_elements > 0:
        cmd.extend(["--min-elements", str(min_elements)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    QUANTIZE FAILED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[:5]:
                print(f"    stderr: {line}")
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-5:]:
                print(f"    stdout: {line}")
        return False
    return True


def run_inference(model_path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run ONNX Runtime inference and return outputs."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, inputs)
    return outputs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(dot / (norm_a * norm_b))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def top_k_match(a: np.ndarray, b: np.ndarray, k: int = 5) -> bool:
    """Check if the top-K indices match (for classification models)."""
    if a.ndim < 1 or a.size < k:
        return True
    a_flat = a.flatten()
    b_flat = b.flatten()
    top_a = set(np.argsort(a_flat)[-k:])
    top_b = set(np.argsort(b_flat)[-k:])
    return len(top_a & top_b) >= 1  # at least top-1 overlap


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    model: str
    config: str
    success: bool
    ort_loads: bool
    cosine: float
    max_error: float
    top5_match: bool
    compression: float
    error_msg: str = ""


def validate_config(
    spec: ModelSpec,
    model_path: Path,
    binary: str,
    bits: int,
    per_channel: bool,
    min_elements: int,
) -> ValidationResult:
    """Validate a single model + quantization config."""
    config_str = f"INT{bits}" + (" per-ch" if per_channel else "")
    min_cosine = spec.min_cosine_int8 if bits == 8 else spec.min_cosine_int4

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / f"quantized_int{bits}.onnx"

        # Step 1: Quantize
        ok = quantize_model(binary, model_path, output_path, bits, per_channel, min_elements)
        if not ok:
            return ValidationResult(
                spec.name, config_str, False, False,
                0.0, 0.0, False, 0.0, "Quantization CLI failed",
            )

        # Step 2: Check ONNX Runtime loads it
        try:
            sess_q = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        except Exception as e:
            return ValidationResult(
                spec.name, config_str, False, False,
                0.0, 0.0, False, 0.0, f"ORT failed to load: {e}",
            )

        # Step 3: Run inference on both models with same input
        np.random.seed(42)
        inputs = {}
        for name, shape in spec.input_shape.items():
            inputs[name] = np.random.randn(*shape).astype(np.float32)

        try:
            fp32_out = run_inference(model_path, inputs)
        except Exception as e:
            return ValidationResult(
                spec.name, config_str, False, True,
                0.0, 0.0, False, 0.0, f"FP32 inference failed: {e}",
            )

        try:
            q_inputs = {}
            for inp in sess_q.get_inputs():
                if inp.name in inputs:
                    q_inputs[inp.name] = inputs[inp.name]
            quant_out = sess_q.run(None, q_inputs)
        except Exception as e:
            return ValidationResult(
                spec.name, config_str, False, True,
                0.0, 0.0, False, 0.0, f"Quantized inference failed: {e}",
            )

        # Step 4: Compare outputs
        cos = cosine_similarity(fp32_out[0], quant_out[0])
        mae = max_abs_error(fp32_out[0], quant_out[0])
        top5 = top_k_match(fp32_out[0], quant_out[0])

        has_nan = any(np.isnan(o).any() for o in quant_out)
        has_inf = any(np.isinf(o).any() for o in quant_out)

        orig_size = model_path.stat().st_size
        quant_size = output_path.stat().st_size
        compression = orig_size / max(quant_size, 1)

        # Step 5: Determine pass/fail
        errors = []
        if has_nan:
            errors.append("Output contains NaN")
        if has_inf:
            errors.append("Output contains Inf")
        if cos < min_cosine:
            errors.append(f"Cosine {cos:.4f} < threshold {min_cosine}")

        return ValidationResult(
            model=spec.name,
            config=config_str,
            success=len(errors) == 0,
            ort_loads=True,
            cosine=cos,
            max_error=mae,
            top5_match=top5,
            compression=compression,
            error_msg="; ".join(errors),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate quantize-rs on real ONNX models")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run a single model only")
    parser.add_argument("--no-download", action="store_true", help="Skip downloads, use cached models")
    parser.add_argument("--binary", default=default_binary(), help="Path to quantize-rs binary")
    parser.add_argument("--bits", type=int, choices=[4, 8], help="Test a single bit width only")
    args = parser.parse_args()

    models = {args.model: MODELS[args.model]} if args.model else MODELS

    print("=" * 70)
    print("quantize-rs Real-World Model Validation")
    print("=" * 70)
    print(f"Binary:  {args.binary}")
    print(f"Models:  {', '.join(m.name for m in models.values())}")
    print(f"Cache:   {MODEL_CACHE}")
    print()

    # Download models
    if not args.no_download:
        print("Downloading models...")
        model_paths: dict[str, Path] = {}
        for key, spec in models.items():
            try:
                model_paths[key] = download_model(spec)
            except Exception as e:
                print(f"  FAILED to download {spec.name}: {e}")
        print()
    else:
        model_paths = {}
        for key, spec in models.items():
            filename = spec.url.split("/")[-1]
            p = MODEL_CACHE / filename
            if p.exists():
                model_paths[key] = p
            else:
                print(f"  SKIP {spec.name} (not cached)")

    # Run validations
    configs = []
    if args.bits is None or args.bits == 8:
        configs.append({"bits": 8, "per_channel": False, "min_elements": 0})
        configs.append({"bits": 8, "per_channel": True, "min_elements": 0})
    if args.bits is None or args.bits == 4:
        configs.append({"bits": 4, "per_channel": False, "min_elements": 128})

    results: list[ValidationResult] = []

    for key, spec in models.items():
        if key not in model_paths:
            continue
        model_path = model_paths[key]

        print(f"--- {spec.name} ({model_path.stat().st_size / 1e6:.1f} MB) ---")
        for cfg in configs:
            label = f"INT{cfg['bits']}" + (" per-ch" if cfg["per_channel"] else "")
            print(f"  [{label}] ", end="", flush=True)

            t0 = time.time()
            result = validate_config(spec, model_path, args.binary, **cfg)
            elapsed = time.time() - t0

            status = "PASS" if result.success else "FAIL"
            print(
                f"{status}  cosine={result.cosine:.4f}  "
                f"max_err={result.max_error:.4f}  "
                f"top5={result.top5_match}  "
                f"compress={result.compression:.2f}x  "
                f"({elapsed:.1f}s)"
            )
            if result.error_msg:
                print(f"         {result.error_msg}")

            results.append(result)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    for r in results:
        icon = "PASS" if r.success else "FAIL"
        print(f"  [{icon}] {r.model:20s}  {r.config:12s}  cosine={r.cosine:.4f}  compress={r.compression:.2f}x")
        if r.error_msg:
            print(f"         {r.error_msg}")

    print()
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} configurations")
    print()

    if failed > 0:
        print("VALIDATION FAILED")
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
