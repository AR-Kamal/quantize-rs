#!/usr/bin/env python3
"""Evaluate FP32, Conv-weight INT8 and calibrated Conv INT8 on labeled images.

Example (public MNIST CNN, 1,000 training images, full 10,000-image test set):
  python eval/cnn_evaluation.py --mnist --download --binary target/debug/quantize-rs

Custom data: --model fixed.onnx --calibration-data train.npy --test-data test.npy
  --labels labels.npy. Images must already be FP32 [N,C,H,W]; labels are integer
  class indices. The caller must supply disjoint calibration and test examples.
Requires numpy, onnx, onnxruntime and psutil. Downloads/artifacts stay in target/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from ort_utils import CPU_SESSION_CONFIG, cpu_session_options
import psutil

MODEL_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mnist/model/mnist-12.onnx"
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MODEL_SHA = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"
DATA_SHA = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(path, url, digest, download):
    if not path.exists():
        if not download:
            raise ValueError(f"Missing {path}; pass --download or provide the cached file")
        temporary = path.with_suffix(path.suffix + ".download")
        with urllib.request.urlopen(url, timeout=60) as source, open(temporary, "wb") as dest:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
        if sha256(temporary) != digest:
            raise ValueError(f"Checksum mismatch: {url}")
        temporary.replace(path)
    if sha256(path) != digest:
        raise ValueError(f"Checksum mismatch: {path}")


def session(path, threads, optimized=True, save=None):
    options = cpu_session_options()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    if save:
        options.optimized_model_filepath = str(save)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def prepare_mnist(root, download):
    model, dataset = root / "mnist-12.onnx", root / "mnist.npz"
    fetch(model, MODEL_URL, MODEL_SHA, download)
    fetch(dataset, DATA_URL, DATA_SHA, download)
    with np.load(dataset, allow_pickle=False) as data:
        indices = np.random.default_rng(2026).choice(len(data["x_train"]), 1000, replace=False)
        np.save(root / "calibration.npy", data["x_train"][indices, None].astype(np.float32) / 255)
        np.save(root / "test.npy", data["x_test"][:, None].astype(np.float32) / 255)
        np.save(root / "labels.npy", data["y_test"])
    converted = onnx.version_converter.convert_version(onnx.load(model), 13)
    onnx.checker.check_model(converted)
    onnx.save(converted, root / "mnist-13.onnx")
    # Verify conversion against the original before any quantization is judged.
    old, new = session(model, 1), session(root / "mnist-13.onnx", 1)
    images = np.load(root / "test.npy", mmap_mode="r", allow_pickle=False)
    for image in images[:100]:
        feed = {old.get_inputs()[0].name: image[None]}
        np.testing.assert_allclose(old.run(None, feed)[0], new.run(None, feed)[0], rtol=1e-5, atol=1e-5)
    return (root / "mnist-13.onnx", root / "calibration.npy", root / "test.npy",
            root / "labels.npy", {"model_url": MODEL_URL, "model_sha256": MODEL_SHA,
            "dataset_url": DATA_URL, "dataset_sha256": DATA_SHA, "seed": 2026,
            "calibration_split": "1000 samples without replacement from x_train",
            "test_split": "all 10000 x_test samples; no threshold tuning",
            "preprocessing": "uint8 / 255 as FP32 NCHW",
            "conversion": "opset 12 to 13; FP32 parity checked on first 100 test images"})


def peak_rss():
    if os.name == "nt":
        return psutil.Process().memory_info().peak_wset
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def worker(model, images_path, labels_path, output, threads, runs):
    """One model per fresh process; no previous ORT session affects memory."""
    images = np.load(images_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    baseline_rss = psutil.Process().memory_info().rss
    optimized = Path(output).with_suffix(".optimized.onnx")
    runtime = session(model, threads, save=optimized)
    input_name = runtime.get_inputs()[0].name
    if len(runtime.get_outputs()) != 1:
        raise ValueError("Evaluation requires a single classification output")
    correct = 0
    for image, label in zip(images, labels):
        scores = runtime.run(None, {input_name: image[None]})[0].reshape(-1)
        if not np.isfinite(scores).all() or not 0 <= int(label) < scores.size:
            raise ValueError("Nonfinite output or label outside model classes")
        correct += int(np.argmax(scores) == label)
    # Accuracy inference warms every test image. An additional fixed warmup and
    # timed repetitions use prepared inputs; model loading/preprocessing excluded.
    feeds = [{input_name: image[None]} for image in images[:min(runs, len(images))]]
    for i in range(50):
        runtime.run(None, feeds[i % len(feeds)])
    timings = []
    for i in range(runs):
        feed = feeds[i % len(feeds)]
        start = time.perf_counter_ns()
        runtime.run(None, feed)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    # Capture memory before making a second, unoptimized validation session.
    memory = {"baseline_rss_mib": baseline_rss / 2**20,
              "session_rss_mib": psutil.Process().memory_info().rss / 2**20,
              "process_peak_rss_mib": peak_rss() / 2**20}
    unfused = session(model, threads, optimized=False)
    max_error = 0.0
    for image in images[:100]:
        feed = {input_name: image[None]}
        a, b = runtime.run(None, feed)[0], unfused.run(None, feed)[0]
        max_error = max(max_error, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    result = {"correct": correct, "examples": len(labels), "top1_accuracy": correct / len(labels),
              "latency_median_ms": float(np.median(timings)),
              "latency_p95_ms": float(np.percentile(timings, 95)),
              "latency_runs": runs, "model_bytes": Path(model).stat().st_size,
              "model_sha256": sha256(model),
              "optimized_vs_unoptimized_max_abs_error_first_100": max_error,
              "optimized_ops": dict(Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}"
                                            for n in onnx.load(optimized).graph.node)),
              **memory}
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def command(argv):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("target/cnn-evaluation"))
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--download", action="store_true")
    for name in ["model", "calibration-data", "test-data", "labels"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--latency-runs", type=int, default=3000)
    parser.add_argument("--max-accuracy-drop-pp", type=float, default=1.0)
    parser.add_argument("--minimum-baseline-accuracy", type=float, default=0.95)
    args = parser.parse_args()
    if args.threads <= 0 or args.latency_runs <= 0:
        parser.error("threads and latency-runs must be positive")
    if (not np.isfinite(args.max_accuracy_drop_pp) or args.max_accuracy_drop_pp < 0
            or not 0 <= args.minimum_baseline_accuracy <= 1):
        parser.error("Accuracy limits must be finite; drop >= 0, baseline in [0,1]")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mnist:
        if any([args.model, args.calibration_data, args.test_data, args.labels]):
            parser.error("--mnist supplies its own model and data paths")
        model, calibration, images, labels, provenance = prepare_mnist(root, args.download)
    else:
        if not all([args.model, args.calibration_data, args.test_data, args.labels]):
            parser.error("Supply --mnist or all four model/data paths")
        model, calibration, images, labels = [p.resolve(strict=True) for p in
                                             [args.model, args.calibration_data, args.test_data, args.labels]]
        provenance = {"split": "user supplied; caller is responsible for avoiding train/test overlap"}
    graph = onnx.load(model)
    onnx.checker.check_model(graph)
    inputs = [v for v in graph.graph.input if v.name not in {t.name for t in graph.graph.initializer}]
    if len(inputs) != 1:
        raise ValueError("One fixed FP32 NCHW model input is required")
    tensor = inputs[0].type.tensor_type
    shape = [d.dim_value for d in tensor.shape.dim]
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(shape) != 4 or shape[0] != 1 or min(shape) <= 0:
        raise ValueError("One fixed FP32 NCHW model input with batch 1 is required")
    arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in [calibration, images]]
    for array in arrays:
        if array.dtype != np.float32 or array.ndim != 4 or list(array.shape[1:]) != shape[1:] or not len(array):
            raise ValueError("Image data must be nonempty FP32 [N,C,H,W] matching the model")
        if not np.isfinite(array).all():
            raise ValueError("Image data contains nonfinite values")
    truth = np.load(labels, allow_pickle=False)
    if truth.shape != (len(arrays[1]),) or truth.dtype.kind not in "iu":
        raise ValueError("Labels must contain one integer class index per test image")
    conv_weights = {n.input[1] for n in graph.graph.node if n.op_type == "Conv"}
    exclusions = [t.name for t in graph.graph.initializer if t.name not in conv_weights]
    binary = args.binary.resolve(strict=True)
    weights, calibrated = root / "weight-int8.onnx", root / "calibrated-int8.onnx"
    if any(p in [model, calibration, images, labels, binary] for p in [weights, calibrated]):
        raise ValueError("Generated model paths must not overwrite evaluation inputs")
    flags = [flag for name in exclusions for flag in ["--exclude", name]]
    command([binary, "quantize", model, "-o", weights, "--bits", "8", "--per-channel", "--symmetric", *flags])
    command([binary, "calibrate", model, "-o", calibrated, "--data", calibration,
             "--method", "minmax", "--per-channel", "--symmetric", *flags])
    weight_tensors = {t.name: t.SerializeToString() for t in onnx.load(weights).graph.initializer}
    calibrated_tensors = {t.name: t.SerializeToString() for t in onnx.load(calibrated).graph.initializer}
    for name in conv_weights:
        for suffix in ["_quantized", "_scale", "_zp"]:
            if weight_tensors[name + suffix] != calibrated_tensors[name + suffix]:
                raise ValueError(f"Weight quantization differs between variants: {name + suffix}")
    variants = {"fp32": model, "weight_int8": weights, "calibrated_int8": calibrated}
    results = {}
    for name, path in variants.items():
        onnx.checker.check_model(str(path))
        output = root / f"{name}.json"
        command([sys.executable, Path(__file__).resolve(), "_worker", path, images, labels,
                 output, args.threads, args.latency_runs])
        results[name] = json.loads(output.read_text(encoding="utf-8"))
        print(f"{name}: accuracy={results[name]['top1_accuracy']:.4%}, "
              f"median={results[name]['latency_median_ms']:.4f} ms", flush=True)
    baseline = results["fp32"]["top1_accuracy"]
    passed = baseline >= args.minimum_baseline_accuracy
    for result in results.values():
        result["accuracy_drop_pp"] = 100 * (baseline - result["top1_accuracy"])
        passed &= result["accuracy_drop_pp"] <= args.max_accuracy_drop_pp
    report = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance,
              "inputs_sha256": {role: sha256(path) for role, path in
                                zip(["model", "calibration", "test", "labels", "binary"],
                                    [model, calibration, images, labels, binary])},
              "environment": {"platform": platform.platform(), "processor": platform.processor(),
                              "python": platform.python_version(), "onnx": onnx.__version__,
                              "onnxruntime": ort.__version__, "numpy": np.__version__,
                              "provider": "CPUExecutionProvider", "intra_op_threads": args.threads,
                              "session_config": CPU_SESSION_CONFIG},
              "policy": {"weights": "same selected Conv weights in both INT8 variants; other weights FP32",
                         "calibration_samples": len(arrays[0]), "method": "minmax",
                         "minimum_baseline_accuracy": args.minimum_baseline_accuracy,
                         "maximum_accuracy_drop_pp": args.max_accuracy_drop_pp,
                         "memory": "separate process per model; RSS includes Python, data and ORT; not tensor memory",
                         "latency": "warm batch-1 session.run wall time; loading and preprocessing excluded"},
              "results": results, "accuracy_gate_passed": bool(passed)}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Accuracy gate failed; see report.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(*sys.argv[2:6], int(sys.argv[6]), int(sys.argv[7]))
    else:
        main()
