"""Pinned, bounded GPT-2 INT8 evaluation. See GPT2_EVALUATION.md for the contract."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
RECORDED_MODEL_SHA256 = "600e17f4410b425b50d53611b3b6c2ffc88be304dc3677275e50211800adb06b"
ASSETS = {
    "test.parquet": ("datasets/Salesforce/wikitext", DATA_REVISION, "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                     "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91", "datasets--wikitext"),
    "vocab.json": ("openai-community/gpt2", MODEL_REVISION, "vocab.json",
                   "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783", "models--gpt2"),
    "merges.txt": ("openai-community/gpt2", MODEL_REVISION, "merges.txt",
                   "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5", "models--gpt2"),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def asset(root, name, download=False):
    repo, revision, filename, expected, legacy_cache = ASSETS[name]
    destination = root / name
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    if not destination.exists():
        cache = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
        canonical_cache = ("datasets--" if repo.startswith("datasets/") else "models--") + repo.removeprefix("datasets/").replace("/", "--")
        candidates = [cache / entry / "snapshots" / revision / filename for entry in (legacy_cache, canonical_cache)]
        existing = next((p for p in candidates if p.is_file()), None)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(destination.suffix + ".download")
        if existing:
            shutil.copyfile(existing, temp)
        elif download:
            with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
                shutil.copyfileobj(response, out)
        else:
            raise ValueError(f"Missing pinned {name}; pass --download or place it at {destination}. No fallback corpus is used.")
        if sha256(temp) != expected:
            raise ValueError(f"Checksum mismatch for {name}")
        temp.replace(destination)
    if sha256(destination) != expected:
        raise ValueError(f"Checksum mismatch for {destination}")
    return destination, dict(url=url, revision=revision, sha256=expected)


def prepare_tokens(root, max_tokens, download=False):
    import pyarrow.parquet as pq
    from tokenizers import ByteLevelBPETokenizer
    assets = {name: asset(root / "assets", name, download) for name in ASSETS}
    rows = pq.read_table(assets["test.parquet"][0], columns=["text"])["text"].to_pylist()
    text = "\n\n".join(rows)  # Preserve empty rows, headings and punctuation.
    tokenizer = ByteLevelBPETokenizer(str(assets["vocab.json"][0]), str(assets["merges.txt"][0]), add_prefix_space=False)
    all_ids = tokenizer.encode(text, add_special_tokens=False).ids
    if max_tokens < 0 or (max_tokens and len(all_ids) < max_tokens):
        raise ValueError("Requested token prefix is invalid or exceeds the fixed test corpus")
    ids = np.asarray(all_ids[:max_tokens] if max_tokens else all_ids, dtype=np.int64)
    if len(ids) < 2:
        raise ValueError("At least two tokens are required")
    path = root / "tokens.npy"
    np.save(path, ids, allow_pickle=False)
    return path, dict(dataset="Salesforce/wikitext", configuration="wikitext-2-raw-v1", split="test",
                      assets={name: value[1] for name, value in assets.items()}, rows=len(rows),
                      text_join="all rows joined with two newlines; no stripping/filtering; no special tokens",
                      full_test_tokens=len(all_ids), evaluated_prefix_tokens=len(ids),
                      token_ids_sha256=sha256(path), full_test_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      tokenizer="tokenizers.ByteLevelBPETokenizer; add_prefix_space=False")


def windows(ids, context, stride):
    """Yield context windows and the local first target; score each token after #0 once."""
    if len(ids) < 2 or not 1 <= stride < context:
        raise ValueError("Need >=2 tokens and 1 <= stride < context")
    previous_end = 1
    for start in range(0, len(ids) - 1, stride):
        end = min(start + context, len(ids))
        first_target = max(previous_end, start + 1) - start
        yield ids[start:end], first_target
        previous_end = end
        if end == len(ids):
            break


def negative_log_likelihood(logits, ids, first_target):
    if logits.ndim != 3 or logits.shape[:2] != (1, len(ids)) or not 1 <= first_target < len(ids):
        raise ValueError("Invalid logits shape or target offset")
    if not np.isfinite(logits).all() or np.any(ids < 0) or np.any(ids >= logits.shape[2]):
        raise ValueError("Nonfinite logits or token outside vocabulary")
    labels = ids[first_target:]
    scores = logits[0, first_target - 1:-1].astype(np.float64)
    maxima = scores.max(axis=-1)
    log_z = maxima + np.log(np.exp(scores - maxima[:, None]).sum(axis=-1))
    return float((log_z - scores[np.arange(len(labels)), labels]).sum()), len(labels)


def session(path, threads, optimize=True, profile=None):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3
    if profile:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile)
    runtime = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    inputs, outputs = runtime.get_inputs(), runtime.get_outputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(int64)" or len(inputs[0].shape) != 2 or len(outputs) != 1:
        raise ValueError("Requires one rank-2 INT64 token input and one logits output; no KV cache")
    return runtime, inputs[0].name


def peak_rss():
    import psutil
    if os.name == "nt":
        return psutil.Process().memory_info().peak_wset
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def worker(config_path):
    """One variant per fresh process; profiling/parity happen after memory measurements."""
    import psutil
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ids = np.load(cfg["tokens"], allow_pickle=False)
    model, output = Path(cfg["model"]), Path(cfg["output"])
    baseline = psutil.Process().memory_info().rss
    runtime, input_name = session(model, cfg["threads"])
    total_nll, total_targets, window_count = 0.0, 0, 0
    for tokens, first_target in windows(ids, cfg["context"], cfg["stride"]):
        logits = runtime.run(None, {input_name: tokens[None]})[0]
        nll, count = negative_log_likelihood(logits, tokens, first_target)
        total_nll += nll
        total_targets += count
        window_count += 1
        if window_count % 32 == 0:
            print(f"[{model.stem}] {total_targets}/{len(ids)-1} targets", flush=True)
    if total_targets != len(ids) - 1:
        raise ValueError("Some next-token targets were missed or counted twice")
    del logits
    feed = {input_name: ids[:cfg["context"]][None]}
    for _ in range(5):
        runtime.run(None, feed)
    timings = []
    for _ in range(cfg["latency_runs"]):
        start = time.perf_counter_ns()
        runtime.run(None, feed)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    memory = dict(baseline_rss_mib=baseline / 2**20, session_rss_mib=psutil.Process().memory_info().rss / 2**20,
                  process_peak_rss_mib=peak_rss() / 2**20)
    probes = [ids[:min(n, len(ids))][None] for n in (8, 31, cfg["context"])]
    expected = [runtime.run(None, {input_name: probe})[0] for probe in probes]
    del runtime
    gc.collect()
    plain, input_name = session(model, cfg["threads"], optimize=False)
    max_error = 0.0
    for probe, a in zip(probes, expected):
        b = plain.run(None, {input_name: probe})[0]
        max_error = max(max_error, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, atol=1e-3, rtol=1e-4)
    del plain, expected, a, b
    gc.collect()
    profiled, input_name = session(model, cfg["threads"], profile=output.with_suffix(".ort-profile"))
    profiled.run(None, {input_name: probes[-1]})
    profile_path = Path(profiled.end_profiling())
    events = json.loads(profile_path.read_text())
    kernels = [e for e in events if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time")]
    if not kernels:
        raise ValueError("ORT profile contains no kernel events")
    matrix_kernels = [e for e in kernels if any(s in e.get("args", {}).get("op_name", "") for s in ("MatMul", "Gemm", "Attention"))]
    result = dict(model_sha256=sha256(model), model_bytes=model.stat().st_size,
                  perplexity=math.exp(total_nll / total_targets), negative_log_likelihood_sum=total_nll,
                  scored_targets=total_targets, windows=window_count,
                  latency_median_ms=float(np.median(timings)), latency_p95_ms=float(np.percentile(timings, 95)),
                  latency_runs=len(timings), latency_input_shape=list(probes[-1].shape),
                  optimized_vs_unoptimized_max_abs_error=max_error, parity_atol=1e-3, parity_rtol=1e-4,
                  profiled_kernel_counts=dict(Counter(e["args"]["op_name"] for e in kernels)),
                  profiled_matrix_input_types=dict(Counter(e["args"]["op_name"] + ":" + ",".join(sorted({key for inp in e["args"].get("input_type_shape", []) for key in inp})) for e in matrix_kernels)),
                  profile_sha256=sha256(profile_path), **memory)
    write_json(output, result)
    print(f"[{model.stem}] perplexity={result['perplexity']:.6f}; median={result['latency_median_ms']:.2f} ms", flush=True)


def quality_gate(fp32, int8, max_increase=3.0, max_baseline=100.0):
    baseline, quantized = fp32["perplexity"], int8["perplexity"]
    if not (math.isfinite(baseline) and math.isfinite(quantized) and baseline > 0 and quantized > 0):
        raise ValueError("Perplexity must be finite and positive")
    change = (quantized / baseline - 1) * 100
    return dict(passed=baseline <= max_baseline and change <= max_increase,
                relative_perplexity_increase_percent=change, maximum_increase_percent=max_increase,
                maximum_baseline_perplexity=max_baseline,
                scope="bounded regression thresholds chosen before measurement; not universal quality guarantees")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/quantize-rs" + (".exe" if os.name == "nt" else "")))
    parser.add_argument("--model", type=Path, default=Path("eval/models/gpt2.onnx"))
    parser.add_argument("--model-sha256", default=RECORDED_MODEL_SHA256, help="Expected FP32 ONNX hash; explicitly override for another export")
    parser.add_argument("--output-dir", type=Path, default=Path("target/gpt2-evaluation"))
    parser.add_argument("--download", action="store_true", help="Fetch checksum-pinned corpus/tokenizer assets if absent")
    parser.add_argument("--evaluate", "--all", action="store_true", help="Audit, quantize and evaluate FP32 versus symmetric per-channel INT8")
    parser.add_argument("--audit", action="store_true", help="Write preflight coverage only")
    parser.add_argument("--quantize", action="store_true", help="Audit, quantize and verify output coverage")
    parser.add_argument("--benchmark", action="store_true", help="Audit and evaluate the existing INT8 artifact; verifies coverage")
    parser.add_argument("--export", action="store_true", help="Export pinned GPT-2; prints hash for a subsequent evaluation")
    parser.add_argument("--min-elements", type=int, default=0)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=32768, help="Fixed test prefix; 0 evaluates the complete test token stream")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--latency-runs", type=int, default=40)
    parser.add_argument("--max-ppl-increase-percent", type=float, default=3.0)
    parser.add_argument("--max-baseline-ppl", type=float, default=100.0)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not any((args.evaluate, args.audit, args.quantize, args.benchmark, args.export)):
        parser.print_help()
        return
    if args.export:
        from export_gpt2 import export_gpt2
        if args.model.name != "gpt2.onnx":
            parser.error("--export uses a model path named gpt2.onnx")
        exported = export_gpt2(args.model.parent, args.download)
        print(f"Export SHA256: {sha256(exported)}", flush=True)
        if not any((args.evaluate, args.audit, args.quantize, args.benchmark)):
            return
    if (not 1 <= args.stride < args.context <= 1024 or args.threads < 1 or args.latency_runs < 1
            or args.min_elements < 0 or args.max_tokens < 0
            or not math.isfinite(args.max_ppl_increase_percent) or args.max_ppl_increase_percent < 0
            or not math.isfinite(args.max_baseline_ppl) or args.max_baseline_ppl <= 0):
        parser.error("Invalid evaluation dimensions, run counts or thresholds")
    model = args.model.resolve(strict=True)
    if sha256(model) != args.model_sha256:
        raise ValueError("FP32 ONNX checksum mismatch; provide the reviewed export's --model-sha256 explicitly")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    quantized = root / "gpt2_int8.onnx"
    if model == quantized:
        raise ValueError("Source model and generated INT8 destination must differ")
    # A failed new attempt must not leave a previous passed report looking current.
    if args.evaluate or args.quantize or args.benchmark:
        write_json(root / "report.json", dict(status="preflight", timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                             model_sha256=args.model_sha256))
    import onnx
    from weight_coverage import audit, verify_export
    source = onnx.load(model, load_external_data=False)
    if any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL for t in source.graph.initializer):
        raise ValueError("This evaluator requires inline tensors")
    onnx.checker.check_model(source)
    coverage = audit(source, args.min_elements, args.exclude)
    coverage.update(model_sha256=args.model_sha256, min_elements=args.min_elements, excluded=args.exclude)
    write_json(root / "coverage-preflight.json", coverage)
    print("[coverage] " + json.dumps(coverage["summary"]), flush=True)
    if args.audit and not any((args.evaluate, args.quantize, args.benchmark)):
        return
    if any(r["status"] == "blocked" for r in coverage["initializers"]):
        raise ValueError("Blocked weights found; inspect coverage-preflight.json. No automatic exclusions are applied.")
    binary = args.binary.resolve(strict=True) if args.evaluate or args.quantize else None
    if args.evaluate or args.quantize:
        command = [str(binary), "quantize", str(model), "-o", str(quantized), "--bits", "8", "--per-channel", "--symmetric", "--min-elements", str(args.min_elements)]
        for name in args.exclude:
            command += ["--exclude", name]
        with (root / "quantize.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    output = onnx.load(quantized)
    verify_export(coverage, source, output)
    write_json(root / "coverage.json", coverage)
    print("[coverage verified] " + json.dumps(coverage["summary"]), flush=True)
    del source, output
    gc.collect()
    if not (args.evaluate or args.benchmark):
        return
    tokens, provenance = prepare_tokens(root, args.max_tokens, args.download)
    config = dict(tokens=str(tokens), context=args.context, stride=args.stride, threads=args.threads, latency_runs=args.latency_runs)
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  source_model=dict(sha256=args.model_sha256, provenance="local ONNX export; hash-pinned; export provenance is not inferred from filename"),
                  data=provenance, settings={**config, "tokens": "tokens.npy", "bits": 8, "per_channel": True, "symmetric": True,
                                             "min_elements": args.min_elements, "excluded": args.exclude, "provider": "CPUExecutionProvider", "inter_op_threads": 1,
                                             "latency_scope": "batch-1 full forward at fixed context, no KV cache; preprocessing/loading/profiling excluded; 5 warmups"},
                  thresholds=dict(max_ppl_increase_percent=args.max_ppl_increase_percent, max_baseline_ppl=args.max_baseline_ppl),
                  binary_sha256=sha256(binary) if binary else None,
                  quantization_performed_in_this_run=bool(binary),
                  coverage_summary=coverage["summary"], coverage_sha256=sha256(root / "coverage.json"),
                  environment=dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                                   packages={name: importlib.metadata.version(name) for name in ("numpy", "onnx", "onnxruntime", "tokenizers", "pyarrow", "psutil")}),
                  implementation_sha256={p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("weight_coverage.py"))},
                  variants={}, status="running")
    write_json(root / "report.json", report)
    for name, path in (("fp32", model), ("int8", quantized)):
        config_path, result_path = root / f"{name}.config.json", root / f"{name}.json"
        write_json(config_path, {**config, "model": str(path), "output": str(result_path)})
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)], check=True, timeout=7200)
        except (subprocess.SubprocessError, OSError) as error:
            report["status"] = "execution_failed"
            report["error"] = str(error)
            write_json(root / "report.json", report)
            raise
        report["variants"][name] = json.loads(result_path.read_text(encoding="utf-8"))
        write_json(root / "report.json", report)
    report["gate"] = quality_gate(report["variants"]["fp32"], report["variants"]["int8"], args.max_ppl_increase_percent, args.max_baseline_ppl)
    report["status"] = "passed" if report["gate"]["passed"] else "failed"
    write_json(root / "report.json", report)
    print(json.dumps(report["gate"], indent=2), flush=True)
    if not report["gate"]["passed"]:
        raise SystemExit("GPT-2 accuracy gate failed; see report.json")


if __name__ == "__main__":
    main()
