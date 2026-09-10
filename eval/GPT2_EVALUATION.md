# GPT-2 weight coverage and INT8 evaluation

The new operator-aware INT8 path passes a bounded WikiText-2 quality gate on the
local GPT-2 export. It quantizes every eligible direct matrix weight and leaves
both embedding tables unchanged. The file is **56.68% smaller**, but median
inference latency is **44.10% higher** on this CPU. Runtime profiling shows
floating-point matrix kernels with weight dequantization; this is not integer
Transformer execution.

## Recorded result

The [machine-readable run](results/gpt2-int8-cpu.json) contains settings, hashes,
environment, memory and runtime kernel counts. The [complete coverage audit](results/gpt2-weight-coverage.json)
lists every initializer, its consumers, selected axis or skip reason.
The separate [export reference check](results/gpt2-export-reference.json) compares
this ONNX file with the pinned GPT-2 checkpoint at sequence lengths 8, 31 and 256.
It passes at `atol=1e-3`, `rtol=1e-4`; the largest absolute logit error is 0.00048828.

| Metric | FP32 | Symmetric per-channel INT8 |
|--------|------|---------------------------|
| Perplexity, fixed 32,768-token prefix | 36.670400 | 37.263413 |
| Relative perplexity increase | baseline | 1.6171% |
| File bytes | 652,613,039 | 282,705,343 |
| Median forward latency | 2,437.53 ms | 3,512.57 ms |
| p95 forward latency | 2,768.96 ms | 3,843.74 ms |
| Process RSS after evaluation | 1,204.33 MiB | 625.20 MiB |
| Peak process RSS through evaluation | 1,495.17 MiB | 867.16 MiB |

The predeclared gate requires FP32 perplexity at most 100 and an INT8 relative
perplexity increase at most 3%. Both pass. These are project regression thresholds,
not a universal quality standard. Timing and memory are observations, not gates.

## Coverage

- 149 serialized initializers: **49 selected**, 100 unchanged. Selection uses
  `min_elements=0`, no exclusions and no layer overrides.
- All 48 Gemm projection weights and the one direct MatMul output weight use
  axis 1. The export has no conflicting shared uses among selected weights.
- Two Gather embedding tables and 98 rank-below-2 tensors remain unchanged;
  their complete protobuf tensor contents are checked by SHA-256 after export.
- Selected weights contain 123,532,032 elements, **75.83%** of the 162,915,840
  elements in serialized FP32 rank-2-or-higher initializers. This denominator
  includes embeddings and duplicated exported weights; it is not the number of
  unique pretrained parameters. The output head is a separate initializer in
  this export even though GPT-2 ties it to the token embedding during training.
- Of 73 matrix operators, 49 have direct initializer weights; the other 24
  MatMul right operands are dynamic/unresolved by the audit. No indirect
  initializer weight paths were found in this export.

The Python audit predicts candidates from direct graph uses. Rust's actual QDQ
output is authoritative: the evaluator requires exact agreement on selected
names, shapes, INT8 types, axis, scale length and symmetric zero points. A
candidate is marked selected only after this verification. Graph outputs must
also retain their complete interface descriptions. This audit is not a general
ONNX validator or a new quantization policy.

## Data and measurement contract

- Source model: the existing local `eval/models/gpt2.onnx`, pinned to SHA-256
  `600e17f4410b425b50d53611b3b6c2ffc88be304dc3677275e50211800adb06b`.
  It has one dynamic rank-2 INT64 token input, one logits output and no KV cache.
  A filename alone does not establish export/checkpoint provenance.
- Dataset: [WikiText-2 raw test Parquet at revision b08601e](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1).
  All 4,358 rows, including empty rows and headings, are joined with two newlines.
  There is no text substitution, filtering, label tuning or activation calibration.
- Tokenizer: GPT-2 vocabulary and merges at
  [revision 607a30d](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e),
  `tokenizers.ByteLevelBPETokenizer`, no prefix space or special tokens.
  Dataset, vocabulary and merges have enforced SHA-256 hashes. The resulting
  full stream has 287,644 tokens; this run evaluates its first 32,768 tokens.
  The joined-full-text hash and evaluated token-array hash are recorded separately.
- Perplexity: 256-token context, 128-token stride, 255 overlapping windows.
  Logits at position `i` predict token `i+1`. Context-only targets are masked;
  exactly **32,767 targets** are scored once each, with stable log-sum-exp and
  f64 accumulation. The approach follows the
  [strided-context evaluation principle](https://huggingface.co/docs/transformers/perplexity).
  Scores depend on corpus slice, tokenizer and context; they are not directly
  comparable with published full-WikiText scores at other context lengths.
- Runtime: Windows 11, Intel Family 6 Model 140, ONNX Runtime 1.22.1,
  CPUExecutionProvider, sequential execution, one intra-op and one inter-op
  thread. Full package versions appear in the JSON report.
- Each variant runs in a fresh process. Latency measures 40 forwards of the same
  prepared `[1,256]` input after accuracy inference and five additional warmups.
  It excludes model loading, tokenization, quantization, loss calculation and
  profiling. It is full-context forward latency, not cached generation latency.
- Memory is whole-process RSS through accuracy and timing, including model load,
  logits and loss computation. It is not isolated tensor memory. Profiling and
  parity sessions are created only after these measurements.
- Optimized/unoptimized logits agree on prefixes of lengths 8, 31 and 256 at
  `atol=1e-3`, `rtol=1e-4`; observed maximum absolute errors are 0.00030518 FP32
  and 0.00043106 INT8. This is a small probe set, not an exhaustive parity proof.

## Runtime finding and next implementation

A separate profiled forward records 48 `Gemm` and 25 `MatMul` kernel executions
with float inputs in both variants. INT8 additionally executes 49
`DequantizeLinear` kernels. These are executed kernel counts from an
[ORT profile](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html),
not merely operator names in the serialized source graph.

The added dequantization and float matrix execution are consistent with the
observed slowdown. This run does not isolate how much time each cause contributes.
The next useful implementation is a **small MatMul/Gemm activation-QDQ prototype**
with integer-fusion and numerical tests. Applying it to GPT-2 also requires an
explicit token-input calibration contract/backend; the existing static path
accepts fixed FP32 NCHW Conv inputs only. Indirect-weight tracing is not the
coverage bottleneck in this particular export.

That [matrix activation prototype](../MATRIX_CALIBRATION.md) is now implemented
for fixed FP32 `[1,K]` inputs, with integer execution demonstrated on simple
graphs. The GPT-2 measurements above remain weight-only; token-input calibration
and full-model validation have not been added by that prototype.

## Reproduce

```bash
python -m pip install numpy onnx onnxruntime tokenizers pyarrow psutil
cargo build --bin quantize-rs
python eval/benchmark_gpt2.py --audit
python eval/benchmark_gpt2.py --evaluate --download
python -m unittest discover -s eval -p test_gpt2_evaluation.py -v
```

Use `--binary /absolute/path/to/quantize-rs.exe` on Windows if needed. Outputs
go to `target/gpt2-evaluation`. Pinned assets are copied from an existing matching
Hugging Face cache or downloaded into that workspace directory with `--download`.
Subsequent runs can omit downloads. Missing or mismatched inputs fail explicitly.
`--audit` writes candidate/skip/conflict details to `coverage-preflight.json`
without changing the completed evaluation report or verified `coverage.json`;
`--quantize` also exports and verifies coverage; `--benchmark` evaluates an
existing INT8 artifact after rechecking its coverage.

The 653 MB local export is not checked into Git or downloaded by the evaluator.
If it is absent, `--export --download` uses the optional pinned export helper
(requires torch and transformers). Different exporter/dependency versions can
produce different graphs and hashes. Review that export, then supply its printed
`--model-sha256` explicitly. Every evaluation records the actual artifact hash;
do not substitute another file under the recorded hash.

With the pinned GPT-2 checkpoint cached locally, the additional provenance check
can be reproduced after preparing tokens:

```bash
python eval/verify_gpt2_export.py
# Or supply --checkpoint-dir /path/to/pinned/snapshot and --model/--model-sha256.
```

It requires torch and transformers, verifies checkpoint/configuration hashes,
and performs no downloads. It checks three fixed prefixes, not every possible
input. It runs separately from the timed/RSS evaluation.

`--max-tokens 0` evaluates the complete test token stream. Other context/stride,
thread count and threshold settings are explicit CLI options recorded in the
report. Do not tune thresholds against the held-out run. INT4 and greedy text
generation from the old demo are outside this INT8 milestone; the old
`--bits`, `--model-dir` and `--project-root` options are replaced by this contract.
Use `--model`, `--binary` and `--output-dir` for paths.

CI/release verification runs the small offline math/coverage tests. The full
GPT-2 evaluation remains a manual gate because it needs a large reviewed export
and substantial CPU time. It does not establish full-corpus quality, other
export layouts, other models, GPU behavior or general Transformer acceleration.
