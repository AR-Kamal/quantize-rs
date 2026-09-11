# Labeled CNN calibration evaluation

The unreleased calibration path preserves accuracy on the published MNIST CNN.
It produces smaller files and ONNX Runtime fuses both selected convolutions into
integer operators. It was **slower than FP32** on the measured CPU. These results
support correctness on this task, not a general speed or production-readiness claim.

## Recorded result

The complete machine-readable record, including hashes, environment, timestamp,
memory baselines and optimized operator counts, is [results/mnist-cpu.json](results/mnist-cpu.json).
This is the completed run after this task's Rust compilation stopped.

| Variant | Correct / 10,000 | Top-1 accuracy | Model bytes | Median latency | p95 latency | Peak process RSS |
|---------|------------------|----------------|-------------|----------------|-------------|------------------|
| FP32 | 9,890 | 98.90% | 26,143 | 0.0430 ms | 0.0669 ms | 96.54 MiB |
| Conv weights INT8 | 9,890 | 98.90% | 16,455 | 0.0910 ms | 0.2283 ms | 96.54 MiB |
| Calibrated Conv INT8 | 9,891 | 98.91% | 18,172 | 0.1031 ms | 0.1727 ms | 96.49 MiB |

The one additional correct prediction is not evidence of a meaningful accuracy
improvement. Both quantized variants pass the predeclared gate: FP32 accuracy at
least 95%, with at most 1 percentage point degradation from FP32.

The calibrated optimized graph contains two `com.microsoft::QLinearConv` nodes.
It also contains quantize/dequantize, transpose and layout-reordering nodes.
Integer fusion is confirmed for this runtime configuration; identifying the cause
of slower inference requires deeper profiling. Whole-process RSS is effectively
unchanged at this scale and is dominated by Python, the dataset and runtime.

## Methodology

- Model: the [ONNX model repository's MNIST CNN](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist),
  `mnist-12.onnx`, converted with ONNX's version converter to opset 13. The original
  and converted FP32 models agree on the first 100 test images within `rtol=1e-5`,
  `atol=1e-5`. Input shape is `[1,1,28,28]`; pixel values are divided by 255.
- Data: the [MNIST archive used by Keras](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py).
  Calibration uses 1,000 images sampled without replacement from the 60,000-image
  training split, NumPy seed 2026. Accuracy uses all 10,000 separate test images.
  Test labels do not choose the calibration range or tune thresholds.
- Selection: both INT8 variants quantize the same two Conv weight initializers,
  with symmetric axis-0 per-channel weights. Other weights remain FP32. Serialized
  weight values/scales/zero points are checked to be identical between variants.
  The calibrated variant additionally uses scalar asymmetric INT8 activation QDQ
  with MinMax ranges. The evaluator does not claim full-network INT8 conversion.
- Execution: Windows 11, Intel Family 6 Model 140, Python 3.13.7, ONNX 1.20.1,
  ONNX Runtime 1.22.1 CPUExecutionProvider, NumPy 2.4.2. Sequential execution,
  one intra-op thread and one inter-op thread, all graph optimizations enabled.
- Latency: after full accuracy inference and 50 additional warmups, measure
  3,000 batch-1 `session.run` calls on prepared images. Loading, preprocessing,
  downloading and offline quantization are excluded. The Rust CLI used a debug
  build; inference timing measures ONNX Runtime, not Rust quantization throughput.
- Memory: each model runs in a fresh process. Record RSS before creating the
  session, after inference, and peak process RSS. This includes the Python
  interpreter, data and ORT; it is not isolated model/tensor memory.
- Optimization parity: compare optimized and unoptimized outputs on the first
  100 test images (`rtol=1e-4`, `atol=1e-4`). Maximum absolute errors were below
  `1.6e-5` for each variant. The unoptimized session is created after recording
  memory to avoid counting two sessions.

This is a small real digit-recognition CNN on one workstation. It does not
establish ImageNet accuracy, Transformer support, GPU performance, or universal
latency/memory savings. Timing is subject to scheduling and power-state variation;
the evaluation gates accuracy, not latency or RSS. Other calibration methods have
synthetic regressions but were not evaluated on this labeled run.

## CPU precision in subsequent runs

The evaluation now sets `session.x64quantprecision=1` through `ort_utils.py`
and records it in `environment.session_config`. This avoids intermediate integer
saturation on x64 CPUs without VNNI while retaining graph optimization. See
[CPU runtime precision](../CALIBRATION.md#cpu-runtime-precision). The historical
measurements above and their checked-in JSON report are unchanged; compare new
latency measurements only with their recorded runtime and session settings.

## Reproduce

```bash
python -m pip install numpy onnx onnxruntime psutil
cargo build --bin quantize-rs
python eval/cnn_evaluation.py --mnist --download --binary target/debug/quantize-rs
```

Use `target/debug/quantize-rs.exe` on Windows. Downloads are checksum-pinned and
cached under `target/cnn-evaluation`; later runs can omit `--download`. The script
writes `report.json`, per-model JSON records, and generated/optimized models there.
Optimized models are inspection artifacts specific to the measured runtime.
CI and release verification run the same accuracy gate; CI uploads the JSON report.
Run timing measurements after unrelated CPU-intensive work has stopped.

For a deployment CNN and its preprocessed representative data:

```bash
python eval/cnn_evaluation.py --binary target/debug/quantize-rs \
  --model classifier.onnx --calibration-data calibration.npy \
  --test-data test.npy --labels labels.npy --output-dir target/deployment-evaluation
```

The model must satisfy the static Conv contract. Images must be finite FP32 arrays
`[N,C,H,W]` matching its fixed input, and labels one integer class index per test
image. Provide disjoint calibration/test samples. Adjust the documented accuracy
thresholds before evaluation if the task needs a different acceptance criterion.

The model repository lists the model under MIT. MNIST attribution and licensing
are described by the linked Keras dataset source (Yann LeCun and Corinna Cortes,
CC BY-SA 3.0). Downloaded weights/images are not committed to this repository.
