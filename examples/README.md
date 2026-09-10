# Examples

Runnable examples demonstrating quantize-rs library usage.

## Prerequisites

Place one or more ONNX model files in the project root before running:

```bash
# MNIST
curl -L -o mnist.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx

# ResNet-18
curl -L -o resnet18-v1-7.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx
```

## Running

```bash
cargo run --example basic_quantization
cargo run --example batch_quantize
cargo run --example activation_calibration -- --model model.onnx --calibration-data samples.npy --per-channel
cargo run --example validate_real_model -- --bits 8 --per-channel model.onnx
```

## Descriptions

### `basic_quantization.rs`

Minimal INT8 quantization workflow: load a model, select direct Conv/MatMul/Gemm weights, quantize per-channel with inferred axes, save, and print compression results.

### `batch_quantize.rs`

Iterates over multiple model files and quantizes each one using the same model-aware selection and axes, skipping any that are not found on disk.

### `activation_calibration.rs`

Static INT8 Conv calibration using representative `.npy` samples. Requires
opset >= 13 and one fixed FP32 NCHW input with batch 1. Weights use their own
ranges, and activation QDQ uses observed tensor statistics. Accepts `--model`,
`--calibration-data`, `--output`, `--bits 8`, and `--per-channel`. Missing data is
an error; the example no longer generates random samples or promises an accuracy
improvement. See [CALIBRATION.md](../CALIBRATION.md).

### `validate_real_model.rs`

Loads an ONNX file, quantizes selected direct Conv/MatMul/Gemm weights, and reports per-tensor mean squared error (MSE) and compression ratio. Optionally saves the quantized model and validates the resulting QDQ graph. Useful for checking weight reconstruction quality without running inference. Accepts `--bits`, `--per-channel`, `--min-elements`, and `--output`. See [selection rules and axes](../OPERATOR_QUANTIZATION.md).
