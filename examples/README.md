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
cargo run --example activation_calibration
```

## Descriptions

### `basic_quantization.rs`

Minimal INT8 quantization workflow: load model, extract weights, quantize, save, and print compression results.

### `batch_quantize.rs`

Iterates over multiple model files and quantizes each one, skipping any that are not found on disk.

### `activation_calibration.rs`

Full activation-based calibration pipeline. Loads or generates calibration data, runs inference through tract to collect per-layer activation statistics, quantizes using the observed ranges, and saves the result. Accepts `--model`, `--calibration-data`, `--output`, `--bits`, `--per-channel`, and `--shape` arguments.
