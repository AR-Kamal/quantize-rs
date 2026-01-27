# Examples

This directory contains example code showing how to use quantize-rs.

## Running Examples
```bash
# Basic quantization
cargo run --example basic_quantization

# Batch processing
cargo run --example batch_quantize
```

## Examples

### `basic_quantization.rs`
Shows the fundamental workflow:
1. Load model
2. Extract weights
3. Quantize
4. Save

### `batch_quantize.rs`
Demonstrates processing multiple models in a loop.

## Prerequisites

Make sure you have an ONNX model file (e.g., `mnist.onnx`) in the project root before running examples.

Download sample models:
```bash
# MNIST
curl -L -o mnist.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx

# ResNet-18
curl -L -o resnet18-v1-7.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx
```