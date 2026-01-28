#!/bin/bash
mkdir -p test_models

# Download MNIST
echo "Downloading MNIST model..."
curl -L -o test_models/mnist.onnx \
  https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx

# Download ResNet-18
echo "Downloading ResNet-18 model..."
curl -L -o test_models/resnet18-v1-7.onnx \
  https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx

echo "✓ Test models downloaded!"