"""
quantize-rs Python Examples
============================

Comprehensive examples demonstrating how to use quantize-rs for
ONNX model quantization across different scenarios.

Install:
    pip install quantization-rs

Build from source:
    pip install maturin
    maturin develop --release --features python
"""

import quantize_rs
import numpy as np
import os


# ---------------------------------------------------------------------------
# 1. Model Inspection
# ---------------------------------------------------------------------------

def inspect_model(model_path: str):
    """Inspect an ONNX model before quantization."""
    info = quantize_rs.model_info(model_path)
    print(f"Model:   {info.name}")
    print(f"Version: {info.version}")
    print(f"Nodes:   {info.num_nodes}")
    print(f"Inputs:  {info.inputs}")
    print(f"Outputs: {info.outputs}")
    return info


# ---------------------------------------------------------------------------
# 2. Basic Quantization
# ---------------------------------------------------------------------------

def basic_int8(input_path: str, output_path: str):
    """Simplest case: INT8 per-tensor quantization.

    Good default for most models. Typically reduces file size ~4x
    with minimal accuracy loss.
    """
    quantize_rs.quantize(input_path, output_path, bits=8)
    print(f"INT8 quantized: {output_path}")


def basic_int4(input_path: str, output_path: str):
    """INT4 quantization for maximum compression.

    ~8x file size reduction. Higher accuracy loss than INT8,
    best suited for models where size matters more than precision
    (e.g., mobile/edge deployment).
    """
    quantize_rs.quantize(input_path, output_path, bits=4)
    print(f"INT4 quantized: {output_path}")


def per_channel_int8(input_path: str, output_path: str):
    """Per-channel INT8 quantization.

    Computes separate scale and zero-point for each output channel.
    Better accuracy than per-tensor when channels have different
    weight distributions (common in convolutional layers).
    Requires ONNX opset >= 13 (automatically upgraded).
    """
    quantize_rs.quantize(input_path, output_path, bits=8, per_channel=True)
    print(f"Per-channel INT8 quantized: {output_path}")


# ---------------------------------------------------------------------------
# 3. Selective Quantization
# ---------------------------------------------------------------------------

def exclude_sensitive_layers(input_path: str, output_path: str):
    """Skip quantization on accuracy-sensitive layers.

    First and last layers often benefit from staying in FP32.
    Use model_info() or inspect the ONNX graph to find layer names.
    """
    quantize_rs.quantize(
        input_path,
        output_path,
        bits=8,
        excluded_layers=[
            "conv1.weight",       # first conv — keeps input fidelity
            "fc.weight",          # final classifier — keeps output precision
        ],
    )
    print(f"Quantized with excluded layers: {output_path}")


def skip_small_tensors(input_path: str, output_path: str):
    """Skip tensors with fewer than N elements.

    Small tensors (biases, 1x1 projections) have few values, so
    quantization noise is relatively high. Keeping them in FP32
    costs negligible extra size but preserves accuracy.
    """
    quantize_rs.quantize(
        input_path,
        output_path,
        bits=8,
        min_elements=512,  # skip tensors with < 512 elements
    )
    print(f"Quantized (small tensors skipped): {output_path}")


def mixed_precision(input_path: str, output_path: str):
    """Different bit widths per layer.

    Use INT8 on sensitive layers and INT4 on the rest for an
    optimal size-vs-accuracy tradeoff. layer_bits overrides the
    global bits setting for specific layers.
    """
    quantize_rs.quantize(
        input_path,
        output_path,
        bits=4,  # default: INT4 for most layers
        layer_bits={
            "conv1.weight": 8,    # INT8 for first conv
            "fc.weight": 8,       # INT8 for classifier head
            "layer4.1.conv2.weight": 8,  # INT8 for a sensitive layer
        },
    )
    print(f"Mixed precision quantized: {output_path}")


# ---------------------------------------------------------------------------
# 4. Activation-Based Calibration
# ---------------------------------------------------------------------------

def calibrate_with_real_data(input_path: str, output_path: str, data_path: str):
    """Calibration with real data (best accuracy).

    Runs forward passes on representative samples to determine
    optimal quantization ranges per layer. Much better than
    weight-only quantization for accuracy-sensitive deployments.

    The .npy file should have shape [num_samples, ...input_dims],
    e.g., [100, 3, 224, 224] for ImageNet models.
    """
    quantize_rs.quantize_with_calibration(
        input_path,
        output_path,
        calibration_data=data_path,
        bits=8,
        per_channel=True,
        method="minmax",
    )
    print(f"Calibrated (real data, minmax): {output_path}")


def calibrate_with_random_samples(input_path: str, output_path: str):
    """Calibration with random samples (no data needed).

    When you don't have calibration data, random samples still
    give better results than weight-only quantization. The input
    shape is auto-detected from the model.
    """
    quantize_rs.quantize_with_calibration(
        input_path,
        output_path,
        num_samples=50,
        method="percentile",
    )
    print(f"Calibrated (random samples, percentile): {output_path}")


def compare_calibration_methods(input_path: str, output_dir: str):
    """Compare all four calibration methods side by side.

    - minmax:     Uses observed min/max. Fast, sometimes sensitive to outliers.
    - percentile: Clips at 99.9th percentile. Robust to outliers.
    - entropy:    Minimizes KL divergence. Best for distributions with long tails.
    - mse:        Minimizes mean squared error. Good general-purpose choice.
    """
    os.makedirs(output_dir, exist_ok=True)

    for method in ["minmax", "percentile", "entropy", "mse"]:
        output_path = os.path.join(output_dir, f"model_{method}.onnx")
        quantize_rs.quantize_with_calibration(
            input_path,
            output_path,
            num_samples=50,
            method=method,
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  {method:12s} -> {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# 5. Real-World Scenarios
# ---------------------------------------------------------------------------

def scenario_vision_classification(model_path: str, output_path: str):
    """Image classification model (ResNet, EfficientNet, MobileNet, etc.).

    Recommended: per-channel INT8 with calibration.
    First/last layers in FP32 for best accuracy.
    """
    quantize_rs.quantize(
        model_path,
        output_path,
        bits=8,
        per_channel=True,
        min_elements=256,  # skip small bias tensors
    )


def scenario_nlp_transformer(model_path: str, output_path: str):
    """Transformer/NLP model (BERT, GPT-2, etc.).

    Embedding layers and final LM head are accuracy-sensitive.
    Use mixed precision: INT4 body + INT8 head for best compression
    with acceptable quality.
    """
    quantize_rs.quantize(
        model_path,
        output_path,
        bits=4,
        min_elements=8192,  # skip small projection layers
        layer_bits={
            # keep embedding and output head at INT8
            "transformer.wte.weight": 8,
            "transformer.wpe.weight": 8,
            "lm_head.weight": 8,
        },
    )


def scenario_edge_deployment(model_path: str, output_path: str):
    """Maximum compression for edge/mobile/IoT devices.

    INT4 across the board. Accept higher accuracy loss for
    smallest possible model size (~8x compression).
    """
    quantize_rs.quantize(
        model_path,
        output_path,
        bits=4,
        per_channel=True,
    )


def scenario_high_accuracy(model_path: str, output_path: str, data_path: str):
    """Accuracy-critical deployment (medical, finance, autonomous).

    Per-channel INT8 with real calibration data and entropy method.
    Exclude first/last layers. Minimize quantization error at every step.
    """
    quantize_rs.quantize_with_calibration(
        model_path,
        output_path,
        calibration_data=data_path,
        bits=8,
        per_channel=True,
        method="entropy",
    )


def scenario_batch_processing(input_dir: str, output_dir: str):
    """Quantize all ONNX models in a directory.

    Useful for model registries or CI pipelines that need
    quantized variants of every model.
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)
    models = glob.glob(os.path.join(input_dir, "*.onnx"))

    for model_path in models:
        name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(output_dir, f"{name}_int8.onnx")

        try:
            info = quantize_rs.model_info(model_path)
            print(f"Quantizing {info.name} ({info.num_nodes} nodes)...")
            quantize_rs.quantize(model_path, output_path, bits=8, per_channel=True)

            orig_size = os.path.getsize(model_path) / (1024 * 1024)
            quant_size = os.path.getsize(output_path) / (1024 * 1024)
            ratio = orig_size / quant_size if quant_size > 0 else 0
            print(f"  {orig_size:.1f} MB -> {quant_size:.1f} MB ({ratio:.1f}x)")
        except Exception as e:
            print(f"  FAILED: {e}")


# ---------------------------------------------------------------------------
# 6. ONNX Runtime Verification
# ---------------------------------------------------------------------------

def verify_with_onnxruntime(original_path: str, quantized_path: str):
    """Load both models in ONNX Runtime and compare outputs.

    Checks that the quantized model produces similar results
    to the original. Useful as a post-quantization sanity check.
    """
    import onnxruntime as ort

    # Load both models
    orig_session = ort.InferenceSession(original_path)
    quant_session = ort.InferenceSession(quantized_path)

    # Generate random input matching model's expected shape
    inp = orig_session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.randn(*shape).astype(np.float32)

    # Run inference
    orig_out = orig_session.run(None, {inp.name: x})[0]
    quant_out = quant_session.run(None, {inp.name: x})[0]

    # Compare
    cos_sim = np.dot(orig_out.flat, quant_out.flat) / (
        np.linalg.norm(orig_out) * np.linalg.norm(quant_out) + 1e-8
    )
    max_err = np.max(np.abs(orig_out - quant_out))

    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max absolute error: {max_err:.6f}")
    print(f"Result: {'PASS' if cos_sim > 0.95 else 'CHECK ACCURACY'}")


# ---------------------------------------------------------------------------
# 7. Preparing Calibration Data
# ---------------------------------------------------------------------------

def prepare_imagenet_calibration(image_dir: str, output_path: str, num_samples: int = 100):
    """Create a calibration .npy file from a directory of images.

    Preprocesses images to ImageNet standard: resize to 256,
    center-crop to 224, normalize with ImageNet mean/std.
    """
    from PIL import Image

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    samples = []
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:num_samples]

    for fname in image_files:
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")

        # Resize shortest side to 256, center crop 224x224
        ratio = 256.0 / min(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        left = (img.width - 224) // 2
        top = (img.height - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))

        # Convert to CHW float32 and normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        samples.append(arr)

    calibration_data = np.stack(samples)
    np.save(output_path, calibration_data)
    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"Shape: {calibration_data.shape}")


# ---------------------------------------------------------------------------
# 8. Full Pipeline Example
# ---------------------------------------------------------------------------

def full_pipeline(model_path: str, output_dir: str):
    """Complete quantization pipeline: inspect, quantize multiple
    variants, verify, and report results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Inspect
    print("=" * 60)
    print("Step 1: Model Inspection")
    print("=" * 60)
    info = inspect_model(model_path)
    orig_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Size: {orig_size:.2f} MB\n")

    # Step 2: Quantize variants
    variants = {
        "int8": dict(bits=8),
        "int8_perchannel": dict(bits=8, per_channel=True),
        "int4": dict(bits=4),
        "int4_perchannel": dict(bits=4, per_channel=True),
    }

    print("=" * 60)
    print("Step 2: Quantization")
    print("=" * 60)
    results = {}
    for name, kwargs in variants.items():
        out_path = os.path.join(output_dir, f"{info.name}_{name}.onnx")
        try:
            quantize_rs.quantize(model_path, out_path, **kwargs)
            size = os.path.getsize(out_path) / (1024 * 1024)
            ratio = orig_size / size if size > 0 else 0
            results[name] = (out_path, size, ratio)
            print(f"  {name:20s}  {size:7.2f} MB  ({ratio:.1f}x)")
        except Exception as e:
            print(f"  {name:20s}  FAILED: {e}")
    print()

    # Step 3: Verify with ONNX Runtime (if available)
    print("=" * 60)
    print("Step 3: Verification")
    print("=" * 60)
    try:
        import onnxruntime  # noqa: F401
        for name, (out_path, _, _) in results.items():
            print(f"\n  {name}:")
            verify_with_onnxruntime(model_path, out_path)
    except ImportError:
        print("  onnxruntime not installed, skipping verification.")
        print("  Install with: pip install onnxruntime")


# ---------------------------------------------------------------------------
# Main — run examples on a model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="quantize-rs Python examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples.py --model resnet18.onnx
  python examples.py --model resnet18.onnx --example calibrate --data samples.npy
  python examples.py --model resnet18.onnx --example pipeline --output-dir results/
        """,
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--example",
        default="pipeline",
        choices=[
            "inspect", "int8", "int4", "perchannel", "exclude",
            "mixed", "calibrate", "compare", "verify", "pipeline",
        ],
        help="Which example to run (default: pipeline)",
    )
    parser.add_argument("--output", default="model_quantized.onnx", help="Output path")
    parser.add_argument("--output-dir", default="quantized", help="Output directory")
    parser.add_argument("--data", default=None, help="Calibration data .npy path")
    args = parser.parse_args()

    if args.example == "inspect":
        inspect_model(args.model)

    elif args.example == "int8":
        basic_int8(args.model, args.output)

    elif args.example == "int4":
        basic_int4(args.model, args.output)

    elif args.example == "perchannel":
        per_channel_int8(args.model, args.output)

    elif args.example == "exclude":
        exclude_sensitive_layers(args.model, args.output)

    elif args.example == "mixed":
        mixed_precision(args.model, args.output)

    elif args.example == "calibrate":
        if args.data:
            calibrate_with_real_data(args.model, args.output, args.data)
        else:
            calibrate_with_random_samples(args.model, args.output)

    elif args.example == "compare":
        compare_calibration_methods(args.model, args.output_dir)

    elif args.example == "verify":
        # Quantize first, then verify
        basic_int8(args.model, args.output)
        verify_with_onnxruntime(args.model, args.output)

    elif args.example == "pipeline":
        full_pipeline(args.model, args.output_dir)
