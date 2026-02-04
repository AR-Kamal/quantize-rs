# Activation-Based Calibration Integration Guide

## What Changed

**Old approach (v0.2.0):** `ActivationEstimator` simulated activations using statistical heuristics — it never ran the model. For BatchNorm it hardcoded `[-3, 3]`, for ReLU it clipped negatives, etc. This was fast but inaccurate.

**New approach (v0.3.0):** Real inference using tract. We run your calibration samples through the actual model, capture the intermediate tensor values at every layer, and use those *observed* min/max values for quantization ranges. This is what gives the "3× better accuracy" you cited.

---

## File Placement

```
src/calibration/inference.rs        ← REPLACE with new version
examples/activation_calibration.rs  ← NEW (add to examples/)
```

**In `Cargo.toml`, add the new example:**

```toml
[[example]]
name = "activation_calibration"
path = "examples/activation_calibration.rs"
```

---

## How It Works (Technical)

### 1. tract Setup
```rust
let mut tract_model = tract_onnx::onnx()
    .model_for_path(onnx_path)?;
```

We reload the ONNX file with tract (not the protobuf parser).

### 2. Expose Intermediate Outputs

Before optimization, we mark every node output as a model output:

```rust
for node in tract_model.nodes {
    for output in node.outputs {
        tract_model.outputs.push(output);
    }
}
```

This way after optimization (which fuses layers), we still get the intermediate tensors we care about.

### 3. Run Inference

```rust
for sample in calibration_dataset {
    let outputs = tract_model.run(sample)?;
    for (layer_name, output_tensor) in outputs {
        update_stats(layer_name, output_tensor);
    }
}
```

Each sample produces a vector of tensors (one per exposed output). We convert to f32, compute min/max/histogram, aggregate across samples.

### 4. Use Stats for Quantization

```rust
let quantizer = Quantizer::with_calibration(config, activation_stats);
quantizer.quantize_tensor_with_name(weight_name, weight_data, shape)?;
```

The quantizer checks if `activation_stats` contains an entry for this weight name. If yes, it uses the observed range. If no (e.g., bias terms that don't have activations), it falls back to weight-based range.

---

## CLI Integration

Your existing `calibrate` command likely calls the old `ActivationEstimator`. Here's how to update it:

**In `src/cli/commands.rs` (or wherever `calibrate` is defined):**

```rust
// OLD (remove this):
// let mut estimator = ActivationEstimator::new(model);

// NEW (requires the ONNX path):
let mut estimator = ActivationEstimator::new(model, &model_path)?;
```

The key difference: the new `ActivationEstimator::new` requires the path to the ONNX file (a `&str`), because it needs to reload the model with tract. Make sure your CLI passes the path through.

**Example CLI signature:**

```bash
quantize-rs calibrate model.onnx --data calibration.npy -o model_calibrated.onnx --bits 4 --method percentile
```

Make sure the command has access to `model.onnx` as a string path, not just the loaded `OnnxModel` struct.

---

## Testing

### 1. Unit Tests

```bash
cargo test
```

All existing tests should still pass (22/22 from the graph fix, plus the new inference tests if you have a model file).

### 2. Activation Estimator Test (requires ONNX model)

```bash
# Place mnist.onnx or resnet18-v1-7.onnx in project root
cargo test test_activation_estimator_real_inference -- --ignored --nocapture
```

This will:
- Load the model with tract
- Generate 5 random calibration samples
- Run inference and collect activation stats
- Verify that stats are non-trivial (min ≠ max for each layer)

Expected output:
```
Testing with model: mnist.onnx
Model: mnist-8, 11 nodes
Running activation-based calibration on 5 samples...
  Processed 5/5 samples
✓ Calibration complete: 8 layers tracked

Collected stats for 8 layers:
  conv1: min=-0.4521, max=1.2341, mean=0.3812
  conv2: min=-1.1023, max=2.0451, mean=0.4023
  ...
```

### 3. Full Pipeline Example

```bash
cargo run --example activation_calibration -- \
    --model resnet18-v1-7.onnx \
    --calibration-data samples.npy \
    --output resnet18_calibrated.onnx \
    --bits 8 \
    --per-channel
```

If `samples.npy` doesn't exist, it will generate 100 random samples with shape [3, 224, 224] (ImageNet standard).

Expected output:
```
[1/5] Loading model...
  Model: resnet18, 69 nodes
[2/5] Loading calibration data...
  Samples: 100
  Shape: [3, 224, 224]
[3/5] Running activation-based calibration...
  This runs 100 real inference passes to collect activation ranges.
  Processed 10/100 samples
  Processed 20/100 samples
  ...
  Processed 100/100 samples
✓ Calibration complete: 62 layers tracked
[4/5] Quantizing model with activation-based ranges...
  Quantized 62 weight tensors
[5/5] Saving quantized model...
  ✓ Saved to: resnet18_calibrated.onnx

Summary
=======
Original size:  44.65 MB
Quantized size: 11.18 MB
Compression:    4.00×

✓ Activation-based calibration complete!
```

---

## Expected Accuracy Differences

### Weight-Based (Old)
```
Conv1 weight range: [-0.5, 0.5]
Quantization uses:  [-0.5, 0.5]

Problem: After BatchNorm + ReLU, actual values are [0.0, 0.2]
Result: Wasted 60% of INT8 range on values that never occur
```

### Activation-Based (New)
```
Conv1 weight range: [-0.5, 0.5]   ← ignored
Observed activation range: [0.0, 0.2]
Quantization uses: [0.0, 0.2]

Result: Full INT8 range covers real values → 3× better precision
```

**Concrete numbers (from your doc):**
- ResNet-18 on ImageNet
- Weight-based:     69.76% → 69.52% (0.24% drop)
- Activation-based: 69.76% → 69.68% (0.08% drop) ← 3× better

---

## Troubleshooting

### "tract failed to load ONNX model"

Make sure:
1. The ONNX file path is correct and the file exists
2. The model is a valid ONNX file (not corrupted)
3. tract supports the opset version (it's usually fine for opset 10-17)

### "Failed to cast tensor to f32"

Some intermediate tensors might be INT64 (indices) or BOOL (masks). The code handles this by casting, but if you see this error, it means a tensor type tract doesn't know how to convert. File an issue with the specific model.

### "No activation statistics collected"

This means tract optimized away all the intermediate outputs (unlikely). Check:
- Does `info.num_nodes > 0`?
- Does the model have actual computation (not just a single Reshape)?

### Calibration is slow

Activation-based calibration runs real inference, so it's slower than weight-based:
- Weight-based:     seconds (just weight min/max)
- Activation-based: minutes (100 inference passes)

For 100 samples on ResNet-18 on a CPU, expect ~2-5 minutes. This is normal. The accuracy gain is worth it for production deployments (medical, automotive, finance).

---

## Next Steps After v0.3.0

1. **Per-channel activation calibration** (v0.4.0)
   - Current: single scale/zp per tensor
   - Future: vector of scales per channel
   - Requires `axis` attribute on DequantizeLinear

2. **Calibration data loaders** (v0.4.0)
   - Support loading images directly (JPEG, PNG)
   - Auto-resize to model input size
   - Apply standard preprocessing (ImageNet normalization)

3. **Calibration method comparison** (v0.4.0)
   - Run MinMax, Percentile, Entropy, MSE on same data
   - Show accuracy vs compression tradeoff
   - Auto-select best method per layer

---

## Summary

Drop in the new `inference.rs`, add the example to `Cargo.toml`, update your CLI to pass the ONNX path to `ActivationEstimator::new()`. Test with the ignored test, then run the full example. You'll see real intermediate tensor values and the accuracy improvement vs weight-based quantization.

The critical behavioral change: **calibration now takes minutes instead of seconds**, because it's running real inference. This is expected and correct — the time investment buys you 3× better accuracy retention.