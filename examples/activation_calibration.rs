// examples/activation_calibration.rs
//! Activation-based calibration example.
//!
//! Demonstrates the full pipeline:
//!   1. Load model and calibration data
//!   2. Run real inference to collect activation statistics
//!   3. Quantize using observed activation ranges
//!   4. Save quantized model
//!
//! Run with:
//!   cargo run --example activation_calibration -- \
//!     --model resnet18-v1-7.onnx \
//!     --calibration-data samples.npy \
//!     --output resnet18_int8_calibrated.onnx

use anyhow::Result;
use quantize_rs::{
    OnnxModel, Quantizer, QuantConfig,
    CalibrationDataset, ActivationEstimator,
};
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    // --- Parse arguments ---
    let model_path = get_arg(&args, "--model").unwrap_or_else(|| "resnet18-v1-7.onnx".to_string());
    let calib_path = get_arg(&args, "--calibration-data").unwrap_or_else(|| "calibration_samples.npy".to_string());
    let output_path = get_arg(&args, "--output").unwrap_or_else(|| "model_int8_calibrated.onnx".to_string());
    let bits: u8 = get_arg(&args, "--bits")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let per_channel = args.contains(&"--per-channel".to_string());
    
    // Parse custom shape (e.g., --shape "1,28,28" for MNIST)
    let custom_shape: Option<Vec<usize>> = get_arg(&args, "--shape")
        .map(|s| s.split(',')
            .filter_map(|dim| dim.trim().parse().ok())
            .collect());

    println!("Activation-Based Calibration Pipeline");
    println!("======================================");
    println!("Model:            {}", model_path);
    println!("Calibration data: {}", calib_path);
    println!("Output:           {}", output_path);
    println!("Bits:             INT{}", bits);
    println!("Per-channel:      {}", per_channel);
    if let Some(ref shape) = custom_shape {
        println!("Custom shape:     {:?}", shape);
    }
    println!();

    // --- Step 1: Load model ---
    println!("[1/5] Loading model...");
    let model = OnnxModel::load(&model_path)?;
    let info = model.info();
    println!("  Model: {}", info.name);
    println!("  Nodes: {}", info.num_nodes);
    println!();

    // --- Step 2: Load or generate calibration data ---
    println!("[2/5] Loading calibration data...");
    let dataset = if std::path::Path::new(&calib_path).exists() {
        CalibrationDataset::from_numpy(&calib_path)?
    } else {
        println!("  ⚠ File not found, generating random samples");
        
        // Use custom shape if provided, otherwise auto-detect from model
        let input_shape = if let Some(shape) = custom_shape {
            println!("  Using custom shape: {:?}", shape);
            shape
        } else if !info.inputs.is_empty() {
            // Parse shape from input info string (e.g., "input: float32[1,1,28,28]")
            let input_str = &info.inputs[0];
            if let Some(shape_part) = input_str.split('[').nth(1) {
                if let Some(shape_str) = shape_part.split(']').next() {
                    let dims: Vec<usize> = shape_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    
                    // Skip batch dimension (first dim), use remaining dims
                    if dims.len() >= 2 {
                        let detected = dims[1..].to_vec();
                        println!("  Auto-detected shape: {:?}", detected);
                        detected
                    } else {
                        println!("  Could not parse shape, using ImageNet default");
                        vec![3, 224, 224]
                    }
                } else {
                    vec![3, 224, 224]
                }
            } else {
                vec![3, 224, 224]
            }
        } else {
            println!("  Using ImageNet default shape");
            vec![3, 224, 224]
        };
        
        CalibrationDataset::random(input_shape, 100, (0.0, 1.0))?
    };
    println!("  Samples: {}", dataset.len());
    println!("  Shape:   {:?}", dataset.sample_shape());
    println!();

    // --- Step 3: Run activation-based calibration ---
    println!("[3/5] Running activation-based calibration...");
    println!("  This runs {} real inference passes to collect activation ranges.", dataset.len());
    let mut estimator = ActivationEstimator::new(model, &model_path)?;
    estimator.calibrate(&dataset)?;
    let activation_stats: std::collections::HashMap<String, quantize_rs::ActivationStats> =
        estimator.get_layer_stats()
            .into_iter()
            .map(|(k, v)| (k, v.clone()))
            .collect();
    println!("  Collected stats for {} layers", activation_stats.len());
    println!();

    // --- Step 4: Quantize using activation statistics ---
    println!("[4/5] Quantizing model with activation-based ranges...");
    let mut model = estimator.into_model();

    let config = QuantConfig {
        bits,
        per_channel,
        calibration_method: Some(quantize_rs::calibration::methods::CalibrationMethod::MinMax),
        ..Default::default()
    };

    let quantizer = Quantizer::with_calibration(config, activation_stats);
    let weights = model.extract_weights();

    let mut quantized_data = Vec::new();
    for weight in &weights {
        let quantized = quantizer.quantize_tensor_with_name(
            &weight.name,
            &weight.data,
            weight.shape.clone(),
        )?;

        let (scales, zero_points) = quantized.get_all_scales_zero_points();
        let is_per_channel = quantized.is_per_channel();

        quantized_data.push(QdqWeightInput {
            original_name: weight.name.clone(),
            quantized_values: quantized.data(),
            scales,
            zero_points,
            bits: quantized.bits(),
            axis: if is_per_channel { Some(0) } else { None },
        });
    }
    println!("  Quantized {} weight tensors", quantized_data.len());
    println!();

    // --- Step 5: Save quantized model ---
    println!("[5/5] Saving quantized model...");
    model.save_quantized(&quantized_data, &output_path)?;
    println!("  ✓ Saved to: {}", output_path);
    println!();

    // --- Summary ---
    let original_size = std::fs::metadata(&model_path)?.len();
    let quantized_size = std::fs::metadata(&output_path)?.len();
    let compression_ratio = original_size as f64 / quantized_size as f64;

    println!("Summary");
    println!("=======");
    println!("Original size:  {:.2} MB", original_size as f64 / 1_048_576.0);
    println!("Quantized size: {:.2} MB", quantized_size as f64 / 1_048_576.0);
    println!("Compression:    {:.2}×", compression_ratio);
    println!();
    println!("✓ Activation-based calibration complete!");
    println!();
    println!("Next steps:");
    println!("  1. Load the quantized model in ONNX Runtime");
    println!("  2. Run inference and compare accuracy vs original");
    println!("  3. Expect ~3× better accuracy than weight-based quantization");

    Ok(())
}

// Helper to extract command-line arguments
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|pos| args.get(pos + 1))
        .cloned()
}