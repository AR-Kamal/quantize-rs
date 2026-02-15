//! Batch quantization example
//!
//! Quantize multiple models at once
//!
//! Run with: cargo run --example batch_quantize

use anyhow::Result;
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use std::path::Path;

fn quantize_model(input_path: &str, output_path: &str) -> Result<()> {
    println!("Processing: {}", input_path);

    let mut model = OnnxModel::load(input_path)?;
    let weights = model.extract_weights();

    let config = QuantConfig::int8();
    let quantizer = Quantizer::new(config);

    let mut quantized_data = Vec::new();
    for weight in &weights {
        let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())?;

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

    model.save_quantized(&quantized_data, output_path)?;
    println!("  ✓ Saved to: {}\n", output_path);

    Ok(())
}

fn main() -> Result<()> {
    println!("=== Batch Quantization Example ===\n");

    let models = vec![
        ("mnist.onnx", "mnist_int8.onnx"),
        ("resnet18-v1-7.onnx", "resnet18_int8.onnx"),
    ];

    println!("Quantizing {} models...\n", models.len());

    for (input, output) in models {
        if !Path::new(input).exists() {
            println!("⚠️  Skipping {} (file not found)", input);
            continue;
        }

        match quantize_model(input, output) {
            Ok(_) => {}
            Err(e) => println!("  ✗ Error: {}\n", e),
        }
    }

    println!("Batch quantization complete!");

    Ok(())
}