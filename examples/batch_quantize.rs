//! Batch quantization example
//!
//! Quantize multiple models at once
//!
//! Run with: cargo run --example batch_quantize

use anyhow::Result;
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};
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
        quantized_data.push((weight.name.clone(), quantized.data, quantized.params));
    }

    model.save_quantized(&quantized_data, output_path)?;
    println!("  ✓ Saved to: {}\n", output_path);

    Ok(())
}

fn main() -> Result<()> {
    println!("=== Batch Quantization Example ===\n");

    // List of models to quantize
    let models = vec![
        ("mnist.onnx", "mnist_int8.onnx"),
        ("resnet18-v1-7.onnx", "resnet18_int8.onnx"),
        // Add more models here
    ];

    println!("Quantizing {} models...\n", models.len());

    for (input, output) in models {
        // Check if file exists
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
