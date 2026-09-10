//! Batch quantization example
//!
//! Quantize multiple models at once
//!
//! Run with: cargo run --example batch_quantize

use anyhow::Result;
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};
use std::path::Path;

fn quantize_model(input_path: &str, output_path: &str) -> Result<()> {
    println!("Processing: {}", input_path);

    let mut model = OnnxModel::load(input_path)?;
    let config = QuantConfig::int8()
        .with_per_channel(true)
        .with_symmetric(true);
    let outputs = Quantizer::new(config).quantize_model(&model)?;
    anyhow::ensure!(!outputs.is_empty(), "No eligible Conv/MatMul/Gemm weights");
    let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

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
