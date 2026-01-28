//! Basic quantization example
//!
//! Shows how to quantize a single ONNX model
//!
//! Run with: cargo run --example basic_quantization

use anyhow::Result;
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};

fn main() -> Result<()> {
    println!("=== Basic Quantization Example ===\n");

    let input_path = "mnist.onnx";
    println!("Loading model: {}", input_path);

    let mut model = OnnxModel::load(input_path)?;
    let info = model.info();

    println!("✓ Loaded: {}", info.name);
    println!("  Nodes: {}", info.num_nodes);
    println!();

    println!("Extracting weights...");
    let weights = model.extract_weights();
    println!("✓ Found {} weight tensors\n", weights.len());

    let config = QuantConfig::int8();
    let quantizer = Quantizer::new(config);

    println!("Quantizing...");
    let mut quantized_data = Vec::new();
    let mut total_error = 0.0;

    for weight in &weights {
        let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())?;
        let error = quantized.quantization_error(&weight.data);
        total_error += error;

        let (scale, zero_point) = quantized.get_scale_zero_point();
        let bits = quantized.bits();

        quantized_data.push((
            weight.name.clone(),
            quantized.data(),
            scale,
            zero_point,
            bits,
        ));
    }

    let avg_error = total_error / weights.len() as f32;
    println!("✓ Quantized {} tensors", weights.len());
    println!("  Average MSE: {:.6}\n", avg_error);

    let output_path = "mnist_quantized.onnx";
    println!("Saving to: {}", output_path);
    model.save_quantized(&quantized_data, output_path)?;
    println!("✓ Saved!\n");

    let original_size = std::fs::metadata(input_path)?.len();
    let quantized_size = std::fs::metadata(output_path)?.len();
    let compression = original_size as f32 / quantized_size as f32;

    println!("Results:");
    println!("  Original:  {:.2} MB", original_size as f32 / 1_048_576.0);
    println!("  Quantized: {:.2} MB", quantized_size as f32 / 1_048_576.0);
    println!("  Compression: {:.2}x smaller", compression);

    Ok(())
}