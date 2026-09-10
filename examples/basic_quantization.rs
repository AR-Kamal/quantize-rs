//! Basic quantization example
//!
//! Shows how to quantize a single ONNX model
//!
//! Run with: cargo run --example basic_quantization

use anyhow::Result;
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
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

    let config = QuantConfig::int8()
        .with_per_channel(true)
        .with_symmetric(true);
    let outputs = Quantizer::new(config).quantize_model(&model)?;
    anyhow::ensure!(!outputs.is_empty(), "No eligible Conv/MatMul/Gemm weights");
    let avg_error = outputs.iter().map(|o| o.mse).sum::<f32>() / outputs.len() as f32;
    println!(
        "Quantized {} tensors; average MSE: {:.6}",
        outputs.len(),
        avg_error
    );
    let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

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
