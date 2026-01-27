//! Basic quantization example
//!
//! Run with: cargo run --example basic_quantization

use anyhow::Result;
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};

fn main() -> Result<()> {
    println!("=== Basic Quantization Example ===\n");

    // 1. Load an ONNX model
    println!("Loading model...");
    let mut model = OnnxModel::load("mnist.onnx")?;
    println!("✓ Model loaded\n");

    // 2. Get model info
    let info = model.info();
    println!("Model Info:");
    println!("   Name: {}", info.name);
    println!("   Nodes: {}", info.num_nodes);
    println!("   Inputs: {}", info.inputs.len());
    println!("   Outputs: {}\n", info.outputs.len());

    // 3. Extract weights
    println!("Extracting weights...");
    let weights = model.extract_weights();
    println!("✓ Found {} weight tensors\n", weights.len());

    // 4. Create quantizer
    let config = QuantConfig::int8();
    let quantizer = Quantizer::new(config);

    // 5. Quantize weights
    println!("Quantizing...");
    let mut quantized_data = Vec::new();

    for weight in &weights {
        let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())?;

        quantized_data.push((weight.name.clone(), quantized.data, quantized.params));
    }

    println!("✓ Quantization complete\n");

    // 6. Save quantized model
    println!("Saving...");
    model.save_quantized(&quantized_data, "mnist_quantized.onnx")?;
    println!("✓ Saved to: mnist_quantized.onnx\n");

    println!("Done!");

    Ok(())
}
