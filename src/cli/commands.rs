//! Command implementations

use anyhow::Result;
use colored::Colorize;
use quantify::onnx_utils::OnnxModel;
use quantify::quantization::{Quantizer, QuantConfig};

pub fn quantize(input: &str, output: &str, bits: u8, per_channel: bool) -> Result<()> {
    println!("📦 Loading model: {}", input.bold());
    
    // Load model
    let model = OnnxModel::load(input)?;
    let info = model.info();
    
    println!("✓ Model loaded");
    println!("  Nodes: {}", info.num_nodes);
    println!();
    
    // Extract weights
    println!("🔍 Extracting weights...");
    let weights = model.extract_weights();
    let original_size = model.total_size_bytes();
    
    println!("✓ Found {} weight tensors", weights.len());
    println!("  Original size: {:.2} MB", original_size as f32 / 1_048_576.0);
    println!();
    
    // Quantize
    println!("🔧 Quantizing to INT{}...", bits);
    
    if weights.is_empty() {
        println!("⚠️  No weights found to quantize!");
        return Ok(());
    }
    
    let config = QuantConfig { bits, per_channel };
    let quantizer = Quantizer::new(config);
    
    let mut quantized_weights = Vec::new();
    let mut total_error = 0.0;
    
    for weight in &weights {
        let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())?;
        let error = quantized.quantization_error(&weight.data);
        total_error += error;
        quantized_weights.push(quantized);
    }
    
    let avg_error = if quantized_weights.is_empty() {
        0.0
    } else {
        total_error / quantized_weights.len() as f32
    };
    let quantized_size: usize = quantized_weights.iter()
        .map(|w| w.size_bytes())
        .sum();
    
    let compression_ratio = original_size as f32 / quantized_size as f32;
    
    println!("✓ Quantization complete");
    println!();
    println!("📊 Results:");
    println!("  Original size:    {:.2} MB", original_size as f32 / 1_048_576.0);
    println!("  Quantized size:   {:.2} MB", quantized_size as f32 / 1_048_576.0);
    println!("  Compression:      {:.2}x smaller", compression_ratio);
    println!("  Avg MSE error:    {:.6}", avg_error);
    println!();
    
    println!("{}", "💾 Save functionality not yet implemented".yellow());
    println!("   Output would be: {}", output);
    
    Ok(())
}

pub fn info(input: &str) -> Result<()> {
    println!("📊 Model Information: {}", input.bold());
    println!();
    
    // Load model
    let model = OnnxModel::load(input)?;
    let info = model.info();
    
    // Display info
    println!("  Name:       {}", info.name.cyan());
    println!("  Version:    {}", info.version);
    println!("  Nodes:      {}", info.num_nodes);
    println!();
    
    println!("  Inputs ({}):", info.inputs.len());
    for input in &info.inputs {
        println!("    - {}", input);
    }
    println!();
    
    println!("  Outputs ({}):", info.outputs.len());
    for output in &info.outputs {
        println!("    - {}", output);
    }
    
    Ok(())
}

pub fn benchmark(original: &str, quantized: &str) -> Result<()> {
    println!("⚡ Benchmarking:");
    println!("  Original:  {}", original);
    println!("  Quantized: {}", quantized);
    println!();
    
    // TODO: Implement benchmarking
    println!("{}", "✗ Not implemented yet".red());
    
    Ok(())
}