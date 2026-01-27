//! Command implementations

use anyhow::{Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use quantize_rs::onnx_utils::OnnxModel;
use quantize_rs::quantization::{QuantConfig, Quantizer};

pub fn quantize(input: &str, output: &str, bits: u8, per_channel: bool) -> Result<()> {
    println!("Loading model: {}", input.bold());

    // Load model
    let mut model = OnnxModel::load(input)?;
    let info = model.info();

    println!("✓ Model loaded");
    println!("  Nodes: {}", info.num_nodes);
    println!();

    // Extract weights
    println!("Extracting weights...");
    let weights = model.extract_weights();
    let original_size = model.total_size_bytes();

    if weights.is_empty() {
        println!("⚠️  No weights found to quantize!");
        return Ok(());
    }

    println!("✓ Found {} weight tensors", weights.len());
    println!(
        "  Original size: {:.2} MB",
        original_size as f32 / 1_048_576.0
    );
    println!();

    // Quantize with progress bar
    println!("Quantizing to INT{}...", bits);

    let config = QuantConfig { bits, per_channel };
    let quantizer = Quantizer::new(config);

    // Create progress bar
    let pb = ProgressBar::new(weights.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tensors ({eta})")
            .expect("Failed to set progress bar template")
            .progress_chars("#>-")
    );

    let mut quantized_weights = Vec::new();
    let mut quantized_data = Vec::new();
    let mut total_error = 0.0;

    for weight in &weights {
        let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())?;
        let error = quantized.quantization_error(&weight.data);
        total_error += error;

        // Store name, data, and params for saving
        quantized_data.push((
            weight.name.clone(),
            quantized.data.clone(),
            quantized.params.clone(),
        ));

        quantized_weights.push(quantized);
        pb.inc(1);
    }

    pb.finish_with_message("done");

    let avg_error = if quantized_weights.is_empty() {
        0.0
    } else {
        total_error / quantized_weights.len() as f32
    };

    let quantized_size: usize = quantized_weights.iter().map(|w| w.size_bytes()).sum();

    let compression_ratio = original_size as f32 / quantized_size as f32;

    println!("✓ Quantization complete");
    println!();
    println!("Results:");
    println!(
        "  Original size:    {:.2} MB",
        original_size as f32 / 1_048_576.0
    );
    println!(
        "  Quantized size:   {:.2} MB",
        quantized_size as f32 / 1_048_576.0
    );
    println!("  Compression:      {:.2}x smaller", compression_ratio);
    println!("  Avg MSE error:    {:.6}", avg_error);
    println!();

    // Save the quantized model with progress indicator
    println!("Saving quantized model...");
    model.save_quantized(&quantized_data, output)?;
    println!("✓ Saved to: {}", output.green());

    Ok(())
}

pub fn info(input: &str) -> Result<()> {
    println!("Model Information: {}", input.bold());
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
    println!("Benchmarking Models");
    println!("{}", "=".repeat(50));
    println!();

    println!("Original:  {}", original.cyan());
    println!("Quantized: {}", quantized.yellow());
    println!();

    // Load both models
    println!("Loading models...");
    let original_model = OnnxModel::load(original).context("Failed to load original model")?;
    let quantized_model = OnnxModel::load(quantized).context("Failed to load quantized model")?;

    let original_info = original_model.info();
    let quantized_info = quantized_model.info();

    // Extract weights
    let original_weights = original_model.extract_weights();
    let quantized_weights = quantized_model.extract_weights();

    let original_size = original_model.total_size_bytes();
    let quantized_size = quantized_model.total_size_bytes();

    // Get file sizes
    let original_file_size = std::fs::metadata(original)
        .context("Failed to read original file")?
        .len();
    let quantized_file_size = std::fs::metadata(quantized)
        .context("Failed to read quantized file")?
        .len();

    println!("✓ Models loaded");
    println!();

    // Compare structure
    println!("{}", "Model Structure:".bold());
    println!(
        "  Nodes:       {} vs {}",
        original_info.num_nodes, quantized_info.num_nodes
    );
    println!(
        "  Inputs:      {} vs {}",
        original_info.inputs.len(),
        quantized_info.inputs.len()
    );
    println!(
        "  Outputs:     {} vs {}",
        original_info.outputs.len(),
        quantized_info.outputs.len()
    );
    println!();

    // Compare weights
    println!("{}", "Weight Analysis:".bold());
    println!(
        "  Tensors:     {} vs {}",
        original_weights.len(),
        quantized_weights.len()
    );
    println!(
        "  Total size:  {:.2} KB vs {:.2} KB",
        original_size as f32 / 1024.0,
        quantized_size as f32 / 1024.0
    );

    let compression = original_size as f32 / quantized_size.max(1) as f32;
    println!("  Compression: {:.2}x smaller", compression);
    println!();

    // File size comparison
    println!("{}", "File Size:".bold());
    println!(
        "  Original:    {:.2} MB",
        original_file_size as f32 / 1_048_576.0
    );
    println!(
        "  Quantized:   {:.2} MB",
        quantized_file_size as f32 / 1_048_576.0
    );

    let file_compression = original_file_size as f32 / quantized_file_size.max(1) as f32;
    let size_reduction =
        ((original_file_size - quantized_file_size) as f32 / original_file_size as f32) * 100.0;

    println!("  Reduction:   {:.1}%", size_reduction);
    println!("  Ratio:       {:.2}x", file_compression);
    println!();

    // Summary
    println!("{}", "Summary:".bold().green());

    // Check if structure matches
    let structure_match = original_info.num_nodes == quantized_info.num_nodes
        && original_info.inputs.len() == quantized_info.inputs.len()
        && original_info.outputs.len() == quantized_info.outputs.len();

    if structure_match {
        println!("  ✓ Structure preserved");
    } else {
        println!("  ✗ Structure mismatch detected");
    }

    if size_reduction >= 70.0 {
        println!("  ✓ Excellent compression ({:.1}%)", size_reduction);
    } else if size_reduction >= 50.0 {
        println!("  ✓ Good compression ({:.1}%)", size_reduction);
    } else {
        println!("  ⚠ Low compression ({:.1}%)", size_reduction);
    }

    if original_weights.len() == quantized_weights.len() {
        println!("  ✓ All weights quantized");
    } else {
        println!(
            "  ⚠ Weight count mismatch ({} vs {})",
            original_weights.len(),
            quantized_weights.len()
        );
    }

    println!();

    Ok(())
}
