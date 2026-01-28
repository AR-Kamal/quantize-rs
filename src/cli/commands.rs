// src/cli/commands.rs

//! Command implementations

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use crate::config::Config;
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
        println!("⚠  No weights found to quantize!");
        return Ok(());
    }

    println!("✓ Found {} weight tensors", weights.len());
    println!(
        "  Original size: {:.2} MB",
        original_size as f32 / 1_048_576.0
    );
    println!();

    // Quantize with progress bar
    if per_channel {
        println!("Quantizing to INT{} (per-channel)...", bits);
    } else {
        println!("Quantizing to INT{}...", bits);
    }

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

        if weight.data.len() > 10000 && per_channel {
            eprintln!("  {}: {} elements, MSE: {:.8}", 
                    weight.name, weight.data.len(), error);
        }
        total_error += error;

        // Get quantization parameters
        let (scale, zero_point) = quantized.get_scale_zero_point();
        let bits_used = quantized.bits();
        
        quantized_data.push((
            weight.name.clone(),
            quantized.data(),
            scale,
            zero_point,
            bits_used,
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

    // Save the quantized model
    println!("Saving quantized model...");
    model.save_quantized(&quantized_data, output)?;
    println!("✓ Saved to: {}", output.green());

    // Optional: Validate saved model
    println!("Validating saved model...");
    match OnnxModel::load(output) {
        Ok(_) => println!("✓ Model validation passed"),
        Err(e) => println!("⚠  Warning: Could not validate saved model: {}", e),
    }

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

pub fn validate(original_path: &str, quantized_path: &str, detailed: bool) -> Result<()> {
    println!("{}", "Validating Quantized Model".bold());
    println!("{}", "=".repeat(60));
    println!();
    
    println!("Original:  {}", original_path.cyan());
    println!("Quantized: {}", quantized_path.yellow());
    println!();
    
    // Step 1: Load both models
    println!("Loading models...");
    let original_model = OnnxModel::load(original_path)
        .context("Failed to load original model")?;
    let quantized_model = OnnxModel::load(quantized_path)
        .context("Failed to load quantized model")?;
    
    println!("✓ Both models loaded successfully");
    println!();
    
    // Step 2: Compare structure
    println!("{}", "Structure Validation".bold());
    println!("{}", "-".repeat(60));
    
    let original_info = original_model.info();
    let quantized_info = quantized_model.info();
    
    let mut validation_passed = true;
    
    // Check nodes
    if original_info.num_nodes == quantized_info.num_nodes {
        println!("✓ Node count matches: {}", original_info.num_nodes);
    } else {
        println!("✗ Node count mismatch: {} vs {}", 
                 original_info.num_nodes, quantized_info.num_nodes);
        validation_passed = false;
    }
    
    // Check inputs
    if original_info.inputs.len() == quantized_info.inputs.len() {
        println!("✓ Input count matches: {}", original_info.inputs.len());
    } else {
        println!("✗ Input count mismatch: {} vs {}", 
                 original_info.inputs.len(), quantized_info.inputs.len());
        validation_passed = false;
    }
    
    // Check outputs
    if original_info.outputs.len() == quantized_info.outputs.len() {
        println!("✓ Output count matches: {}", original_info.outputs.len());
    } else {
        println!("✗ Output count mismatch: {} vs {}", 
                 original_info.outputs.len(), quantized_info.outputs.len());
        validation_passed = false;
    }
    
    println!();
    
    // Step 3: Compare weights
    println!("{}", "Weight Validation".bold());
    println!("{}", "-".repeat(60));
    
    let original_weights = original_model.extract_weights();
    let quantized_weights = quantized_model.extract_weights();
    
    if original_weights.len() == quantized_weights.len() {
        println!("✓ Weight tensor count matches: {}", original_weights.len());
    } else {
        println!("⚠  Weight tensor count differs: {} vs {}", 
                 original_weights.len(), quantized_weights.len());
        println!("   (This might be expected if quantization added/merged tensors)");
    }
    
    // Check shapes match
    let mut shape_mismatches = 0;
    for (orig, quant) in original_weights.iter().zip(quantized_weights.iter()) {
        if orig.shape != quant.shape {
            shape_mismatches += 1;
            if detailed {
                println!("⚠  Shape mismatch in '{}': {:?} vs {:?}", 
                         orig.name, orig.shape, quant.shape);
            }
        }
    }
    
    if shape_mismatches == 0 {
        println!("✓ All weight shapes match");
    } else {
        println!("⚠  {} weight tensors have shape mismatches", shape_mismatches);
        validation_passed = false;
    }
    
    // Check for suspicious patterns in quantized model
    println!();
    println!("Checking for numerical issues...");

    let mut all_zeros = 0;
    let mut all_same = 0;
    let mut suspicious = false;

    for weight in &quantized_weights {
        // Check if all values are zero
        if weight.data.iter().all(|&v| v == 0.0) {
            all_zeros += 1;
        }
        
        // Check if all values are the same (might indicate bad quantization)
        if weight.data.len() > 1 {
            let first = weight.data[0];
            if weight.data.iter().all(|&v| v == first) {
                all_same += 1;
            }
        }
    }

    // Small tensors (like biases) can legitimately be all zeros
    if all_zeros > original_weights.len() / 4 {
        println!("⚠  {} tensors are all zeros (might indicate issues)", all_zeros);
        suspicious = true;
    } else if all_zeros > 0 {
        println!("ℹ️  {} small tensors are zero (likely biases)", all_zeros);
    }

    if all_same > original_weights.len() / 10 {
        println!("⚠  {} tensors have identical values (might indicate quantization failure)", all_same);
        suspicious = true;
    }

    if !suspicious && all_zeros == 0 && all_same == 0 {
        println!("✓ No numerical issues detected");
    }
    
    println!();
    
    // Step 4: Size comparison
    println!("{}", "Size Analysis".bold());
    println!("{}", "-".repeat(60));
    
    let original_size = std::fs::metadata(original_path)?.len();
    let quantized_size = std::fs::metadata(quantized_path)?.len();
    let compression = original_size as f32 / quantized_size as f32;
    let reduction = ((original_size - quantized_size) as f32 / original_size as f32) * 100.0;
    
    println!("Original:  {:.2} MB", original_size as f32 / 1_048_576.0);
    println!("Quantized: {:.2} MB", quantized_size as f32 / 1_048_576.0);
    println!("Reduction: {:.1}% ({:.2}x smaller)", reduction, compression);
    
    if compression < 2.0 {
        println!("⚠  Warning: Low compression ratio. Quantization may not be working correctly.");
    } else if compression >= 3.5 && compression <= 4.5 {
        println!("✓ Expected compression for INT8 quantization");
    }
    
    println!();
    
    // Step 5: Detailed per-layer analysis (if requested)
    if detailed {
        println!("{}", "Detailed Layer Analysis".bold());
        println!("{}", "-".repeat(60));
        
        use quantize_rs::quantization::{QuantConfig, Quantizer};
        
        let config = QuantConfig::int8();
        let quantizer = Quantizer::new(config);
        
        println!("{:<40} {:>12} {:>15}", "Layer", "Elements", "MSE Error");
        println!("{}", "-".repeat(68));
        
        for weight in original_weights.iter().take(10) { // Show first 10
            if let Ok(quantized) = quantizer.quantize_tensor(&weight.data, weight.shape.clone()) {
                let error = quantized.quantization_error(&weight.data);
                
                let name = if weight.name.len() > 37 {
                    format!("{}...", &weight.name[..37])
                } else {
                    weight.name.clone()
                };
                
                println!("{:<40} {:>12} {:>15.8}", 
                         name, 
                         weight.data.len(),
                         error);
            }
        }
        
        if original_weights.len() > 10 {
            println!("... ({} more layers)", original_weights.len() - 10);
        }
        
        println!();
    }
    
    // Final verdict
    println!("{}", "=".repeat(60));
    if validation_passed {
        println!("{}", "✓ VALIDATION PASSED".green().bold());
        println!();
        println!("The quantized model appears to be valid and should work correctly.");
    } else {
        println!("{}", "⚠  VALIDATION FAILED".yellow().bold());
        println!();
        println!("Issues detected. The quantized model may not work correctly.");
        println!("Review the warnings above and consider re-quantizing.");
    }
    println!("{}", "=".repeat(60));
    
    Ok(())
}

pub fn batch(
    inputs: &[String],
    output_dir: &str,
    bits: u8,
    per_channel: bool,
    skip_existing: bool,
    continue_on_error: bool,
) -> Result<()> {
    println!("{}", "Batch Quantization".bold());
    println!("{}", "=".repeat(60));
    println!();

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    println!("Output directory: {}", output_dir.cyan());
    println!("Quantization: INT{}{}", bits, if per_channel { " (per-channel)" } else { "" });
    println!();

    // Expand wildcards and collect all input files
    let mut input_files = Vec::new();
    for pattern in inputs {
        // Check if it's a direct file path
        if Path::new(pattern).exists() {
            input_files.push(PathBuf::from(pattern));
        } else {
            // Try glob pattern
            match glob::glob(pattern) {
                Ok(paths) => {
                    for path in paths.filter_map(Result::ok) {
                        if path.extension().and_then(|s| s.to_str()) == Some("onnx") {
                            input_files.push(path);
                        }
                    }
                }
                Err(_) => {
                    println!("⚠  Invalid pattern or file not found: {}", pattern);
                }
            }
        }
    }

    if input_files.is_empty() {
        println!("✗ No ONNX files found matching the input patterns");
        return Ok(());
    }

    println!("Found {} models to process", input_files.len());
    println!();

    // Process each model
    let mut results = Vec::new();
    let total = input_files.len();

    for (idx, input_path) in input_files.iter().enumerate() {
        let input_str = input_path.to_string_lossy();
        let filename = input_path.file_name().unwrap().to_string_lossy();
        
        // Generate output filename (add _int8 suffix)
        let output_filename = if let Some(stem) = input_path.file_stem() {
            format!("{}_int8.onnx", stem.to_string_lossy())
        } else {
            format!("{}_int8.onnx", filename)
        };
        
        let output_path = Path::new(output_dir).join(&output_filename);
        let output_str = output_path.to_string_lossy();

        println!("{}", format!("[{}/{}]", idx + 1, total).bold());
        println!("Processing: {}", filename.cyan());

        // Skip if already exists
        if skip_existing && output_path.exists() {
            println!("Skipped (already exists)");
            println!();
            results.push((input_str.to_string(), "Skipped".to_string()));
            continue;
        }

        // Quantize the model
        match quantize(&input_str, &output_str, bits, per_channel) {
            Ok(_) => {
                println!("✓ Success");
                println!();
                results.push((input_str.to_string(), "Success".green().to_string()));
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
                println!();
                results.push((input_str.to_string(), format!("Failed: {}", e).red().to_string()));
                
                if !continue_on_error {
                    println!("✗ Stopping batch processing due to error");
                    println!("   Use --continue-on-error to process remaining models");
                    break;
                }
            }
        }
    }

    // Summary
    println!("{}", "=".repeat(60));
    println!("{}", "Batch Summary".bold());
    println!("{}", "=".repeat(60));
    println!();

    let success_count = results.iter()
        .filter(|(_, status)| status.contains("Success"))
        .count();
    let skipped_count = results.iter()
        .filter(|(_, status)| status.contains("Skipped"))
        .count();
    let failed_count = results.iter()
        .filter(|(_, status)| status.contains("Failed"))
        .count();

    println!("Total:    {}", results.len());
    println!("Success:  {}", success_count.to_string().green());
    if skipped_count > 0 {
        println!("Skipped:  {}", skipped_count);
    }
    if failed_count > 0 {
        println!("Failed:   {}", failed_count.to_string().red());
    }
    println!();

    // Show details if there were failures
    if failed_count > 0 {
        println!("Failed models:");
        for (input, status) in &results {
            if status.contains("Failed") {
                println!("  ✗ {}", input);
            }
        }
        println!();
    }

    if failed_count == 0 {
        println!("{}", "✓ All models processed successfully!".green().bold());
    } else if success_count > 0 {
        println!("{}", "⚠  Some models failed, but others succeeded".yellow().bold());
    } else {
        println!("{}", "✗ All models failed".red().bold());
    }

    Ok(())
}

pub fn run_config(config_path: &str, dry_run: bool) -> Result<()> {
    println!("{}", "Running from Config File".bold());
    println!("{}", "=".repeat(60));
    println!();

    // Load config
    println!("Loading config: {}", config_path.cyan());
    let config = Config::from_file(config_path)
        .context("Failed to load configuration file")?;

    // Validate
    config.validate()
        .context("Config validation failed")?;

    println!("✓ Config loaded and validated");
    println!();

    if dry_run {
        println!("{}", "DRY RUN MODE - No files will be modified".yellow().bold());
        println!();
    }

    // Show config summary
    println!("Global settings:");
    println!("  Bits: {}", config.bits);
    println!("  Per-channel: {}", config.per_channel);
    println!();

    let mut total_tasks = 0;

    // Process individual models
    if !config.models.is_empty() {
        println!("{}", format!("Individual Models: {}", config.models.len()).bold());
        println!("{}", "-".repeat(60));

        for (idx, model_config) in config.models.iter().enumerate() {
            let bits = config.get_bits(model_config);
            let per_channel = config.get_per_channel(model_config);

            println!();
            println!("[{}] {}", idx + 1, model_config.input.cyan());
            println!("    → {}", model_config.output.green());
            println!("    Bits: {}, Per-channel: {}", bits, per_channel);

            if model_config.skip_existing && Path::new(&model_config.output).exists() {
                println!("Skipped (already exists)");
                continue;
            }

            if dry_run {
                println!("Would quantize (dry run)");
                total_tasks += 1;
                continue;
            }

            if let Some(parent) = Path::new(&model_config.output).parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create output directory: {:?}", parent))?;
            }

            // Actually quantize
            match quantize(&model_config.input, &model_config.output, bits, per_channel) {
                Ok(_) => {
                    println!("✓ Success");
                    total_tasks += 1;
                }
                Err(e) => {
                    println!("✗ Failed: {}", e);
                }
            }
        }
        println!();
    }

    // Process batch
    if let Some(batch_config) = &config.batch {
        println!("{}", "Batch Processing".bold());
        println!("{}", "-".repeat(60));
        println!();

        if dry_run {
            println!("Would process batch:");
            println!("  Input:  {}", batch_config.input_dir);
            println!("  Output: {}", batch_config.output_dir);
            println!("  (dry run)");
        } else {
            // Expand input_dir into file list
            let inputs = vec![batch_config.input_dir.clone()];
            
            batch(
                &inputs,
                &batch_config.output_dir,
                config.bits,
                config.per_channel,
                batch_config.skip_existing,
                batch_config.continue_on_error,
            )?;
        }
    }

    // Summary
    println!();
    println!("{}", "=".repeat(60));
    if dry_run {
        println!("{}", "✓ Dry run complete".green().bold());
        println!("Run without --dry-run to actually quantize models");
    } else {
        println!("{}", "✓ Config execution complete".green().bold());
        if total_tasks > 0 {
            println!("Processed {} models", total_tasks);
        }
    }

    Ok(())
}