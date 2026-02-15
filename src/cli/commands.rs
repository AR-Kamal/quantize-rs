//src/cli/commands.rs
use quantize_rs::config::Config;

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use quantize_rs::onnx_utils::OnnxModel;
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use quantize_rs::quantization::{QuantConfig, Quantizer};
use quantize_rs::calibration::{CalibrationDataset, ActivationEstimator, methods::CalibrationMethod};

pub fn quantize(input: &str, output: &str, bits: u8, per_channel: bool) -> Result<()> {
    println!("Loading model: {}", input.bold());

    let mut model = OnnxModel::load(input)?;
    let info = model.info();

    println!("✓ Model loaded");
    println!("  Nodes: {}", info.num_nodes);
    println!();

    println!("Extracting weights...");
    let weights = model.extract_weights();
    let original_size: usize = weights.iter().map(|w| w.size_bytes()).sum();

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

    if per_channel {
        println!("Quantizing to INT{} (per-channel)...", bits);
    } else {
        println!("Quantizing to INT{}...", bits);
    }

    let config = QuantConfig { 
        bits, 
        per_channel,
        calibration_method: None,
    };
    let quantizer = Quantizer::new(config);

    // Quantize all weights in parallel
    let results: Vec<_> = weights.par_iter()
        .map(|weight| {
            let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())?;
            let error = quantized.quantization_error(&weight.data);
            let (scales, zero_points) = quantized.get_all_scales_zero_points();
            let is_per_channel = quantized.is_per_channel();
            let bits_used = quantized.bits();
            let size = quantized.size_bytes();

            let qdq = QdqWeightInput {
                original_name:    weight.name.clone(),
                quantized_values: quantized.data(),
                scales,
                zero_points,
                bits:             bits_used,
                axis:             if is_per_channel { Some(0) } else { None },
            };

            Ok::<_, anyhow::Error>((qdq, error, size))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut quantized_data = Vec::new();
    let mut total_error = 0.0_f32;
    let mut quantized_size = 0_usize;

    for (qdq, error, size) in results {
        total_error += error;
        quantized_size += size;
        quantized_data.push(qdq);
    }

    let avg_error = if quantized_data.is_empty() {
        0.0
    } else {
        total_error / quantized_data.len() as f32
    };

    let compression_ratio = original_size as f32 / quantized_size.max(1) as f32;

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

    println!("Saving quantized model...");
    model.save_quantized(&quantized_data, output)?;
    println!("✓ Saved to: {}", output.green());

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

    let model = OnnxModel::load(input)?;
    let info = model.info();

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

    println!("Loading models...");
    let original_model = OnnxModel::load(original).context("Failed to load original model")?;
    let quantized_model = OnnxModel::load(quantized).context("Failed to load quantized model")?;

    let original_info = original_model.info();
    let quantized_info = quantized_model.info();

    let original_weights = original_model.extract_weights();
    let original_size: usize = original_weights.iter().map(|w| w.size_bytes()).sum();

    let quantized_weight_info = quantized_model.load_quantized_info();
    let is_qdq = !quantized_weight_info.is_empty();

    let (quantized_weight_count, quantized_size) = if is_qdq {
        let count = quantized_weight_info.len();
        // Each quantized element is 1 byte (INT8 storage)
        let size: usize = quantized_weight_info.iter().map(|w| w.original_length).sum();
        (count, size)  // size in bytes (INT8 = 1 byte per element)
    } else {
        let weights = quantized_model.extract_weights();
        let size: usize = weights.iter().map(|w| w.size_bytes()).sum();
        (weights.len(), size)
    };

    let original_file_size = std::fs::metadata(original)
        .context("Failed to read original file")?
        .len();
    let quantized_file_size = std::fs::metadata(quantized)
        .context("Failed to read quantized file")?
        .len();

    println!("✓ Models loaded");
    println!();

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

    println!("{}", "Weight Analysis:".bold());
    println!(
        "  Tensors:     {} vs {}",
        original_weights.len(),
        quantized_weight_count
    );
    println!(
        "  Total size:  {:.2} KB vs {:.2} KB",
        original_size as f32 / 1024.0,
        quantized_size as f32 / 1024.0
    );

    let compression = original_size as f32 / quantized_size.max(1) as f32;
    println!("  Compression: {:.2}x smaller", compression);
    println!();

    println!("{}", "File Size:".bold());
    println!(
        "  Original:    {:.2} MB",
        original_file_size as f32 / 1_048_576.0
    );
    println!(
        "  Quantized:   {:.2} MB",
        quantized_file_size as f32 / 1_048_576.0
    );

    let file_compression = original_file_size as f64 / quantized_file_size.max(1) as f64;
    let size_reduction =
        (original_file_size as f64 - quantized_file_size as f64) / original_file_size.max(1) as f64 * 100.0;

    println!("  Reduction:   {:.1}%", size_reduction);
    println!("  Ratio:       {:.2}x", file_compression);
    println!();

    println!("{}", "Summary:".bold().green());

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

    if original_weights.len() == quantized_weight_count {
        println!("  ✓ All weights quantized");
    } else {
        println!(
            "  ⚠ Weight count mismatch ({} vs {})",
            original_weights.len(),
            quantized_weight_count
        );
    }

    println!();

    Ok(())
}

pub fn calibrate(
    input_path: &str,
    data_path: &str,
    output_path: &str,
    bits: u8,
    per_channel: bool,
    method_str: &str,
) -> Result<()> {
    println!("{}", "Calibration-Based Quantization".bold());
    println!("{}", "=".repeat(60));
    println!();
    
    // Parse calibration method
    let method: CalibrationMethod = method_str.parse()?;

    println!("Method: {}", format!("{}", method).cyan());
    println!();
    
    // Load model first so we can auto-detect shape
    println!("Loading model: {}", input_path);
    let model = OnnxModel::load(input_path)?;

    // Load calibration data
    println!("Loading calibration data: {}", data_path);
    let dataset = if data_path.ends_with(".npy") {
        CalibrationDataset::from_numpy(data_path)?
    } else {
        // Generate random data for testing
        println!("⚠  No .npy file provided, using random data for demo");
        let input_shape = model.input_shapes().into_iter().next()
            .and_then(|dims| {
                let shape: Vec<usize> = dims.into_iter()
                    .filter_map(|d| if d > 0 { Some(d as usize) } else { None })
                    .collect();
                // Skip batch dimension if present
                if shape.len() >= 2 {
                    Some(shape[1..].to_vec())
                } else if !shape.is_empty() {
                    Some(shape)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![3, 224, 224]);
        CalibrationDataset::random(input_shape, 100, (0.0, 1.0))?
    };

    println!("✓ Loaded {} samples", dataset.len());
    println!("  Sample shape: {:?}", dataset.sample_shape());
    println!();
    println!("✓ Model loaded");
    println!();
    
    // Run calibration
    println!("Running calibration...");
    let mut estimator = ActivationEstimator::new(model, input_path)?;
    estimator.calibrate(&dataset)?;
    println!();
    
    // Get calibration statistics
    let calib_stats = estimator.get_layer_stats()
        .into_iter()
        .map(|(k, v)| (k, v.clone()))
        .collect();
    
    // Extract weights
    println!("Extracting weights...");
    let mut model = estimator.into_model();
    let weights = model.extract_weights();
    println!("✓ Found {} weight tensors", weights.len());
    println!();
    
    // Quantize with calibration
    println!("Quantizing with calibration...");
    let config = QuantConfig {
        bits,
        per_channel,
        calibration_method: Some(method),
    };
    let quantizer = Quantizer::with_calibration(config, calib_stats);

    let results: Vec<_> = weights.par_iter()
        .map(|weight| {
            let quantized = quantizer.quantize_tensor_with_name(
                &weight.name,
                &weight.data,
                weight.shape.clone(),
            )?;

            let error = quantized.quantization_error(&weight.data);
            let (scales, zero_points) = quantized.get_all_scales_zero_points();
            let is_per_channel = quantized.is_per_channel();
            let bits_used = quantized.bits();

            let qdq = QdqWeightInput {
                original_name:    weight.name.clone(),
                quantized_values: quantized.data(),
                scales,
                zero_points,
                bits:             bits_used,
                axis:             if is_per_channel { Some(0) } else { None },
            };

            Ok::<_, anyhow::Error>((qdq, error))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut quantized_data = Vec::new();
    let mut total_error = 0.0_f32;
    for (qdq, error) in results {
        total_error += error;
        quantized_data.push(qdq);
    }

    let avg_error = if weights.is_empty() { 0.0 } else { total_error / weights.len() as f32 };
    
    println!();
    println!("Results:");
    println!("  Avg MSE error: {:.8}", avg_error);
    println!();
    
    // Save
    println!("Saving calibrated model...");
    model.save_quantized(&quantized_data, output_path)?;
    println!("✓ Saved to: {}", output_path.green());
    
    Ok(())
}

pub fn validate(original_path: &str, quantized_path: &str, detailed: bool) -> Result<()> {
    println!("{}", "Validating Quantized Model".bold());
    println!("{}", "=".repeat(60));
    println!();
    
    println!("Original:  {}", original_path.cyan());
    println!("Quantized: {}", quantized_path.yellow());
    println!();
    
    println!("Loading models...");
    let original_model = OnnxModel::load(original_path)
        .context("Failed to load original model")?;
    let quantized_model = OnnxModel::load(quantized_path)
        .context("Failed to load quantized model")?;
    
    println!("✓ Both models loaded successfully");
    println!();
    
    println!("{}", "Structure Validation".bold());
    println!("{}", "-".repeat(60));

    let original_info = original_model.info();
    let quantized_info = quantized_model.info();

    let mut validation_passed = true;

    // QDQ models add DequantizeLinear nodes and may remove weight inputs.
    // Detect QDQ by checking for DequantizeLinear nodes in the quantized model.
    let quantized_weight_info = quantized_model.load_quantized_info();
    let is_qdq = !quantized_weight_info.is_empty();
    let num_dq_nodes = quantized_weight_info.len();

    if is_qdq {
        let expected_nodes = original_info.num_nodes + num_dq_nodes;
        if quantized_info.num_nodes == expected_nodes {
            println!("✓ Node count: {} ({} original + {} DequantizeLinear)",
                     quantized_info.num_nodes, original_info.num_nodes, num_dq_nodes);
        } else {
            println!("⚠  Node count: {} (expected {} = {} + {} DQ nodes)",
                     quantized_info.num_nodes, expected_nodes,
                     original_info.num_nodes, num_dq_nodes);
        }
    } else if original_info.num_nodes == quantized_info.num_nodes {
        println!("✓ Node count matches: {}", original_info.num_nodes);
    } else {
        println!("✗ Node count mismatch: {} vs {}",
                 original_info.num_nodes, quantized_info.num_nodes);
        validation_passed = false;
    }

    // QDQ removes weight names from graph.input to avoid "duplicate definition"
    if is_qdq {
        let expected_inputs = original_info.inputs.len().saturating_sub(num_dq_nodes);
        if quantized_info.inputs.len() >= expected_inputs {
            println!("✓ Input count: {} (weight inputs removed for QDQ)",
                     quantized_info.inputs.len());
        } else {
            println!("⚠  Input count: {} (expected >= {})",
                     quantized_info.inputs.len(), expected_inputs);
        }
    } else if original_info.inputs.len() == quantized_info.inputs.len() {
        println!("✓ Input count matches: {}", original_info.inputs.len());
    } else {
        println!("✗ Input count mismatch: {} vs {}",
                 original_info.inputs.len(), quantized_info.inputs.len());
        validation_passed = false;
    }

    if original_info.outputs.len() == quantized_info.outputs.len() {
        println!("✓ Output count matches: {}", original_info.outputs.len());
    } else {
        println!("✗ Output count mismatch: {} vs {}",
                 original_info.outputs.len(), quantized_info.outputs.len());
        validation_passed = false;
    }

    println!();

    println!("{}", "Graph Connectivity".bold());
    println!("{}", "-".repeat(60));

    let connectivity = quantized_model.validate_connectivity();
    print!("{}", connectivity.summary());
    if !connectivity.valid {
        validation_passed = false;
    }

    println!();

    println!("{}", "Weight Validation".bold());
    println!("{}", "-".repeat(60));

    let original_weights = original_model.extract_weights();

    if is_qdq {
        // For QDQ models, use load_quantized_info which understands the QDQ format
        println!("✓ QDQ format detected: {} quantized weight tensors", quantized_weight_info.len());

        if original_weights.len() == quantized_weight_info.len() {
            println!("✓ All {} original weights have quantized counterparts",
                     original_weights.len());
        } else {
            println!("⚠  {} original weights, {} quantized ({} unquantized)",
                     original_weights.len(), quantized_weight_info.len(),
                     original_weights.len().saturating_sub(quantized_weight_info.len()));
        }

        // Check scale/zero-point sanity
        let mut bad_scales = 0;
        for qw in &quantized_weight_info {
            if qw.scale <= 0.0 || !qw.scale.is_finite() {
                bad_scales += 1;
                if detailed {
                    println!("⚠  Bad scale for '{}': {}", qw.name, qw.scale);
                }
            }
        }
        if bad_scales > 0 {
            println!("⚠  {} weights have invalid scales", bad_scales);
            validation_passed = false;
        } else {
            println!("✓ All quantization scales are valid");
        }
    } else {
        // Non-QDQ: compare weight tensors directly
        let quantized_weights = quantized_model.extract_weights();

        if original_weights.len() == quantized_weights.len() {
            println!("✓ Weight tensor count matches: {}", original_weights.len());
        } else {
            println!("⚠  Weight tensor count differs: {} vs {}",
                     original_weights.len(), quantized_weights.len());
        }

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
    }
    
    println!();
    
    println!("{}", "Size Analysis".bold());
    println!("{}", "-".repeat(60));
    
    let original_size = std::fs::metadata(original_path)?.len();
    let quantized_size = std::fs::metadata(quantized_path)?.len();
    let compression = original_size as f64 / quantized_size.max(1) as f64;
    let reduction = (original_size as f64 - quantized_size as f64) / original_size.max(1) as f64 * 100.0;

    println!("Original:  {:.2} MB", original_size as f64 / 1_048_576.0);
    println!("Quantized: {:.2} MB", quantized_size as f64 / 1_048_576.0);
    println!("Reduction: {:.1}% ({:.2}x smaller)", reduction, compression);

    if reduction < 0.0 {
        println!("⚠  Warning: Quantized model is larger than original (QDQ overhead on small models).");
    } else if compression < 2.0 {
        println!("⚠  Warning: Low compression ratio. Quantization may not be working correctly.");
    } else if (3.5..=4.5).contains(&compression) {
        println!("✓ Expected compression for INT8 quantization");
    }
    
    println!();
    
    if detailed {
        println!("{}", "Detailed Layer Analysis".bold());
        println!("{}", "-".repeat(60));

        // Detect bit width from QDQ info if available
        let detected_bits = if is_qdq && !quantized_weight_info.is_empty() {
            quantized_weight_info[0].bits
        } else {
            8
        };
        let config = QuantConfig { bits: detected_bits, per_channel: false, calibration_method: None };
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

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    println!("Output directory: {}", output_dir.cyan());
    println!("Quantization: INT{}{}", bits, if per_channel { " (per-channel)" } else { "" });
    println!();

    let mut input_files = Vec::new();
    for pattern in inputs {
        if Path::new(pattern).exists() {
            input_files.push(PathBuf::from(pattern));
        } else {
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

    let mut results = Vec::new();
    let total = input_files.len();

    for (idx, input_path) in input_files.iter().enumerate() {
        let input_str = input_path.to_string_lossy();
        let filename = input_path.file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| input_path.to_string_lossy().to_string());
        
        let output_filename = if let Some(stem) = input_path.file_stem() {
            format!("{}_int{}.onnx", stem.to_string_lossy(), bits)
        } else {
            format!("{}_int{}.onnx", filename, bits)
        };
        
        let output_path = Path::new(output_dir).join(&output_filename);
        let output_str = output_path.to_string_lossy();

        println!("{}", format!("[{}/{}]", idx + 1, total).bold());
        println!("Processing: {}", filename.cyan());

        if skip_existing && output_path.exists() {
            println!("Skipped (already exists)");
            println!();
            results.push((input_str.to_string(), "Skipped".to_string()));
            continue;
        }

        match quantize(&input_str, &output_str, bits, per_channel) {
            Ok(_) => {
                println!("✓ Success");
                println!();
                results.push((input_str.to_string(), "Success".to_string()));
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
                println!();
                results.push((input_str.to_string(), format!("Failed: {}", e)));
                
                if !continue_on_error {
                    println!("✗ Stopping batch processing due to error");
                    println!("   Use --continue-on-error to process remaining models");
                    break;
                }
            }
        }
    }

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

    println!("Loading config: {}", config_path.cyan());
    let config = Config::from_file(config_path)
        .context("Failed to load configuration file")?;

    config.validate()
        .context("Config validation failed")?;

    println!("✓ Config loaded and validated");
    println!();

    if dry_run {
        println!("{}", "DRY RUN MODE - No files will be modified".yellow().bold());
        println!();
    }

    println!("Global settings:");
    println!("  Bits: {}", config.bits);
    println!("  Per-channel: {}", config.per_channel);
    println!();

    let mut total_tasks = 0;

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