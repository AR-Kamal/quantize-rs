//src/cli/commands.rs
use quantize_rs::config::Config;

use anyhow::{Context, Result};
use colored::Colorize;
#[cfg(feature = "calibration")]
use quantize_rs::calibration::{
    methods::CalibrationMethod, ActivationEstimator, CalibrationDataset,
};
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use quantize_rs::onnx_utils::{OnnxModel, SaveOptions};
use quantize_rs::quantization::{QuantConfig, Quantizer};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[allow(clippy::too_many_arguments)]
pub fn quantize(
    input: &str,
    output: &str,
    bits: u8,
    per_channel: bool,
    excluded_layers: &[String],
    min_elements: usize,
    layer_bits: &std::collections::HashMap<String, u8>,
    native_int4: bool,
    symmetric: bool,
) -> Result<()> {
    quantize_inner(
        input,
        output,
        bits,
        per_channel,
        excluded_layers,
        min_elements,
        layer_bits,
        native_int4,
        symmetric,
        false,
    )
}

/// Same as [`quantize`] but suppresses all per-file progress output when
/// `quiet` is true.  Used by [`batch`] under `--jobs > 1` where interleaved
/// prints from multiple workers would be unreadable.
#[allow(clippy::too_many_arguments)]
fn quantize_inner(
    input: &str,
    output: &str,
    bits: u8,
    per_channel: bool,
    excluded_layers: &[String],
    min_elements: usize,
    layer_bits: &std::collections::HashMap<String, u8>,
    native_int4: bool,
    symmetric: bool,
    quiet: bool,
) -> Result<()> {
    macro_rules! p {
        () => { if !quiet { println!(); } };
        ($($tt:tt)*) => { if !quiet { println!($($tt)*); } };
    }

    p!("Loading model: {}", input.bold());

    let mut model = OnnxModel::load(input)?;
    let info = model.info();

    p!("✓ Model loaded");
    p!("  Nodes: {}", info.num_nodes);
    p!();

    p!("Extracting weights...");
    let weights = model.extract_weights();
    let original_size: usize = weights.iter().map(|w| w.size_bytes()).sum();

    if weights.is_empty() {
        p!("⚠  No weights found to quantize!");
        return Ok(());
    }

    p!("✓ Found {} weight tensors", weights.len());
    p!(
        "  Original size: {:.2} MB",
        original_size as f32 / 1_048_576.0
    );
    p!();

    if !layer_bits.is_empty() {
        p!(
            "Quantizing (mixed precision, default INT{}){}...",
            bits,
            if per_channel { " per-channel" } else { "" }
        );
        p!(
            "  Layer overrides: {} layer(s) with custom bit-width",
            layer_bits.len()
        );
    } else if per_channel {
        p!("Quantizing to INT{} (per-channel)...", bits);
    } else {
        p!("Quantizing to INT{}...", bits);
    }

    if !excluded_layers.is_empty() {
        p!("  Excluded layers: {}", excluded_layers.join(", "));
    }
    if min_elements > 0 {
        p!("  Min elements:    {}", min_elements);
    }

    let config = QuantConfig {
        bits,
        per_channel,
        symmetric,
        calibration_method: None,
        excluded_layers: excluded_layers.to_vec(),
        min_elements,
        layer_bits: layer_bits.clone(),
    };

    // Shared helper: filter, parallel-quantize, honour per-layer overrides.
    let outputs = Quantizer::new(config).quantize_model(&model)?;

    let skipped = weights.len() - outputs.len();
    if skipped > 0 {
        p!(
            "  Skipping {} layer(s) (excluded or below min-elements threshold)",
            skipped
        );
    }
    if outputs.is_empty() {
        p!(
            "{}",
            "⚠  All weight tensors were excluded — no quantization performed.".yellow()
        );
        p!("  Check --exclude, --min-elements, and layer_bits settings.");
        return Ok(());
    }

    let total_error: f32 = outputs.iter().map(|o| o.mse).sum();
    let quantized_size: usize = outputs.iter().map(|o| o.quantized_size_bytes).sum();
    let avg_error = total_error / outputs.len() as f32;
    let compression_ratio = original_size as f32 / quantized_size.max(1) as f32;
    let int8_count = outputs.iter().filter(|o| o.qdq.bits == 8).count();
    let int4_count = outputs.iter().filter(|o| o.qdq.bits == 4).count();

    let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

    p!("✓ Quantization complete");
    p!();
    p!("Results:");
    p!(
        "  Quantized:        {}/{} tensors",
        quantized_data.len(),
        weights.len()
    );
    if int8_count > 0 && int4_count > 0 {
        p!("  INT8 layers:      {}", int8_count);
        p!("  INT4 layers:      {}", int4_count);
    }
    p!(
        "  Original size:    {:.2} MB",
        original_size as f32 / 1_048_576.0
    );
    p!(
        "  Quantized size:   {:.2} MB",
        quantized_size as f32 / 1_048_576.0
    );
    p!("  Compression:      {:.2}x smaller", compression_ratio);
    p!("  Avg MSE error:    {:.6}", avg_error);
    p!();

    let save_options = SaveOptions::default().with_native_int4(native_int4);
    if native_int4 && int4_count > 0 {
        p!("Saving quantized model (native INT4, opset 21)...");
    } else {
        p!("Saving quantized model...");
    }
    model.save_quantized_with_options(&quantized_data, output, save_options)?;
    p!("✓ Saved to: {}", output.green());

    p!("Validating saved model...");
    match OnnxModel::load(output) {
        Ok(_) => p!("✓ Model validation passed"),
        Err(e) => p!("⚠  Warning: Could not validate saved model: {}", e),
    }

    Ok(())
}

#[derive(Serialize)]
struct InfoReport<'a> {
    path: &'a str,
    name: String,
    version: i64,
    num_nodes: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

pub fn info(input: &str, format: &str) -> Result<()> {
    let model = OnnxModel::load(input)?;
    let info = model.info();

    if format == "json" {
        let report = InfoReport {
            path: input,
            name: info.name,
            version: info.version,
            num_nodes: info.num_nodes,
            inputs: info.inputs,
            outputs: info.outputs,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("Model Information: {}", input.bold());
    println!();
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

#[derive(Serialize)]
struct BenchmarkReport<'a> {
    original_path: &'a str,
    quantized_path: &'a str,
    original_nodes: usize,
    quantized_nodes: usize,
    original_inputs: usize,
    quantized_inputs: usize,
    original_outputs: usize,
    quantized_outputs: usize,
    is_qdq: bool,
    original_weight_count: usize,
    quantized_weight_count: usize,
    original_weight_size_bytes: usize,
    quantized_weight_size_bytes: usize,
    weight_compression_ratio: f32,
    original_file_size_bytes: u64,
    quantized_file_size_bytes: u64,
    file_compression_ratio: f64,
    file_size_reduction_percent: f64,
    structure_preserved: bool,
}

pub fn benchmark(original: &str, quantized: &str, format: &str) -> Result<()> {
    let json_mode = format == "json";

    if !json_mode {
        println!("Benchmarking Models");
        println!("{}", "=".repeat(50));
        println!();
        println!("Original:  {}", original.cyan());
        println!("Quantized: {}", quantized.yellow());
        println!();
        println!("Loading models...");
    }

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
        // Use the actual on-disk byte count, which accounts for native INT4
        // packing (two values per byte) vs INT8 storage (one value per byte).
        let size: usize = quantized_weight_info.iter().map(|w| w.storage_bytes).sum();
        (count, size)
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

    let file_compression = original_file_size as f64 / quantized_file_size.max(1) as f64;
    let size_reduction = (original_file_size as f64 - quantized_file_size as f64)
        / original_file_size.max(1) as f64
        * 100.0;
    let compression = original_size as f32 / quantized_size.max(1) as f32;
    // QDQ adds one DequantizeLinear per quantized weight and removes the
    // weight from `graph.input` (when it appeared there).  Compare against
    // the expected post-transform shape, not the pre-transform shape.
    let num_dq_nodes = quantized_weight_info.len();
    let expected_quantized_nodes = if is_qdq {
        original_info.num_nodes + num_dq_nodes
    } else {
        original_info.num_nodes
    };
    let expected_quantized_inputs = if is_qdq {
        original_info.inputs.len().saturating_sub(num_dq_nodes)
    } else {
        original_info.inputs.len()
    };
    let structure_match = quantized_info.num_nodes == expected_quantized_nodes
        && quantized_info.inputs.len() >= expected_quantized_inputs
        && original_info.outputs.len() == quantized_info.outputs.len();

    if json_mode {
        let report = BenchmarkReport {
            original_path: original,
            quantized_path: quantized,
            original_nodes: original_info.num_nodes,
            quantized_nodes: quantized_info.num_nodes,
            original_inputs: original_info.inputs.len(),
            quantized_inputs: quantized_info.inputs.len(),
            original_outputs: original_info.outputs.len(),
            quantized_outputs: quantized_info.outputs.len(),
            is_qdq,
            original_weight_count: original_weights.len(),
            quantized_weight_count,
            original_weight_size_bytes: original_size,
            quantized_weight_size_bytes: quantized_size,
            weight_compression_ratio: compression,
            original_file_size_bytes: original_file_size,
            quantized_file_size_bytes: quantized_file_size,
            file_compression_ratio: file_compression,
            file_size_reduction_percent: size_reduction,
            structure_preserved: structure_match,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

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
    println!("  Reduction:   {:.1}%", size_reduction);
    println!("  Ratio:       {:.2}x", file_compression);
    println!();

    println!("{}", "Summary:".bold().green());

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

#[cfg(feature = "calibration")]
#[allow(clippy::too_many_arguments)]
pub fn calibrate(
    input_path: &str,
    data_path: &str,
    output_path: &str,
    bits: u8,
    per_channel: bool,
    method_str: &str,
    excluded_layers: &[String],
    min_elements: usize,
    layer_bits: &std::collections::HashMap<String, u8>,
    native_int4: bool,
    symmetric: bool,
) -> Result<()> {
    println!("{}", "Calibration-Based Quantization".bold());
    println!("{}", "=".repeat(60));
    println!();

    // Parse calibration method
    let method: CalibrationMethod = method_str.parse()?;

    println!("Method: {}", format!("{}", method).cyan());
    if !excluded_layers.is_empty() {
        println!("Excluded layers: {}", excluded_layers.join(", "));
    }
    if min_elements > 0 {
        println!("Min elements:    {}", min_elements);
    }
    if !layer_bits.is_empty() {
        println!("Layer overrides: {} layer(s)", layer_bits.len());
    }
    if symmetric {
        println!("Symmetric:       true");
    }
    if native_int4 {
        println!("Native INT4:     true (opset 21)");
    }
    println!();

    // Load model first so we can auto-detect shape
    println!("Loading model: {}", input_path);
    let model = OnnxModel::load(input_path)?;

    // Load calibration data
    println!("Loading calibration data: {}", data_path);
    let dataset = if data_path.ends_with(".npy") {
        CalibrationDataset::from_numpy(data_path)?
    } else if data_path.ends_with(".safetensors") {
        #[cfg(feature = "safetensors-input")]
        {
            CalibrationDataset::from_safetensors(data_path)?
        }
        #[cfg(not(feature = "safetensors-input"))]
        {
            return Err(anyhow::anyhow!(
                ".safetensors input requires building with --features safetensors-input"
            ));
        }
    } else {
        // Generate random data for testing
        println!("⚠  No .npy file provided, using random data for demo");
        let input_shape = model
            .input_shapes()
            .into_iter()
            .next()
            .and_then(|dims| {
                let shape: Vec<usize> = dims
                    .into_iter()
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
    let calib_stats = estimator
        .get_layer_stats()
        .into_iter()
        .map(|(k, v)| (k, v.clone()))
        .collect();

    // Extract weights
    println!("Extracting weights...");
    let mut model = estimator.into_model();
    let weights = model.extract_weights();
    println!("✓ Found {} weight tensors", weights.len());
    println!();

    // Quantize with calibration — honours the same selection filters as
    // `quantize` (excluded_layers, min_elements, layer_bits, symmetric).
    println!("Quantizing with calibration...");
    let config = QuantConfig {
        bits,
        per_channel,
        symmetric,
        calibration_method: Some(method),
        excluded_layers: excluded_layers.to_vec(),
        min_elements,
        layer_bits: layer_bits.clone(),
    };

    let outputs = Quantizer::with_calibration(config, calib_stats).quantize_model(&model)?;

    let skipped = weights.len() - outputs.len();
    if skipped > 0 {
        println!(
            "  Skipping {} layer(s) (excluded or below min-elements)",
            skipped
        );
    }

    let total_error: f32 = outputs.iter().map(|o| o.mse).sum();
    let avg_error = if outputs.is_empty() {
        0.0
    } else {
        total_error / outputs.len() as f32
    };
    let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

    println!();
    println!("Results:");
    println!(
        "  Quantized:     {}/{} tensors",
        quantized_data.len(),
        weights.len()
    );
    println!("  Avg MSE error: {:.8}", avg_error);
    println!();

    // Save
    let save_options = SaveOptions::default().with_native_int4(native_int4);
    println!("Saving calibrated model...");
    model.save_quantized_with_options(&quantized_data, output_path, save_options)?;
    println!("✓ Saved to: {}", output_path.green());

    Ok(())
}

#[derive(Serialize)]
struct ValidateReport<'a> {
    original_path: &'a str,
    quantized_path: &'a str,
    original_nodes: usize,
    quantized_nodes: usize,
    original_inputs: usize,
    quantized_inputs: usize,
    original_outputs: usize,
    quantized_outputs: usize,
    is_qdq: bool,
    num_dq_nodes: usize,
    connectivity_valid: bool,
    broken_refs: Vec<String>,
    original_weight_count: usize,
    quantized_weight_count: usize,
    num_bad_scales: usize,
    original_file_size_bytes: u64,
    quantized_file_size_bytes: u64,
    compression_ratio: f64,
    size_reduction_percent: f64,
    validation_passed: bool,
}

pub fn validate(
    original_path: &str,
    quantized_path: &str,
    detailed: bool,
    format: &str,
) -> Result<()> {
    let json_mode = format == "json";

    // Local macro: gate human-mode prints.  Captures `json_mode` from the
    // enclosing function.
    macro_rules! phuman {
        () => { if !json_mode { println!(); } };
        ($($tt:tt)*) => { if !json_mode { println!($($tt)*); } };
    }
    macro_rules! phuman_inline {
        ($($tt:tt)*) => { if !json_mode { print!($($tt)*); } };
    }

    phuman!("{}", "Validating Quantized Model".bold());
    phuman!("{}", "=".repeat(60));
    phuman!();
    phuman!("Original:  {}", original_path.cyan());
    phuman!("Quantized: {}", quantized_path.yellow());
    phuman!();
    phuman!("Loading models...");

    let original_model = OnnxModel::load(original_path).context("Failed to load original model")?;
    let quantized_model =
        OnnxModel::load(quantized_path).context("Failed to load quantized model")?;

    phuman!("✓ Both models loaded successfully");
    phuman!();
    phuman!("{}", "Structure Validation".bold());
    phuman!("{}", "-".repeat(60));

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
            phuman!(
                "✓ Node count: {} ({} original + {} DequantizeLinear)",
                quantized_info.num_nodes,
                original_info.num_nodes,
                num_dq_nodes
            );
        } else {
            phuman!(
                "⚠  Node count: {} (expected {} = {} + {} DQ nodes)",
                quantized_info.num_nodes,
                expected_nodes,
                original_info.num_nodes,
                num_dq_nodes
            );
        }
    } else if original_info.num_nodes == quantized_info.num_nodes {
        phuman!("✓ Node count matches: {}", original_info.num_nodes);
    } else {
        phuman!(
            "✗ Node count mismatch: {} vs {}",
            original_info.num_nodes,
            quantized_info.num_nodes
        );
        validation_passed = false;
    }

    // QDQ removes weight names from graph.input to avoid "duplicate definition"
    if is_qdq {
        let expected_inputs = original_info.inputs.len().saturating_sub(num_dq_nodes);
        if quantized_info.inputs.len() >= expected_inputs {
            phuman!(
                "✓ Input count: {} (weight inputs removed for QDQ)",
                quantized_info.inputs.len()
            );
        } else {
            phuman!(
                "⚠  Input count: {} (expected >= {})",
                quantized_info.inputs.len(),
                expected_inputs
            );
        }
    } else if original_info.inputs.len() == quantized_info.inputs.len() {
        phuman!("✓ Input count matches: {}", original_info.inputs.len());
    } else {
        phuman!(
            "✗ Input count mismatch: {} vs {}",
            original_info.inputs.len(),
            quantized_info.inputs.len()
        );
        validation_passed = false;
    }

    if original_info.outputs.len() == quantized_info.outputs.len() {
        phuman!("✓ Output count matches: {}", original_info.outputs.len());
    } else {
        phuman!(
            "✗ Output count mismatch: {} vs {}",
            original_info.outputs.len(),
            quantized_info.outputs.len()
        );
        validation_passed = false;
    }

    phuman!();

    phuman!("{}", "Graph Connectivity".bold());
    phuman!("{}", "-".repeat(60));

    let connectivity = quantized_model.validate_connectivity();
    phuman_inline!("{}", connectivity.summary());
    if !connectivity.valid {
        validation_passed = false;
    }

    phuman!();

    phuman!("{}", "Weight Validation".bold());
    phuman!("{}", "-".repeat(60));

    let original_weights = original_model.extract_weights();

    let mut quantized_weight_count_for_report = quantized_weight_info.len();
    let mut bad_scales = 0usize;

    if is_qdq {
        // For QDQ models, use load_quantized_info which understands the QDQ format
        phuman!(
            "✓ QDQ format detected: {} quantized weight tensors",
            quantized_weight_info.len()
        );

        if original_weights.len() == quantized_weight_info.len() {
            phuman!(
                "✓ All {} original weights have quantized counterparts",
                original_weights.len()
            );
        } else {
            phuman!(
                "⚠  {} original weights, {} quantized ({} unquantized)",
                original_weights.len(),
                quantized_weight_info.len(),
                original_weights
                    .len()
                    .saturating_sub(quantized_weight_info.len())
            );
        }

        // Check scale/zero-point sanity — iterate all per-channel values.
        for qw in &quantized_weight_info {
            for (ch, &s) in qw.scales.iter().enumerate() {
                if s <= 0.0 || !s.is_finite() {
                    bad_scales += 1;
                    if detailed {
                        if qw.is_per_channel() {
                            phuman!("⚠  Bad scale for '{}' channel {}: {}", qw.name, ch, s);
                        } else {
                            phuman!("⚠  Bad scale for '{}': {}", qw.name, s);
                        }
                    }
                }
            }
        }
        if bad_scales > 0 {
            phuman!("⚠  {} weights have invalid scales", bad_scales);
            validation_passed = false;
        } else {
            phuman!("✓ All quantization scales are valid");
        }
    } else {
        // Non-QDQ: compare weight tensors directly
        let quantized_weights = quantized_model.extract_weights();
        quantized_weight_count_for_report = quantized_weights.len();

        if original_weights.len() == quantized_weights.len() {
            phuman!("✓ Weight tensor count matches: {}", original_weights.len());
        } else {
            phuman!(
                "⚠  Weight tensor count differs: {} vs {}",
                original_weights.len(),
                quantized_weights.len()
            );
        }

        let mut shape_mismatches = 0;
        for (orig, quant) in original_weights.iter().zip(quantized_weights.iter()) {
            if orig.shape != quant.shape {
                shape_mismatches += 1;
                if detailed {
                    phuman!(
                        "⚠  Shape mismatch in '{}': {:?} vs {:?}",
                        orig.name,
                        orig.shape,
                        quant.shape
                    );
                }
            }
        }
        if shape_mismatches == 0 {
            phuman!("✓ All weight shapes match");
        } else {
            phuman!(
                "⚠  {} weight tensors have shape mismatches",
                shape_mismatches
            );
            validation_passed = false;
        }
    }

    phuman!();

    phuman!("{}", "Size Analysis".bold());
    phuman!("{}", "-".repeat(60));

    let original_size = std::fs::metadata(original_path)?.len();
    let quantized_size = std::fs::metadata(quantized_path)?.len();
    let compression = original_size as f64 / quantized_size.max(1) as f64;
    let reduction =
        (original_size as f64 - quantized_size as f64) / original_size.max(1) as f64 * 100.0;

    phuman!("Original:  {:.2} MB", original_size as f64 / 1_048_576.0);
    phuman!("Quantized: {:.2} MB", quantized_size as f64 / 1_048_576.0);
    phuman!("Reduction: {:.1}% ({:.2}x smaller)", reduction, compression);

    if reduction < 0.0 {
        phuman!(
            "⚠  Warning: Quantized model is larger than original (QDQ overhead on small models)."
        );
    } else if compression < 2.0 {
        phuman!("⚠  Warning: Low compression ratio. Quantization may not be working correctly.");
    } else if (3.5..=4.5).contains(&compression) {
        phuman!("✓ Expected compression for INT8 quantization");
    }

    phuman!();

    if detailed && !json_mode {
        println!("{}", "Detailed Layer Analysis".bold());
        println!("{}", "-".repeat(68));

        // Build a per-layer bit map from QDQ metadata so mixed-precision
        // models show the correct error per layer.
        let layer_bits_map: std::collections::HashMap<String, u8> = if is_qdq {
            quantized_weight_info
                .iter()
                .map(|qw| (qw.name.clone(), qw.bits))
                .collect()
        } else {
            std::collections::HashMap::new()
        };
        let default_bits = layer_bits_map.values().next().copied().unwrap_or(8);

        println!(
            "{:<40} {:>5} {:>12} {:>15}",
            "Layer", "Bits", "Elements", "MSE Error"
        );
        println!("{}", "-".repeat(75));

        for weight in original_weights.iter().take(10) {
            let weight_bits = layer_bits_map
                .get(&weight.name)
                .copied()
                .unwrap_or(default_bits);
            let wconfig = QuantConfig {
                bits: weight_bits,
                per_channel: false,
                calibration_method: None,
                ..Default::default()
            };
            if let Ok(quantized) =
                Quantizer::new(wconfig).quantize_tensor(&weight.data, weight.shape.clone())
            {
                let error = quantized.quantization_error(&weight.data);

                let name = if weight.name.len() > 37 {
                    format!("{}...", &weight.name[..37])
                } else {
                    weight.name.clone()
                };

                println!(
                    "{:<40} {:>5} {:>12} {:>15.8}",
                    name,
                    weight_bits,
                    weight.data.len(),
                    error
                );
            }
        }

        if original_weights.len() > 10 {
            println!("... ({} more layers)", original_weights.len() - 10);
        }

        println!();
    }

    if json_mode {
        let report = ValidateReport {
            original_path,
            quantized_path,
            original_nodes: original_info.num_nodes,
            quantized_nodes: quantized_info.num_nodes,
            original_inputs: original_info.inputs.len(),
            quantized_inputs: quantized_info.inputs.len(),
            original_outputs: original_info.outputs.len(),
            quantized_outputs: quantized_info.outputs.len(),
            is_qdq,
            num_dq_nodes,
            connectivity_valid: connectivity.valid,
            broken_refs: connectivity.broken_refs.clone(),
            original_weight_count: original_weights.len(),
            quantized_weight_count: quantized_weight_count_for_report,
            num_bad_scales: bad_scales,
            original_file_size_bytes: original_size,
            quantized_file_size_bytes: quantized_size,
            compression_ratio: compression,
            size_reduction_percent: reduction,
            validation_passed,
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
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

#[allow(clippy::too_many_arguments)]
pub fn batch(
    inputs: &[String],
    output_dir: &str,
    bits: u8,
    per_channel: bool,
    skip_existing: bool,
    continue_on_error: bool,
    excluded_layers: &[String],
    min_elements: usize,
    layer_bits: &std::collections::HashMap<String, u8>,
    native_int4: bool,
    symmetric: bool,
    jobs: usize,
) -> Result<()> {
    println!("{}", "Batch Quantization".bold());
    println!("{}", "=".repeat(60));
    println!();

    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    let effective_jobs = if jobs == 0 { 1 } else { jobs };
    println!("Output directory: {}", output_dir.cyan());
    println!(
        "Quantization: INT{}{}",
        bits,
        if per_channel { " (per-channel)" } else { "" }
    );
    if effective_jobs > 1 {
        println!("Parallel jobs: {}", effective_jobs);
    }
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
                Err(e) => {
                    println!("⚠  Invalid glob pattern '{}': {}", pattern, e);
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

    // Build per-file plan: (input_path, output_path, filename).
    let total = input_files.len();
    let plans: Vec<(PathBuf, PathBuf, String)> = input_files
        .iter()
        .map(|p| {
            let filename = p
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| p.to_string_lossy().to_string());
            let output_filename = if let Some(stem) = p.file_stem() {
                format!("{}_int{}.onnx", stem.to_string_lossy(), bits)
            } else {
                format!("{}_int{}.onnx", filename, bits)
            };
            let output_path = Path::new(output_dir).join(&output_filename);
            (p.clone(), output_path, filename)
        })
        .collect();

    let mut results: Vec<(String, String)> = Vec::with_capacity(total);

    if effective_jobs <= 1 {
        // Serial path — unchanged human output, can stop on first error.
        for (idx, (input_path, output_path, filename)) in plans.iter().enumerate() {
            let input_str = input_path.to_string_lossy();
            let output_str = output_path.to_string_lossy();

            println!("{}", format!("[{}/{}]", idx + 1, total).bold());
            println!("Processing: {}", filename.cyan());

            if skip_existing && output_path.exists() {
                println!("Skipped (already exists)");
                println!();
                results.push((input_str.to_string(), "Skipped".to_string()));
                continue;
            }

            match quantize(
                &input_str,
                &output_str,
                bits,
                per_channel,
                excluded_layers,
                min_elements,
                layer_bits,
                native_int4,
                symmetric,
            ) {
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
    } else {
        // Parallel path — per-file progress is suppressed (workers would
        // interleave).  Each file prints one atomic status line when done.
        use rayon::prelude::*;
        use std::sync::Mutex;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(effective_jobs)
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build rayon thread pool: {e}"))?;
        let stdout = Mutex::new(());
        let collected: Vec<(String, String)> = pool.install(|| {
            plans
                .par_iter()
                .enumerate()
                .map(|(idx, (input_path, output_path, filename))| {
                    let input_str = input_path.to_string_lossy().to_string();
                    let output_str = output_path.to_string_lossy().to_string();

                    if skip_existing && output_path.exists() {
                        let _guard = stdout.lock().unwrap();
                        println!(
                            "{} {} (skipped — output exists)",
                            format!("[{}/{}]", idx + 1, total).bold(),
                            filename.cyan()
                        );
                        return (input_str, "Skipped".to_string());
                    }

                    let res = quantize_inner(
                        &input_str,
                        &output_str,
                        bits,
                        per_channel,
                        excluded_layers,
                        min_elements,
                        layer_bits,
                        native_int4,
                        symmetric,
                        /* quiet */ true,
                    );
                    let _guard = stdout.lock().unwrap();
                    match res {
                        Ok(_) => {
                            println!(
                                "{} {} ✓",
                                format!("[{}/{}]", idx + 1, total).bold(),
                                filename.cyan()
                            );
                            (input_str, "Success".to_string())
                        }
                        Err(e) => {
                            println!(
                                "{} {} ✗ {}",
                                format!("[{}/{}]", idx + 1, total).bold(),
                                filename.cyan(),
                                e
                            );
                            (input_str, format!("Failed: {}", e))
                        }
                    }
                })
                .collect()
        });
        results.extend(collected);
        // In parallel mode `continue_on_error` is effectively always on:
        // all workers finish before we can summarize.  Document this in the
        // summary if any failed.
        if !continue_on_error && results.iter().any(|(_, s)| s.starts_with("Failed")) {
            println!(
                "(--jobs > 1 runs all inputs to completion regardless of --continue-on-error)"
            );
        }
        println!();
    }

    println!("{}", "=".repeat(60));
    println!("{}", "Batch Summary".bold());
    println!("{}", "=".repeat(60));
    println!();

    let success_count = results
        .iter()
        .filter(|(_, status)| status.contains("Success"))
        .count();
    let skipped_count = results
        .iter()
        .filter(|(_, status)| status.contains("Skipped"))
        .count();
    let failed_count = results
        .iter()
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

    if failed_count == 0 && success_count > 0 {
        println!("{}", "✓ All models processed successfully!".green().bold());
    } else if failed_count == 0 && skipped_count > 0 {
        // No work done, no failures — report neutrally instead of pretending
        // success.
        println!(
            "{}",
            format!("• All {} model(s) skipped — nothing to do.", skipped_count)
                .yellow()
                .bold()
        );
    } else if failed_count == 0 {
        println!("{}", "• No models matched.".yellow().bold());
    } else if success_count > 0 {
        println!(
            "{}",
            "⚠  Some models failed, but others succeeded"
                .yellow()
                .bold()
        );
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
    let config = Config::from_file(config_path).context("Failed to load configuration file")?;

    config.validate().context("Config validation failed")?;

    println!("✓ Config loaded and validated");
    println!();

    if dry_run {
        println!(
            "{}",
            "DRY RUN MODE - No files will be modified".yellow().bold()
        );
        println!();
    }

    println!("Global settings:");
    println!("  Bits: {}", config.bits);
    println!("  Per-channel: {}", config.per_channel);
    println!();

    let mut total_tasks = 0;

    if !config.models.is_empty() {
        println!(
            "{}",
            format!("Individual Models: {}", config.models.len()).bold()
        );
        println!("{}", "-".repeat(60));

        for (idx, model_config) in config.models.iter().enumerate() {
            let bits = config.get_bits(model_config);
            let per_channel = config.get_per_channel(model_config);
            let excluded = config.get_excluded_layers(model_config);
            let min_elements = config.get_min_elements(model_config);
            let layer_bits = config.get_layer_bits(model_config);
            let native_int4 = config.get_native_int4(model_config);
            let symmetric = config.get_symmetric(model_config);

            println!();
            println!("[{}] {}", idx + 1, model_config.input.cyan());
            println!("    → {}", model_config.output.green());
            println!("    Bits: {}, Per-channel: {}", bits, per_channel);
            if !layer_bits.is_empty() {
                println!("    Layer overrides: {} layer(s)", layer_bits.len());
            }

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

            match quantize(
                &model_config.input,
                &model_config.output,
                bits,
                per_channel,
                &excluded,
                min_elements,
                &layer_bits,
                native_int4,
                symmetric,
            ) {
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
                &config.excluded_layers,
                config.min_elements,
                &Default::default(),
                config.native_int4,
                config.symmetric,
                /* jobs */ 1,
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
