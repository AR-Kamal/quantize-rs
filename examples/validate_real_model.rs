//! Quantize and validate a real ONNX model.
//!
//! This example loads any ONNX file, quantizes its weights, reports quality
//! metrics, and optionally saves the quantized model.
//!
//! # Usage
//!
//! ```text
//! cargo run --example validate_real_model -- model.onnx
//! cargo run --example validate_real_model -- model.onnx --bits 4 --per-channel --output quantized.onnx
//! ```
//!
//! # Output
//!
//! For each weight tensor the example prints:
//! - element count and shape
//! - mean absolute quantization error (MAE)
//! - quantized byte size vs. original FP32 byte size
//!
//! At the end it prints a summary with totals and the overall compression ratio.

use clap::Parser;
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "validate_real_model",
    about = "Quantize and validate a real ONNX model",
    long_about = None,
)]
struct Args {
    /// Path to the input ONNX model.
    input: String,

    /// Quantization bit width: 4 or 8.
    #[arg(long, default_value_t = 8)]
    bits: u8,

    /// Use per-channel quantization (separate scale per output channel).
    #[arg(long, default_value_t = false)]
    per_channel: bool,

    /// Minimum number of elements a tensor must have to be quantized.
    /// Tensors smaller than this are kept in FP32. Defaults to 128.
    #[arg(long, default_value_t = 128)]
    min_elements: usize,

    /// Optional path to save the quantized model. If omitted, only metrics
    /// are reported without writing any file.
    #[arg(long)]
    output: Option<String>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.bits != 4 && args.bits != 8 {
        anyhow::bail!("--bits must be 4 or 8, got {}", args.bits);
    }

    // ------------------------------------------------------------------
    // Load model
    // ------------------------------------------------------------------
    println!("Loading model: {}", args.input);
    let mut model = OnnxModel::load(&args.input)?;
    let info = model.info();

    let file_bytes = std::fs::metadata(&args.input)?.len() as usize;
    println!("  graph: \"{}\"  nodes: {}  inputs: {:?}  outputs: {:?}",
        info.name, info.num_nodes, info.inputs, info.outputs);
    println!("  file size: {}", fmt_bytes(file_bytes));

    // ------------------------------------------------------------------
    // Extract weights
    // ------------------------------------------------------------------
    let weights = model.extract_weights();
    println!("\nFound {} weight tensors ({} will be quantized, {} skipped by min_elements={})\n",
        weights.len(),
        weights.iter().filter(|w| w.data.len() >= args.min_elements).count(),
        weights.iter().filter(|w| w.data.len() < args.min_elements).count(),
        args.min_elements,
    );

    // INT4 values are stored in INT8 containers in ONNX (DequantizeLinear
    // requires INT8 input for opset < 21), so actual file storage = 1 byte
    // per element regardless of bit width.  We track both figures separately
    // so the summary can report the truth.
    let int4_mode = args.bits == 4;

    // Column header
    let onnx_col = if int4_mode { "ONNX bytes" } else { "Quantized" };
    println!("{:<40}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "Tensor name", "Elements", "FP32", onnx_col, "MAE", "Bits");
    println!("{}", "-".repeat(100));

    // ------------------------------------------------------------------
    // Quantize each weight and collect metrics
    // ------------------------------------------------------------------
    let config = QuantConfig {
        bits: args.bits,
        per_channel: args.per_channel,
        min_elements: args.min_elements,
        ..Default::default()
    };
    let quantizer = Quantizer::new(config.clone());

    let mut qdq_data: Vec<QdqWeightInput> = Vec::new();
    let mut total_fp32_bytes: usize = 0;
    // Actual bytes written to the ONNX file (1 byte per element — INT8 container).
    let mut total_onnx_bytes: usize = 0;
    // Theoretical minimum if values were bit-packed (ceil(n/2) for INT4, n for INT8).
    let mut total_packed_bytes: usize = 0;
    let mut total_elements: usize = 0;
    let mut skipped: usize = 0;

    for w in &weights {
        let fp32_bytes = w.data.len() * 4;
        total_fp32_bytes += fp32_bytes;
        total_elements += w.data.len();

        if !config.should_quantize(&w.name, w.data.len()) {
            println!("{:<40}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
                truncate(&w.name, 40),
                fmt_count(w.data.len()),
                fmt_bytes(fp32_bytes),
                "skipped",
                "-",
                "-",
            );
            skipped += 1;
            continue;
        }

        let quantized = quantizer.quantize_tensor(&w.data, w.shape.clone())?;
        // Actual ONNX storage: one INT8 byte per element regardless of bit width.
        let onnx_bytes = w.data.len();
        // Theoretical packed size (ceil(n/2) for INT4, n for INT8).
        let packed_bytes = quantized.size_bytes();
        let mae = quantized.quantization_error(&w.data);
        let bits_used = quantized.bits();

        println!("{:<40}  {:>10}  {:>10}  {:>10}  {:>10.2e}  {:>8}",
            truncate(&w.name, 40),
            fmt_count(w.data.len()),
            fmt_bytes(fp32_bytes),
            fmt_bytes(onnx_bytes),
            mae,
            bits_used,
        );

        let (scales, zero_points) = quantized.get_all_scales_zero_points();
        let is_pc = quantized.is_per_channel();
        qdq_data.push(QdqWeightInput {
            original_name: w.name.clone(),
            quantized_values: quantized.data(),
            scales,
            zero_points,
            bits: bits_used,
            axis: if is_pc { Some(0) } else { None },
        });

        total_onnx_bytes += onnx_bytes;
        total_packed_bytes += packed_bytes;
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    println!("{}", "-".repeat(100));
    println!("\nSummary");
    println!("  Total tensors : {} ({} quantized, {} skipped)",
        weights.len(), qdq_data.len(), skipped);
    println!("  Total elements: {}", fmt_count(total_elements));
    println!("  FP32 weight bytes    : {}", fmt_bytes(total_fp32_bytes));
    println!("  ONNX storage (actual): {}", fmt_bytes(total_onnx_bytes));
    if total_fp32_bytes > 0 {
        let ratio = total_onnx_bytes as f64 / total_fp32_bytes as f64;
        println!("  Compression ratio    : {:.1}x  ({:.1}% of original)",
            1.0 / ratio, ratio * 100.0);
    }
    if int4_mode && total_packed_bytes < total_onnx_bytes {
        let ratio = total_packed_bytes as f64 / total_fp32_bytes as f64;
        println!(
            "  Theoretical INT4 packed: {}  ({:.1}x / {:.1}% of original)",
            fmt_bytes(total_packed_bytes),
            1.0 / ratio,
            ratio * 100.0,
        );
        println!("  (INT4 values stored as INT8 in ONNX — opset 21 required for true 8x)");
    }

    // ------------------------------------------------------------------
    // Save (optional)
    // ------------------------------------------------------------------
    if let Some(ref out_path) = args.output {
        if qdq_data.is_empty() {
            println!("\nNo tensors were quantized; skipping save.");
        } else {
            model.save_quantized(&qdq_data, out_path)?;

            // Validate the saved model
            let reloaded = OnnxModel::load(out_path)?;
            let report = reloaded.validate_connectivity();
            let out_bytes = std::fs::metadata(out_path)?.len() as usize;

            println!("\nSaved to: {out_path}");
            println!("  Output file size: {}", fmt_bytes(out_bytes));
            if report.valid {
                println!("  Connectivity: OK");
            } else {
                println!("  Connectivity: BROKEN ({} broken refs)", report.broken_refs.len());
                for r in &report.broken_refs {
                    println!("    - {r}");
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn fmt_bytes(n: usize) -> String {
    if n >= 1_073_741_824 {
        format!("{:.2} GB", n as f64 / 1_073_741_824.0)
    } else if n >= 1_048_576 {
        format!("{:.2} MB", n as f64 / 1_048_576.0)
    } else if n >= 1_024 {
        format!("{:.1} KB", n as f64 / 1_024.0)
    } else {
        format!("{n} B")
    }
}

fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

/// Build a layer-bits map from `NAME=BITS` strings (mirrors the CLI flag).
#[allow(dead_code)]
fn parse_layer_bits_map(pairs: &[(String, u8)]) -> HashMap<String, u8> {
    pairs.iter().cloned().collect()
}
