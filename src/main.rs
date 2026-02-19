use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::collections::HashMap;

mod cli;

use cli::commands;

fn parse_bits(s: &str) -> Result<u8, String> {
    let bits: u8 = s.parse().map_err(|_| format!("'{}' is not a valid number", s))?;
    if bits == 4 || bits == 8 {
        Ok(bits)
    } else {
        Err(format!("bits must be 4 or 8, got {}", bits))
    }
}

/// Validate calibration method at parse time.
#[cfg(feature = "calibration")]
fn parse_calibration_method(s: &str) -> Result<String, String> {
    match s.to_lowercase().as_str() {
        "minmax" | "percentile" | "entropy" | "mse" => Ok(s.to_string()),
        _ => Err(format!("unknown method '{}'; valid: minmax, percentile, entropy, mse", s)),
    }
}

/// Parse a single `NAME=BITS` layer-bits override (e.g. `conv1.weight=4`).
fn parse_layer_bits(s: &str) -> Result<(String, u8), String> {
    let (name, bits_str) = s.split_once('=')
        .ok_or_else(|| format!("expected NAME=BITS (e.g. conv1.weight=4), got '{}'", s))?;
    if name.is_empty() {
        return Err("layer name must not be empty".into());
    }
    let bits: u8 = bits_str.parse()
        .map_err(|_| format!("'{}' is not a valid bit width", bits_str))?;
    if bits != 4 && bits != 8 {
        return Err(format!("bits must be 4 or 8, got {}", bits));
    }
    Ok((name.to_string(), bits))
}

#[derive(Parser)]
#[command(
    name = "quantize-rs",
    version,
    about = "Neural network quantization toolkit",
    long_about = "Convert ONNX models to INT8/INT4 for faster, smaller deployment"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Quantize {
        #[arg(value_name = "MODEL")]
        input: String,

        #[arg(short, long, default_value = "model_quantized.onnx")]
        output: String,

        #[arg(short, long, default_value = "8", value_parser = parse_bits)]
        bits: u8,

        #[arg(long)]
        per_channel: bool,

        /// Layer names to exclude from quantization (may be specified multiple times).
        #[arg(long = "exclude", value_name = "LAYER")]
        excluded_layers: Vec<String>,

        /// Skip tensors with fewer than this many elements (leave them in FP32).
        #[arg(long, default_value = "0")]
        min_elements: usize,

        /// Per-layer bit-width override (may be specified multiple times).
        /// Format: LAYER=BITS, e.g. --layer-bits conv1.weight=4
        #[arg(long = "layer-bits", value_name = "LAYER=BITS", value_parser = parse_layer_bits)]
        layer_bits: Vec<(String, u8)>,
    },

    Batch {
        #[arg(value_name = "MODELS", required = true)]
        inputs: Vec<String>,

        #[arg(short, long, required = true)]
        output: String,

        #[arg(short, long, default_value = "8", value_parser = parse_bits)]
        bits: u8,

        #[arg(long)]
        per_channel: bool,

        #[arg(long)]
        skip_existing: bool,

        #[arg(long)]
        continue_on_error: bool,

        /// Layer names to exclude from quantization (may be specified multiple times).
        #[arg(long = "exclude", value_name = "LAYER")]
        excluded_layers: Vec<String>,

        /// Skip tensors with fewer than this many elements (leave them in FP32).
        #[arg(long, default_value = "0")]
        min_elements: usize,

        /// Per-layer bit-width override (may be specified multiple times).
        /// Format: LAYER=BITS, e.g. --layer-bits conv1.weight=4
        #[arg(long = "layer-bits", value_name = "LAYER=BITS", value_parser = parse_layer_bits)]
        layer_bits: Vec<(String, u8)>,
    },

    Validate {
        #[arg(value_name = "ORIGINAL")]
        original: String,

        #[arg(value_name = "QUANTIZED")]
        quantized: String,

        #[arg(long)]
        detailed: bool,
    },

    Info {
        #[arg(value_name = "MODEL")]
        input: String,
    },

    Benchmark {
        #[arg(value_name = "ORIGINAL")]
        original: String,

        #[arg(value_name = "QUANTIZED")]
        quantized: String,
    },

    Config {
        #[arg(value_name = "CONFIG")]
        config_file: String,

        #[arg(long)]
        dry_run: bool,
    },

    #[cfg(feature = "calibration")]
    Calibrate {
        #[arg(value_name = "MODEL")]
        input: String,

        #[arg(long, value_name = "DATA")]
        data: String,

        #[arg(short, long, default_value = "model_calibrated.onnx")]
        output: String,

        #[arg(short, long, default_value = "8", value_parser = parse_bits)]
        bits: u8,

        #[arg(long)]
        per_channel: bool,

        #[arg(long, default_value = "percentile", value_parser = parse_calibration_method)]
        method: String,
    },

}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", format!("quantize-rs v{}", env!("CARGO_PKG_VERSION")).bold().cyan());
    println!();

    match cli.command {
        Commands::Quantize {
            input,
            output,
            bits,
            per_channel,
            excluded_layers,
            min_elements,
            layer_bits,
        } => {
            let layer_bits_map: HashMap<String, u8> = layer_bits.into_iter().collect();
            commands::quantize(&input, &output, bits, per_channel, &excluded_layers, min_elements, &layer_bits_map)?;
        }
        Commands::Batch {
            inputs,
            output,
            bits,
            per_channel,
            skip_existing,
            continue_on_error,
            excluded_layers,
            min_elements,
            layer_bits,
        } => {
            let layer_bits_map: HashMap<String, u8> = layer_bits.into_iter().collect();
            commands::batch(&inputs, &output, bits, per_channel, skip_existing, continue_on_error, &excluded_layers, min_elements, &layer_bits_map)?;
        }
        Commands::Validate {
            original,
            quantized,
            detailed,
        } => {
            commands::validate(&original, &quantized, detailed)?;
        }
        Commands::Info { input } => {
            commands::info(&input)?;
        }
        Commands::Benchmark {
            original,
            quantized,
        } => {
            commands::benchmark(&original, &quantized)?;
        }
        Commands::Config { config_file, dry_run } => {
            commands::run_config(&config_file, dry_run)?;
        }

        #[cfg(feature = "calibration")]
        Commands::Calibrate {
            input,
            data,
            output,
            bits,
            per_channel,
            method,
        } => {
            commands::calibrate(&input, &data, &output, bits, per_channel, &method)?;
        }
    }

    Ok(())
}