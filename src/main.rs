//! Quantize-rs CLI
//!
//! Command-line interface for neural network quantization.

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;

mod cli;
mod config;

use cli::commands;

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
    /// Quantize an ONNX model
    Quantize {
        /// Input ONNX model path
        #[arg(value_name = "MODEL")]
        input: String,

        /// Output path for quantized model
        #[arg(short, long, default_value = "model_quantized.onnx")]
        output: String,

        /// Quantization bits (8 or 4)
        #[arg(short, long, default_value = "8")]
        bits: u8,

        /// Use per-channel quantization
        #[arg(long)]
        per_channel: bool,
    },

    /// Batch quantize multiple models
    Batch {
        /// Input model paths (supports wildcards)
        #[arg(value_name = "MODELS", required = true)]
        inputs: Vec<String>,

        /// Output directory
        #[arg(short, long, required = true)]
        output: String,

        /// Quantization bits (8 or 4)
        #[arg(short, long, default_value = "8")]
        bits: u8,

        /// Use per-channel quantization
        #[arg(long)]
        per_channel: bool,

        /// Skip models that already exist in output directory
        #[arg(long)]
        skip_existing: bool,

        /// Continue processing even if some models fail
        #[arg(long)]
        continue_on_error: bool,
    },

    /// Validate quantized model against original
    Validate {
        /// Original model path
        #[arg(value_name = "ORIGINAL")]
        original: String,

        /// Quantized model path
        #[arg(value_name = "QUANTIZED")]
        quantized: String,

        /// Show detailed per-layer analysis
        #[arg(long)]
        detailed: bool,
    },

    /// Show model information
    Info {
        /// ONNX model path
        #[arg(value_name = "MODEL")]
        input: String,
    },

    /// Benchmark quantized vs original
    Benchmark {
        /// Original model
        #[arg(value_name = "ORIGINAL")]
        original: String,

        /// Quantized model
        #[arg(value_name = "QUANTIZED")]
        quantized: String,
    },

    Config {
        /// Path to config file (YAML or TOML)
        #[arg(value_name = "CONFIG")]
        config_file: String,

        /// Dry run (show what would be done without doing it)
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "quantize-rs v0.1.0".bold().cyan());
    println!();

    match cli.command {
        Commands::Quantize {
            input,
            output,
            bits,
            per_channel,
        } => {
            // Validate bits
            if bits != 4 && bits != 8 {
                eprintln!("Error: bits must be 4 or 8");
                std::process::exit(1);
            }
            
            commands::quantize(&input, &output, bits, per_channel)?;
        }
        Commands::Batch {
            inputs,
            output,
            bits,
            per_channel,
            skip_existing,
            continue_on_error,
        } => {
            commands::batch(&inputs, &output, bits, per_channel, skip_existing, continue_on_error)?;
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
    }

    Ok(())
}