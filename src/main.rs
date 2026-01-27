//! Quantize-rs CLI
//!
//! Command-line interface for neural network quantization.

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;

mod cli;

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
            commands::quantize(&input, &output, bits, per_channel)?;
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
    }

    Ok(())
}
