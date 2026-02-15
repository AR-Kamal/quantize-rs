use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;

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

        #[arg(long, default_value = "percentile")]
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
        } => {
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