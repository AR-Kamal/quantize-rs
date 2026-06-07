use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::collections::HashMap;

mod cli;

use cli::commands;

/// Minimal stderr logger for the CLI.  Library code emits warnings through the
/// `log` facade (so library / Python embedders can route or suppress them); the
/// binary installs this so those warnings still reach the terminal as
/// `warning: …` / `error: …` lines, matching the previous behaviour.
struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let label = if record.level() == log::Level::Error {
                "error"
            } else {
                "warning"
            };
            eprintln!("{label}: {}", record.args());
        }
    }
    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

fn parse_bits(s: &str) -> Result<u8, String> {
    let bits: u8 = s
        .parse()
        .map_err(|_| format!("'{}' is not a valid number", s))?;
    if bits == 4 || bits == 8 {
        Ok(bits)
    } else {
        Err(format!("bits must be 4 or 8, got {}", bits))
    }
}

/// Validate the calibration method at parse time.
///
/// Delegates to the library's [`CalibrationMethod`] parser so the CLI accepts
/// exactly what the calibration pipeline does — including the documented
/// `percentile:NN` form (e.g. `percentile:95`) — and the two can never drift
/// apart.  The original string is passed through on success; `commands::calibrate`
/// re-parses it into a `CalibrationMethod`.
#[cfg(feature = "calibration")]
fn parse_calibration_method(s: &str) -> Result<String, String> {
    use quantize_rs::calibration::methods::CalibrationMethod;
    use std::str::FromStr;
    CalibrationMethod::from_str(s)
        .map(|_| s.to_string())
        .map_err(|e| e.to_string())
}

/// Parse the `--format` flag value.  Case-insensitive accepts `human` or `json`.
fn parse_output_format(s: &str) -> Result<String, String> {
    match s.to_lowercase().as_str() {
        "human" | "json" => Ok(s.to_lowercase()),
        _ => Err(format!("format must be 'human' or 'json', got '{}'", s)),
    }
}

/// Parse a single `NAME=BITS` layer-bits override (e.g. `conv1.weight=4`).
fn parse_layer_bits(s: &str) -> Result<(String, u8), String> {
    let (name, bits_str) = s
        .split_once('=')
        .ok_or_else(|| format!("expected NAME=BITS (e.g. conv1.weight=4), got '{}'", s))?;
    if name.is_empty() {
        return Err("layer name must not be empty".into());
    }
    let bits: u8 = bits_str
        .parse()
        .map_err(|_| format!("'{}' is not a valid bit width", bits_str))?;
    if bits != 4 && bits != 8 {
        return Err(format!("bits must be 4 or 8, got {}", bits));
    }
    Ok((name.to_string(), bits))
}

/// Collect `--layer-bits NAME=BITS` pairs into a map, warning on stderr when the
/// same layer is given conflicting bit widths (the last value wins).
fn collect_layer_bits(pairs: Vec<(String, u8)>) -> HashMap<String, u8> {
    let mut map = HashMap::new();
    for (name, bits) in pairs {
        if let Some(prev) = map.insert(name.clone(), bits) {
            if prev != bits {
                log::warn!(
                    "--layer-bits '{}' given more than once with different widths; \
                     using {} (ignoring {})",
                    name,
                    bits,
                    prev
                );
            }
        }
    }
    map
}

#[derive(Parser)]
#[command(
    name = "quantize-rs",
    version,
    about = "Neural network quantization toolkit",
    long_about = "Convert ONNX models to INT8/INT4 weight-only QDQ to shrink model files 4-8x on disk. \
                  Weights are quantized and activations stay FP32, so this reduces download/storage \
                  size rather than guaranteeing faster inference."
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

        /// Per-channel quantization (separate scale/zero-point per output
        /// channel).  Always uses axis 0 (output-channel dim) — Conv and
        /// MatMul weights are the intended target.  For Transformer linear
        /// layers that expect axis=1 quantization, omit `--per-channel` or
        /// implement axis=1 in a library call (not currently supported via
        /// the CLI).
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

        /// Store INT4 weights as native ONNX `DataType::Int4` (opset 21)
        /// instead of widening to INT8.  Gives true 8× compression but
        /// requires an ONNX runtime with opset 21 support.  Has no effect
        /// on INT8-only models.
        #[arg(long = "native-int4")]
        native_int4: bool,

        /// Use symmetric quantization (zero_point == 0).  Required by most
        /// ONNX Runtime / TensorRT INT8 matmul kernels for per-channel
        /// weight quantization.
        #[arg(long = "symmetric")]
        symmetric: bool,
    },

    Batch {
        #[arg(value_name = "MODELS", required = true)]
        inputs: Vec<String>,

        #[arg(short, long, required = true)]
        output: String,

        #[arg(short, long, default_value = "8", value_parser = parse_bits)]
        bits: u8,

        /// Per-channel quantization (separate scale/zero-point per output
        /// channel).  Always uses axis 0 (output-channel dim) — Conv and
        /// MatMul weights are the intended target.  For Transformer linear
        /// layers that expect axis=1 quantization, omit `--per-channel` or
        /// implement axis=1 in a library call (not currently supported via
        /// the CLI).
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

        /// Store INT4 weights as native ONNX `DataType::Int4` (opset 21).
        #[arg(long = "native-int4")]
        native_int4: bool,

        /// Use symmetric quantization (zero_point == 0).
        #[arg(long = "symmetric")]
        symmetric: bool,

        /// Number of models to quantize in parallel.  `1` (default) preserves
        /// the current serial behaviour; higher values run multiple files
        /// concurrently at the cost of interleaved progress output.
        #[arg(long, default_value = "1")]
        jobs: usize,
    },

    Validate {
        #[arg(value_name = "ORIGINAL")]
        original: String,

        #[arg(value_name = "QUANTIZED")]
        quantized: String,

        #[arg(long)]
        detailed: bool,

        /// Output format: `human` (default) or `json`.
        #[arg(long, default_value = "human", value_parser = parse_output_format)]
        format: String,
    },

    Info {
        #[arg(value_name = "MODEL")]
        input: String,

        /// Output format: `human` (default) or `json`.
        #[arg(long, default_value = "human", value_parser = parse_output_format)]
        format: String,
    },

    Benchmark {
        #[arg(value_name = "ORIGINAL")]
        original: String,

        #[arg(value_name = "QUANTIZED")]
        quantized: String,

        /// Output format: `human` (default) or `json`.
        #[arg(long, default_value = "human", value_parser = parse_output_format)]
        format: String,
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

        /// Per-channel quantization (separate scale/zero-point per output
        /// channel).  Always uses axis 0 (output-channel dim) — Conv and
        /// MatMul weights are the intended target.  For Transformer linear
        /// layers that expect axis=1 quantization, omit `--per-channel` or
        /// implement axis=1 in a library call (not currently supported via
        /// the CLI).
        #[arg(long)]
        per_channel: bool,

        #[arg(long, default_value = "percentile", value_parser = parse_calibration_method)]
        method: String,

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

        /// Store INT4 weights as native ONNX `DataType::Int4` (opset 21).
        #[arg(long = "native-int4")]
        native_int4: bool,

        /// Use symmetric quantization (zero_point == 0).
        #[arg(long = "symmetric")]
        symmetric: bool,
    },
}

fn main() -> Result<()> {
    // Route library `log` warnings to stderr (Err only if a logger is already
    // installed, which won't happen for the CLI binary).
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Warn));

    let cli = Cli::parse();

    // Suppress the startup banner when a subcommand emits JSON so the caller
    // gets only parseable output on stdout.
    let json_mode = matches!(
        &cli.command,
        Commands::Validate { format, .. }
        | Commands::Info { format, .. }
        | Commands::Benchmark { format, .. }
        if format == "json"
    );

    if !json_mode {
        println!(
            "{}",
            format!("quantize-rs v{}", env!("CARGO_PKG_VERSION"))
                .bold()
                .cyan()
        );
        println!();
    }

    match cli.command {
        Commands::Quantize {
            input,
            output,
            bits,
            per_channel,
            excluded_layers,
            min_elements,
            layer_bits,
            native_int4,
            symmetric,
        } => {
            let layer_bits_map = collect_layer_bits(layer_bits);
            commands::quantize(
                &input,
                &output,
                bits,
                per_channel,
                &excluded_layers,
                min_elements,
                &layer_bits_map,
                native_int4,
                symmetric,
            )?;
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
            native_int4,
            symmetric,
            jobs,
        } => {
            let layer_bits_map = collect_layer_bits(layer_bits);
            commands::batch(
                &inputs,
                &output,
                bits,
                per_channel,
                skip_existing,
                continue_on_error,
                &excluded_layers,
                min_elements,
                &layer_bits_map,
                native_int4,
                symmetric,
                jobs,
            )?;
        }
        Commands::Validate {
            original,
            quantized,
            detailed,
            format,
        } => {
            commands::validate(&original, &quantized, detailed, &format)?;
        }
        Commands::Info { input, format } => {
            commands::info(&input, &format)?;
        }
        Commands::Benchmark {
            original,
            quantized,
            format,
        } => {
            commands::benchmark(&original, &quantized, &format)?;
        }
        Commands::Config {
            config_file,
            dry_run,
        } => {
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
            excluded_layers,
            min_elements,
            layer_bits,
            native_int4,
            symmetric,
        } => {
            let layer_bits_map = collect_layer_bits(layer_bits);
            commands::calibrate(
                &input,
                &data,
                &output,
                bits,
                per_channel,
                &method,
                &excluded_layers,
                min_elements,
                &layer_bits_map,
                native_int4,
                symmetric,
            )?;
        }
    }

    Ok(())
}

#[cfg(all(test, feature = "calibration"))]
mod tests {
    use super::*;

    #[test]
    fn parse_calibration_method_accepts_percentile_nn() {
        // Regression: the CLI validator used to reject the documented
        // `percentile:NN` form even though the library parser (and the Python
        // bindings) accept it.  See README.md `calibrate --method`.
        assert_eq!(
            parse_calibration_method("percentile:95").unwrap(),
            "percentile:95"
        );
        assert_eq!(
            parse_calibration_method("percentile:99.9").unwrap(),
            "percentile:99.9"
        );
    }

    #[test]
    fn parse_calibration_method_accepts_bare_keywords() {
        for m in ["minmax", "percentile", "entropy", "mse"] {
            assert_eq!(parse_calibration_method(m).unwrap(), m);
        }
    }

    #[test]
    fn parse_calibration_method_rejects_unknown_and_out_of_range() {
        // Unknown keyword.
        assert!(parse_calibration_method("bogus").is_err());
        // Percentile outside [0, 100] is rejected by the library parser, so
        // the CLI rejects it too (no longer silently accepted or mislabeled).
        assert!(parse_calibration_method("percentile:200").is_err());
        assert!(parse_calibration_method("percentile:-1").is_err());
    }

    #[test]
    fn collect_layer_bits_last_value_wins_on_conflict() {
        let map = collect_layer_bits(vec![
            ("a".to_string(), 4),
            ("b".to_string(), 8),
            ("a".to_string(), 8),
        ]);
        assert_eq!(map.get("a"), Some(&8), "last value should win");
        assert_eq!(map.get("b"), Some(&8));
        assert_eq!(map.len(), 2);
    }
}
