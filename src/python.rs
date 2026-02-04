// src/python.rs
//! Python bindings for quantize-rs using PyO3.
//!
//! Provides a simple API:
//!   - quantize() — basic weight-based quantization
//!   - quantize_with_calibration() — activation-based calibration
//!   - model_info() — get model metadata

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::onnx_utils::OnnxModel;
use crate::quantization::{QuantConfig, Quantizer};
use crate::calibration::{CalibrationDataset, ActivationEstimator, methods::CalibrationMethod};

// ===========================================================================
// Python-exposed types
// ===========================================================================

/// Model metadata returned by model_info()
#[pyclass]
#[derive(Clone)]
struct ModelInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    version: i64,
    #[pyo3(get)]
    num_nodes: usize,
    #[pyo3(get)]
    inputs: Vec<String>,
    #[pyo3(get)]
    outputs: Vec<String>,
}

// ===========================================================================
// Core functions
// ===========================================================================

/// Basic weight-based quantization.
///
/// Args:
///     input_path: Path to input ONNX model
///     output_path: Path to save quantized model
///     bits: Bit width (4 or 8)
///     per_channel: Enable per-channel quantization
///
/// Example:
///     >>> import quantize_rs
///     >>> quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)
#[pyfunction]
#[pyo3(signature = (input_path, output_path, bits=8, per_channel=false))]
fn quantize(
    input_path: &str,
    output_path: &str,
    bits: u8,
    per_channel: bool,
) -> PyResult<()> {
    // Load model
    let mut model = OnnxModel::load(input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

    // Extract weights
    let weights = model.extract_weights();
    if weights.is_empty() {
        return Err(PyRuntimeError::new_err("No weights found to quantize"));
    }

    // Quantize
    let config = QuantConfig {
        bits,
        per_channel,
        calibration_method: None,
    };
    let quantizer = Quantizer::new(config);

    let mut quantized_data = Vec::new();
    for weight in &weights {
        let quantized = quantizer.quantize_tensor(&weight.data, weight.shape.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Quantization failed: {}", e)))?;

        let (scale, zero_point) = quantized.get_scale_zero_point();
        let bits_used = quantized.bits();

        quantized_data.push((
            weight.name.clone(),
            quantized.data(),
            scale,
            zero_point,
            bits_used,
        ));
    }

    // Save
    model.save_quantized(&quantized_data, output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to save model: {}", e)))?;

    Ok(())
}

/// Activation-based calibration quantization.
///
/// Args:
///     input_path: Path to input ONNX model
///     output_path: Path to save quantized model
///     calibration_data: Path to .npy calibration data, or None for random samples
///     bits: Bit width (4 or 8)
///     per_channel: Enable per-channel quantization
///     method: Calibration method ("minmax", "percentile", "entropy", "mse")
///     num_samples: Number of random samples if calibration_data is None
///     sample_shape: Shape of random samples (e.g., [3, 224, 224])
///
/// Example:
///     >>> import quantize_rs
///     >>> quantize_rs.quantize_with_calibration(
///     ...     "resnet18.onnx",
///     ...     "resnet18_int8.onnx",
///     ...     calibration_data="samples.npy",
///     ...     method="minmax"
///     ... )
#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_path,
    calibration_data=None,
    bits=8,
    per_channel=false,
    method="minmax",
    num_samples=100,
    sample_shape=None
))]
fn quantize_with_calibration(
    input_path: &str,
    output_path: &str,
    calibration_data: Option<&str>,
    bits: u8,
    per_channel: bool,
    method: &str,
    num_samples: usize,
    sample_shape: Option<Vec<usize>>,
) -> PyResult<()> {
    // Parse calibration method
    let calib_method = match method.to_lowercase().as_str() {
        "minmax" => CalibrationMethod::MinMax,
        "percentile" => CalibrationMethod::Percentile(99.9),
        "entropy" => CalibrationMethod::Entropy,
        "mse" => CalibrationMethod::MSE,
        _ => return Err(PyRuntimeError::new_err(format!("Unknown method: {}", method))),
    };

    // Load calibration data
    let dataset = if let Some(path) = calibration_data {
        CalibrationDataset::from_numpy(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load calibration data: {}", e)))?
    } else {
        // Use provided shape or auto-detect from model
        let shape = if let Some(s) = sample_shape {
            s
        } else {
            // Try to detect from model
            let model = OnnxModel::load(input_path)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;
            let info = model.info();
            
            // Parse shape from first input (best-effort)
            if !info.inputs.is_empty() {
                parse_shape_from_input(&info.inputs[0]).unwrap_or_else(|| vec![3, 224, 224])
            } else {
                vec![3, 224, 224]
            }
        };
        
        CalibrationDataset::random(shape, num_samples, (0.0, 1.0))
    };

    // Load model for calibration
    let model = OnnxModel::load(input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

    // Run calibration
    let mut estimator = ActivationEstimator::new(model, input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create estimator: {}", e)))?;

    estimator.calibrate(&dataset)
        .map_err(|e| PyRuntimeError::new_err(format!("Calibration failed: {}", e)))?;

    let activation_stats = estimator.into_layer_stats();

    // Reload model for quantization
    let mut model = OnnxModel::load(input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reload model: {}", e)))?;

    let weights = model.extract_weights();

    // Quantize with calibration
    let config = QuantConfig {
        bits,
        per_channel,
        calibration_method: Some(calib_method),
    };
    let quantizer = Quantizer::with_calibration(config, activation_stats);

    let mut quantized_data = Vec::new();
    for weight in &weights {
        let quantized = quantizer.quantize_tensor_with_name(
            &weight.name,
            &weight.data,
            weight.shape.clone(),
        ).map_err(|e| PyRuntimeError::new_err(format!("Quantization failed: {}", e)))?;

        let (scale, zero_point) = quantized.get_scale_zero_point();
        let bits_used = quantized.bits();

        quantized_data.push((
            weight.name.clone(),
            quantized.data(),
            scale,
            zero_point,
            bits_used,
        ));
    }

    // Save
    model.save_quantized(&quantized_data, output_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to save model: {}", e)))?;

    Ok(())
}

/// Get model information.
///
/// Args:
///     input_path: Path to ONNX model
///
/// Returns:
///     ModelInfo with name, version, num_nodes, inputs, outputs
///
/// Example:
///     >>> import quantize_rs
///     >>> info = quantize_rs.model_info("model.onnx")
///     >>> print(f"{info.name}: {info.num_nodes} nodes")
#[pyfunction]
fn model_info(input_path: &str) -> PyResult<ModelInfo> {
    let model = OnnxModel::load(input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

    let info = model.info();

    Ok(ModelInfo {
        name: info.name,
        version: info.version,
        num_nodes: info.num_nodes,
        inputs: info.inputs,
        outputs: info.outputs,
    })
}

// ===========================================================================
// Helper functions
// ===========================================================================

/// Parse input shape from model info string (e.g., "Input3: float32[1,1,28,28]")
fn parse_shape_from_input(input_str: &str) -> Option<Vec<usize>> {
    let shape_part = input_str.split('[').nth(1)?;
    let shape_str = shape_part.split(']').nth(0)?;
    
    let dims: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    // Skip batch dimension (first dim), return remaining
    if dims.len() >= 2 {
        Some(dims[1..].to_vec())
    } else {
        None
    }
}

// ===========================================================================
// Python module definition
// ===========================================================================

/// Neural network quantization toolkit
#[pymodule]
fn quantize_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_with_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(model_info, m)?)?;
    m.add_class::<ModelInfo>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}