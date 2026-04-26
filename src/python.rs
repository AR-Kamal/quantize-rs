// src/python.rs
//! Python bindings for quantize-rs using PyO3.
//!
//! Provides a simple API:
//!   - quantize() — basic weight-based quantization
//!   - quantize_with_calibration() — activation-based calibration
//!   - model_info() — get model metadata

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(feature = "calibration")]
use crate::calibration::{methods::CalibrationMethod, ActivationEstimator, CalibrationDataset};
use crate::onnx_utils::graph_builder::QdqWeightInput;
use crate::onnx_utils::{OnnxModel, SaveOptions};
use crate::quantization::{QuantConfig, Quantizer};

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
///     excluded_layers: Layer names to skip (exact match on initializer name)
///     min_elements: Skip tensors with fewer elements than this (0 = no minimum)
///     layer_bits: Per-layer bit-width overrides, e.g. {"conv1.weight": 4}
///     native_int4: If True, store INT4 weights as native ONNX DataType.Int4
///         (opset 21) — 2× smaller on disk but requires an ORT build with
///         opset 21 support.  Has no effect on INT8-only models.  Default False.
///
/// Example:
///     >>> import quantize_rs
///     >>> quantize_rs.quantize("model.onnx", "model_int8.onnx", bits=8)
///     >>> quantize_rs.quantize("model.onnx", "out.onnx", bits=4, native_int4=True)
#[pyfunction]
#[pyo3(signature = (input_path, output_path, bits=8, per_channel=false, excluded_layers=None, min_elements=0, layer_bits=None, native_int4=false, symmetric=false))]
#[allow(clippy::too_many_arguments)]
fn quantize(
    input_path: &str,
    output_path: &str,
    bits: u8,
    per_channel: bool,
    excluded_layers: Option<Vec<String>>,
    min_elements: usize,
    layer_bits: Option<std::collections::HashMap<String, u8>>,
    native_int4: bool,
    symmetric: bool,
) -> PyResult<()> {
    if bits != 4 && bits != 8 {
        return Err(PyValueError::new_err(format!(
            "bits must be 4 or 8, got {}",
            bits
        )));
    }

    // Load model
    let mut model = OnnxModel::load(input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

    if model.extract_weights().is_empty() {
        return Err(PyRuntimeError::new_err("No weights found to quantize"));
    }

    let config = QuantConfig {
        bits,
        per_channel,
        symmetric,
        calibration_method: None,
        excluded_layers: excluded_layers.unwrap_or_default(),
        min_elements,
        layer_bits: layer_bits.unwrap_or_default(),
    };

    let outputs = Quantizer::new(config)
        .quantize_model(&model)
        .map_err(|e| PyRuntimeError::new_err(format!("Quantization failed: {}", e)))?;

    let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

    // Save
    let save_options = SaveOptions::default().with_native_int4(native_int4);
    model
        .save_quantized_with_options(&quantized_data, output_path, save_options)
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
#[cfg(feature = "calibration")]
#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_path,
    calibration_data=None,
    bits=8,
    per_channel=false,
    method="minmax",
    num_samples=100,
    sample_shape=None,
    native_int4=false,
    symmetric=false,
))]
#[allow(clippy::too_many_arguments)]
fn quantize_with_calibration(
    input_path: &str,
    output_path: &str,
    calibration_data: Option<&str>,
    bits: u8,
    per_channel: bool,
    method: &str,
    num_samples: usize,
    sample_shape: Option<Vec<usize>>,
    native_int4: bool,
    symmetric: bool,
) -> PyResult<()> {
    if bits != 4 && bits != 8 {
        return Err(PyValueError::new_err(format!(
            "bits must be 4 or 8, got {}",
            bits
        )));
    }

    // Parse calibration method
    let calib_method: CalibrationMethod = method
        .parse()
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

    // Load model (used for both shape detection and calibration)
    let model = OnnxModel::load(input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

    // Load calibration data
    let dataset = if let Some(path) = calibration_data {
        CalibrationDataset::from_numpy(path).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to load calibration data: {}", e))
        })?
    } else {
        // Use provided shape or auto-detect from model
        let shape = if let Some(s) = sample_shape {
            s
        } else {
            model
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
                .unwrap_or_else(|| vec![3, 224, 224])
        };

        CalibrationDataset::random(shape, num_samples, (0.0, 1.0)).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create random dataset: {}", e))
        })?
    };

    // Run calibration
    let mut estimator = ActivationEstimator::new(model, input_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create estimator: {}", e)))?;

    estimator
        .calibrate(&dataset)
        .map_err(|e| PyRuntimeError::new_err(format!("Calibration failed: {}", e)))?;

    // Collect stats (borrowed) then recover the model without reloading
    let activation_stats: std::collections::HashMap<String, _> = estimator
        .get_layer_stats()
        .into_iter()
        .map(|(k, v)| (k, v.clone()))
        .collect();
    let mut model = estimator.into_model();

    // Quantize with calibration — runs filter + parallel + layer_bits fallback
    // via the shared library helper (previously this path did none of that).
    let config = QuantConfig {
        bits,
        per_channel,
        symmetric,
        calibration_method: Some(calib_method),
        ..Default::default()
    };

    let outputs = Quantizer::with_calibration(config, activation_stats)
        .quantize_model(&model)
        .map_err(|e| PyRuntimeError::new_err(format!("Quantization failed: {}", e)))?;
    let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

    // Save
    let save_options = SaveOptions::default().with_native_int4(native_int4);
    model
        .save_quantized_with_options(&quantized_data, output_path, save_options)
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
// Python module definition
// ===========================================================================

/// Neural network quantization toolkit
#[pymodule]
fn quantize_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    #[cfg(feature = "calibration")]
    m.add_function(wrap_pyfunction!(quantize_with_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(model_info, m)?)?;
    m.add_class::<ModelInfo>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
