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
use crate::calibration::{methods::CalibrationMethod, CalibrationDataset};
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
    opset_version: i64,
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
///     per_channel: Select output-channel axes from graph usage: Conv axis 0,
///         MatMul last weight dimension, Gemm axis 1 (axis 0 for transB=1)
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
    py: Python<'_>,
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

    // Release the GIL for the duration of the I/O- and CPU-heavy work so
    // other Python threads can run.  Quantizer internally
    // uses rayon for parallel weight quantization.
    py.allow_threads(|| -> PyResult<()> {
        let mut model = OnnxModel::load(input_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        let config = QuantConfig {
            bits,
            per_channel,
            symmetric,
            calibration_method: None,
            excluded_layers: excluded_layers.unwrap_or_default(),
            min_elements,
            layer_bits: layer_bits.unwrap_or_default(),
        };

        let weights = model.select_weights(&config)
            .map_err(|e| PyRuntimeError::new_err(format!("Weight selection failed: {e}")))?;
        if weights.is_empty() {
            return Err(PyRuntimeError::new_err(
                "no eligible direct Conv/MatMul/Gemm weights remain after excluded_layers / min_elements; only inline FP32 matrix/tensor weights are supported"
            ));
        }

        let outputs = Quantizer::new(config)
            .quantize_selected_weights(&weights)
            .map_err(|e| PyRuntimeError::new_err(format!("Quantization failed: {}", e)))?;

        if outputs.is_empty() {
            return Err(PyRuntimeError::new_err(
                "all weight tensors were filtered out by excluded_layers / min_elements / \
                 layer_bits; nothing to quantize",
            ));
        }

        let quantized_data: Vec<QdqWeightInput> = outputs.into_iter().map(|o| o.qdq).collect();

        let save_options = SaveOptions::default().with_native_int4(native_int4);
        model
            .save_quantized_with_options(&quantized_data, output_path, save_options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save model: {}", e)))?;

        Ok(())
    })
}

/// Static INT8 activation quantization for selected Conv layers.
///
/// Requires representative calibration_data (.npy), opset >= 13, and one fixed
/// FP32 NCHW input with batch 1. Weight ranges remain weight-derived; activation
/// ranges produce real QDQ pairs. Other operators stay floating point.
/// excluded_layers and min_elements select Conv weights. layer_bits accepts
/// only 8 in this path. INT4/native_int4 and random calibration are rejected.
/// num_samples is retained for signature compatibility but unused with data.
/// sample_shape, when supplied, must match the supplied dataset.
#[cfg(feature = "calibration")]
#[pyfunction]
#[pyo3(signature = (
    input_path, output_path, calibration_data=None, bits=8, per_channel=false,
    method="minmax", num_samples=100, sample_shape=None, native_int4=false,
    symmetric=false, excluded_layers=None, min_elements=0, layer_bits=None,
))]
#[allow(clippy::too_many_arguments)]
fn quantize_with_calibration(
    py: Python<'_>,
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
    excluded_layers: Option<Vec<String>>,
    min_elements: usize,
    layer_bits: Option<std::collections::HashMap<String, u8>>,
) -> PyResult<()> {
    let _ = num_samples;
    let layer_bits = layer_bits.unwrap_or_default();
    if native_int4 || bits != 8 || layer_bits.values().any(|b| *b != 8) {
        return Err(PyValueError::new_err("static activation quantization supports INT8 only; use quantize for INT4 or mixed precision"));
    }
    let data_path = calibration_data.ok_or_else(|| PyValueError::new_err("representative calibration_data (.npy) is required; random calibration has been removed"))?;
    let method: CalibrationMethod = method
        .parse()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    py.allow_threads(|| -> PyResult<()> {
        let dataset = CalibrationDataset::from_numpy(data_path)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        if sample_shape
            .as_ref()
            .is_some_and(|shape| shape != &dataset.shape)
        {
            return Err(PyValueError::new_err(
                "sample_shape does not match calibration_data",
            ));
        }
        let config = QuantConfig {
            bits,
            per_channel,
            symmetric,
            calibration_method: Some(method),
            excluded_layers: excluded_layers.unwrap_or_default(),
            min_elements,
            layer_bits,
        };
        crate::quantize_static(input_path, output_path, &dataset, config)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    })
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
        opset_version: info.opset_version,
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
    // Bridge Rust `log` records (quantize-rs warnings) into Python's `logging`,
    // so callers route or silence them through the standard logging API instead
    // of having stderr written from under them.  Ignore "already initialised".
    let _ = pyo3_log::try_init();

    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    #[cfg(feature = "calibration")]
    m.add_function(wrap_pyfunction!(quantize_with_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(model_info, m)?)?;
    m.add_class::<ModelInfo>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
