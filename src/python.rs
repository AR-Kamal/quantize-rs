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
///     per_channel: Enable per-channel quantization (always axis 0, the
///         output-channel dim — Conv/MatMul-friendly; Transformer-style
///         linear layers expecting axis=1 are not yet supported)
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
    // other Python threads (and asyncio tasks) can run.  Quantizer internally
    // uses rayon for parallel weight quantization.
    py.allow_threads(|| -> PyResult<()> {
        let mut model = OnnxModel::load(input_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        // Extract once and reuse: the empty-check, and the quantizer below, both
        // need the weights — decoding them twice doubles the f32 conversion cost.
        let weights = model.extract_weights();
        if weights.is_empty() {
            let external = model.count_external_data_initializers();
            let non_fp32 = model.count_non_fp32_weight_initializers();
            return Err(PyRuntimeError::new_err(if external > 0 {
                format!(
                    "no inline FP32 weight tensors found: {} initializer(s) store their data in \
                     an external file (ONNX external-data format); quantize-rs reads only inline \
                     tensors — re-save with weights embedded (onnx.load(path, \
                     load_external_data=True) then onnx.save without external data) and retry",
                    external
                )
            } else if non_fp32 > 0 {
                format!(
                    "no FP32 weight tensors found, but the model has {} non-FP32 \
                     weight-shaped initializer(s) (FP16/BF16/Double); quantize-rs \
                     supports only FP32 input — convert the model to FP32 first",
                    non_fp32
                )
            } else {
                "no weight tensors found — model may be empty or already quantized".to_string()
            }));
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
            .quantize_weights(&weights)
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

/// Activation-based calibration quantization.
///
/// Args:
///     input_path: Path to input ONNX model
///     output_path: Path to save quantized model
///     calibration_data: Path to .npy calibration data, or None for random samples
///     bits: Bit width (4 or 8)
///     per_channel: Enable per-channel quantization (always axis 0 — see
///         `quantize` for the axis caveat)
///     method: Calibration method ("minmax", "percentile", "percentile:NN",
///         "entropy", "mse")
///     num_samples: Number of random samples if calibration_data is None
///     sample_shape: Shape of random samples (e.g., [3, 224, 224])
///     native_int4: If True, store INT4 weights as native ONNX DataType.Int4
///         (opset 21) — 2× smaller on disk but requires an ORT build with
///         opset 21 support.  Has no effect on INT8-only models.  Default False.
///     symmetric: If True, force zero_point == 0 (symmetric quantization).
///         Required by most ONNX Runtime / TensorRT INT8 matmul kernels for
///         per-channel weight quantization.  Default False.
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
) -> PyResult<()> {
    if bits != 4 && bits != 8 {
        return Err(PyValueError::new_err(format!(
            "bits must be 4 or 8, got {}",
            bits
        )));
    }

    // Parse the method on the GIL-holding thread (it's a string parse — trivial).
    let calib_method: CalibrationMethod = method
        .parse()
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

    // Release the GIL for the duration of the I/O- and CPU-heavy work
    // (model load, tract inference, parallel quantize, save).  Other Python
    // threads (and asyncio tasks) can run concurrently while calibration
    // crunches numbers in Rust.
    py.allow_threads(|| -> PyResult<()> {
        let model = OnnxModel::load(input_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        let dataset = if let Some(path) = calibration_data {
            CalibrationDataset::from_numpy(path).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to load calibration data: {}", e))
            })?
        } else {
            let shape = if let Some(s) = sample_shape {
                s
            } else {
                model
                    .input_shapes()
                    .into_iter()
                    .next()
                    .and_then(|dims| {
                        // Strip the batch slot (first dim) BEFORE filtering
                        // out symbolic / non-positive dims.  See the matching
                        // comment in src/cli/commands.rs::calibrate.
                        let sample_dims: &[i64] = if dims.len() >= 2 { &dims[1..] } else { &dims };
                        let shape: Vec<usize> = sample_dims
                            .iter()
                            .filter_map(|&d| if d > 0 { Some(d as usize) } else { None })
                            .collect();
                        if shape.is_empty() {
                            None
                        } else {
                            Some(shape)
                        }
                    })
                    .unwrap_or_else(|| vec![3, 224, 224])
            };

            CalibrationDataset::random(shape, num_samples, (0.0, 1.0)).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create random dataset: {}", e))
            })?
        };

        let mut estimator = ActivationEstimator::new(model, input_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create estimator: {}", e)))?;

        estimator
            .calibrate_quiet(&dataset)
            .map_err(|e| PyRuntimeError::new_err(format!("Calibration failed: {}", e)))?;

        let activation_stats: std::collections::HashMap<String, _> = estimator
            .get_layer_stats()
            .into_iter()
            .map(|(k, v)| (k, v.clone()))
            .collect();
        let mut model = estimator.into_model();

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
