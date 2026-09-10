//! Static INT8 activation quantization for selected Conv layers.

use super::{ActivationEstimator, CalibrationDataset};
use crate::errors::Result;
use crate::{OnnxModel, QuantConfig, QuantizeError, Quantizer};

/// Quantize selected Conv weights and their input/output activation edges.
///
/// Requires an inline FP32 model at opset >= 13, one fixed NCHW input with
/// batch 1, and representative finite FP32 calibration samples. Only INT8 is
/// supported. Other operators stay floating point. Exclusions and minimum
/// element counts apply to Conv weights; a shared activation can still carry
/// the quantized output of a selected upstream Conv.
///
/// Weights use their own ranges. Activation parameters are computed from
/// tensor-name-keyed statistics and saved as QuantizeLinear/DequantizeLinear
/// pairs. Input and output tensor names are preserved. Speed depends on the
/// deployment runtime and is not guaranteed.
pub fn quantize_static(
    input_path: &str,
    output_path: &str,
    dataset: &CalibrationDataset,
    config: QuantConfig,
) -> Result<()> {
    let model = OnnxModel::load(input_path)?;
    let shape = model.calibration_sample_shape()?;
    let elements = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| QuantizeError::Calibration {
            reason: "sample shape overflows usize".into(),
        })?;
    if dataset.shape != shape
        || dataset.is_empty()
        || dataset
            .samples
            .iter()
            .any(|s| s.len() != elements || s.iter().any(|v| !v.is_finite()))
    {
        return Err(QuantizeError::Calibration {
            reason: format!(
                "expected nonempty finite FP32 calibration samples with shape {shape:?}"
            ),
        });
    }
    // Validate method values even when callers construct the enum directly.
    let method = config.calibration_method.unwrap_or_default();
    let _: super::methods::CalibrationMethod = method.to_string().parse()?;
    let (weights, edges) = model.calibration_plan(&config)?;
    let outputs = Quantizer::new(config).quantize_weights(&weights)?;
    if outputs
        .iter()
        .any(|o| o.qdq.scales.iter().any(|s| !s.is_finite() || *s <= 0.0))
    {
        return Err(QuantizeError::Calibration {
            reason: "weight quantization produced an invalid scale".into(),
        });
    }
    let weights: Vec<_> = outputs.into_iter().map(|o| o.qdq).collect();
    let mut estimator = ActivationEstimator::for_tensors(model, input_path, &edges)?;
    estimator.calibrate_quiet(dataset)?;
    let stats = estimator
        .get_layer_stats()
        .into_iter()
        .map(|(k, v)| (k, v.clone()))
        .collect();
    let mut model = estimator.into_model();
    model.save_calibrated(&weights, &stats, method, output_path)
}

/// Experimental activation QDQ for fixed-input MatMul/Gemm feed-forward graphs.
///
/// Requires one fixed FP32 `[1, K]` input and representative dataset samples of
/// shape `[K]` (`[samples, K]` in an NPY file). Only MatMul/Gemm with direct rank-2
/// FP32 weights, Add, Relu and Identity are accepted. Gemm supports both transpose
/// flags but requires alpha=beta=1 and an optional constant FP32 `[N]` bias.
///
/// Configure symmetric per-channel INT8 weights. Activations use scalar asymmetric
/// INT8 ranges from real tract inference; weights retain independent ranges.
/// Exclusions/minimum size select matrix weights; shared activation edges can
/// still affect excluded consumers. Output names and unselected tensors survive.
/// Token inputs, dynamic/batched inputs, indirect weights and Transformer
/// calibration are outside this prototype. Integer fusion depends on the runtime.
pub fn quantize_static_matrix(
    input_path: &str,
    output_path: &str,
    dataset: &CalibrationDataset,
    config: QuantConfig,
) -> Result<()> {
    let model = OnnxModel::load(input_path)?;
    let shape = model.matrix_calibration_sample_shape()?;
    if dataset.shape != shape
        || dataset.is_empty()
        || dataset
            .samples
            .iter()
            .any(|s| s.len() != shape[0] || s.iter().any(|v| !v.is_finite()))
    {
        return Err(QuantizeError::Calibration {
            reason: format!(
                "expected nonempty finite FP32 matrix calibration samples with shape {shape:?}"
            ),
        });
    }
    let method = config.calibration_method.unwrap_or_default();
    let _: super::methods::CalibrationMethod = method.to_string().parse()?;
    let (weights, edges) = model.matrix_calibration_plan(&config)?;
    let outputs = Quantizer::new(config).quantize_selected_weights(&weights)?;
    if outputs
        .iter()
        .any(|o| o.qdq.scales.iter().any(|s| !s.is_finite() || *s <= 0.0))
    {
        return Err(QuantizeError::Calibration {
            reason: "matrix weight quantization produced an invalid scale".into(),
        });
    }
    let weights: Vec<_> = outputs.into_iter().map(|o| o.qdq).collect();
    let mut estimator = ActivationEstimator::for_tensors(model, input_path, &edges)?;
    estimator.calibrate_quiet(dataset)?;
    let stats = estimator
        .get_layer_stats()
        .into_iter()
        .map(|(k, v)| (k, v.clone()))
        .collect();
    let mut model = estimator.into_model();
    model.save_calibrated_operators(&weights, &stats, method, output_path, &["MatMul", "Gemm"])
}
