// src/calibration/inference.rs
//! Real activation-based calibration using tract inference.
//!
//! Statistics are keyed by original ONNX tensor names, including input edges.
//! Use `quantize_static` to apply these ranges to activation QDQ pairs.
//! Weight ranges must always be computed from the weights themselves.

use crate::errors::{QuantizeError, Result};
use std::collections::HashMap;
use tract_onnx::prelude::*;

use crate::calibration::stats::ActivationStats;
use crate::calibration::CalibrationDataset;
use crate::onnx_utils::OnnxModel;

// ===========================================================================
// Public API
// ===========================================================================

/// Runs calibration samples through a model and collects activation statistics.
///
/// Usage:
/// ```ignore
/// let model = OnnxModel::load("model.onnx")?;
/// let mut estimator = ActivationEstimator::new(model, "model.onnx")?;
/// let dataset = CalibrationDataset::from_numpy("samples.npy")?;
/// estimator.calibrate(&dataset)?;
/// let stats = estimator.get_layer_stats();  // HashMap<tensor_name, &ActivationStats>
/// ```
pub struct ActivationEstimator {
    /// Original ONNX model (preserved for later use in quantization)
    model: OnnxModel,
    /// tract runnable model with all intermediate outputs exposed
    #[allow(clippy::type_complexity)]
    tract_model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    /// Collected activation stats per ONNX tensor
    layer_stats: HashMap<String, ActivationStats>,
    /// Mapping from tract output index → layer name
    output_names: Vec<String>,
    /// Static quantization must not discard non-finite or non-FP32 activations.
    strict: bool,
}

impl std::fmt::Debug for ActivationEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationEstimator")
            .field("model", &self.model)
            .field("layer_stats_count", &self.layer_stats.len())
            .field("output_names_count", &self.output_names.len())
            .finish()
    }
}

impl ActivationEstimator {
    /// Load model and prepare for calibration.
    ///
    /// This:
    ///   1. Reloads the ONNX file with tract (we need the filepath)
    ///   2. Exposes all layer outputs as model outputs
    ///   3. Optimizes the graph
    ///   4. Creates a runnable plan
    ///
    /// **Important:** The `model` parameter must have been loaded from a file
    /// on disk. We re-parse that file with tract. If the model was constructed
    /// programmatically or the file no longer exists, this will fail.
    pub fn from_path(model: OnnxModel, onnx_path: &str) -> Result<Self> {
        Self::prepare(model, onnx_path, None)
    }

    pub(crate) fn for_tensors(model: OnnxModel, onnx_path: &str, names: &[String]) -> Result<Self> {
        Self::prepare(model, onnx_path, Some(names))
    }

    fn prepare(model: OnnxModel, onnx_path: &str, names: Option<&[String]>) -> Result<Self> {
        // --- Load with tract ---
        let mut tract_model = tract_onnx::onnx().model_for_path(onnx_path).map_err(|e| {
            QuantizeError::Calibration {
                reason: format!("tract failed to load ONNX model '{}': {e}", onnx_path),
            }
        })?;

        // tract labels computational outputs but names sources by ONNX input
        // name. Give those input outlets explicit labels as well.
        for outlet in tract_model.inputs.clone() {
            let label = tract_model.nodes[outlet.node].name.clone();
            tract_model.set_outlet_label(outlet, label).map_err(|e| {
                QuantizeError::Calibration {
                    reason: format!("failed to label model input: {e}"),
                }
            })?;
        }
        // Resolve original ONNX tensor labels BEFORE optimization. Output slots
        // preserve their order across tract optimization, even when nodes fuse.
        let output_names = if let Some(names) = names {
            names.to_vec()
        } else {
            let mut labels: Vec<_> = tract_model.outlet_labels.values().cloned().collect();
            labels.sort();
            labels.dedup();
            labels
        };
        let outlets: Result<Vec<_>> = output_names
            .iter()
            .map(|name| {
                tract_model
                    .outlet_labels
                    .iter()
                    .find(|(_, label)| *label == name)
                    .map(|(outlet, _)| *outlet)
                    .ok_or_else(|| QuantizeError::Calibration {
                        reason: format!("tract could not resolve ONNX tensor '{name}'"),
                    })
            })
            .collect();
        tract_model
            .set_output_outlets(&outlets?)
            .map_err(|e| QuantizeError::Calibration {
                reason: format!("failed to expose activation tensors: {e}"),
            })?;
        let optimized_model =
            tract_model
                .into_optimized()
                .map_err(|e| QuantizeError::Calibration {
                    reason: format!("tract optimization failed: {e}"),
                })?;
        if optimized_model.outputs.len() != output_names.len() {
            return Err(QuantizeError::Calibration {
                reason: "tract changed the number of requested activation outputs".into(),
            });
        }

        let tract_model =
            optimized_model
                .into_runnable()
                .map_err(|e| QuantizeError::Calibration {
                    reason: format!("tract failed to create runnable plan: {e}"),
                })?;

        Ok(Self {
            model,
            tract_model,
            layer_stats: HashMap::new(),
            output_names,
            strict: names.is_some(),
        })
    }

    /// Convenience constructor when you have the model and its path.
    pub fn new(model: OnnxModel, onnx_path: &str) -> Result<Self> {
        Self::from_path(model, onnx_path)
    }

    /// Run calibration samples through the model and collect activation statistics.
    ///
    /// For each sample:
    ///   - Run inference
    ///   - Capture all intermediate tensors
    ///   - Update min/max/histogram for each layer
    ///
    /// Progress is printed every 10 batches.  Library consumers who want
    /// silent calibration (no `println!` on stdout) should call
    /// [`calibrate_quiet`](Self::calibrate_quiet) instead.
    pub fn calibrate(&mut self, dataset: &CalibrationDataset) -> Result<()> {
        self.calibrate_inner(dataset, /* quiet */ false)
    }

    /// Same as [`calibrate`](Self::calibrate) but emits no progress output.
    ///
    /// Useful for library consumers embedding quantize-rs in their own UI
    /// (Python bindings, web services, batch pipelines that already report
    /// progress at a higher level).
    pub fn calibrate_quiet(&mut self, dataset: &CalibrationDataset) -> Result<()> {
        self.calibrate_inner(dataset, /* quiet */ true)
    }

    fn calibrate_inner(&mut self, dataset: &CalibrationDataset, quiet: bool) -> Result<()> {
        if dataset.is_empty() {
            return Err(QuantizeError::Calibration {
                reason: "Calibration dataset is empty".into(),
            });
        }

        if !quiet {
            println!(
                "Running activation-based calibration on {} samples...",
                dataset.len()
            );
        }

        let num_samples = dataset.len();

        for (sample_idx, sample) in dataset.samples.iter().enumerate() {
            self.process_sample(sample, &dataset.shape)?;

            if !quiet
                && ((sample_idx + 1) % (num_samples / 10).max(1) == 0
                    || sample_idx == num_samples - 1)
            {
                println!("  Processed {}/{} samples", sample_idx + 1, num_samples);
            }
        }

        if !quiet {
            println!(
                "✓ Calibration complete: {} layers tracked",
                self.layer_stats.len()
            );
        }
        Ok(())
    }

    /// Process a single calibration sample.
    fn process_sample(&mut self, sample: &[f32], shape: &[usize]) -> Result<()> {
        // --- Prepare input tensor ---
        // tract expects shape [batch, channels, height, width] for images, or
        // [batch, ...] in general. Calibration samples are typically single
        // images without a batch dim, so we prepend batch=1.
        let mut input_shape = vec![1]; // batch size
        input_shape.extend_from_slice(shape);

        let input_tensor =
            tract_core::prelude::Tensor::from_shape(&input_shape, sample).map_err(|e| {
                QuantizeError::Calibration {
                    reason: format!("Failed to create input tensor from calibration sample: {e}"),
                }
            })?;

        // --- Run inference ---
        let outputs = self
            .tract_model
            .run(tvec!(input_tensor.into()))
            .map_err(|e| QuantizeError::Calibration {
                reason: format!("tract inference failed on calibration sample: {e}"),
            })?;

        // --- Update statistics for each output ---
        for (output_idx, tvalue) in outputs.iter().enumerate() {
            // Get the layer name for this output
            let layer_name = if output_idx < self.output_names.len() {
                &self.output_names[output_idx]
            } else {
                // Fallback: use index as name if mapping is incomplete
                // (shouldn't happen, but defensive)
                continue;
            };

            // Convert TValue to Tensor
            // into_tensor() consumes, so we clone first
            let tensor = tvalue.clone().into_tensor();

            // Extract f32 data from the tensor
            if self.strict && tensor.to_array_view::<f32>().is_err() {
                return Err(QuantizeError::Calibration {
                    reason: format!("activation '{layer_name}' is not FP32"),
                });
            }
            let data = extract_f32_data(&tensor)?;
            if self.strict && data.iter().any(|v| !v.is_finite()) {
                return Err(QuantizeError::Calibration {
                    reason: format!("activation '{layer_name}' contains non-finite values"),
                });
            }

            // Update or create ActivationStats
            self.layer_stats
                .entry(layer_name.clone())
                .and_modify(|stats| stats.update(&data))
                .or_insert_with(|| ActivationStats::from_data(&data));
        }

        Ok(())
    }

    /// Get collected activation statistics for all layers (borrowed).
    ///
    /// Returns a map from layer name → &ActivationStats. These stats include
    /// min/max (for range optimization) and histogram (for entropy/MSE methods).
    pub fn get_layer_stats(&self) -> HashMap<String, &ActivationStats> {
        self.layer_stats
            .iter()
            .map(|(name, stats)| (name.clone(), stats))
            .collect()
    }

    /// Consume and return owned activation statistics.
    ///
    /// Keys are original ONNX tensor names, not node or weight names.
    pub fn into_layer_stats(self) -> HashMap<String, ActivationStats> {
        self.layer_stats
    }

    /// Consume the estimator and return the original OnnxModel.
    ///
    /// Useful when you need the model back but have already extracted stats
    /// with `get_layer_stats()` (borrowed). For the typical quantization
    /// pipeline, use `into_layer_stats()` to get owned stats, then reload
    /// the model separately for quantization.
    pub fn into_model(self) -> OnnxModel {
        self.model
    }

    /// Borrow the original model.
    pub fn model(&self) -> &OnnxModel {
        &self.model
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Extract f32 data from a tract tensor.
///
/// tract tensors can be various types (f32, f16, i32, etc.). For activation
/// statistics we only care about f32. If the tensor is another type, convert it.
fn extract_f32_data(tensor: &Tensor) -> Result<Vec<f32>> {
    // Try to access as f32 directly
    match tensor.to_array_view::<f32>() {
        Ok(view) => {
            // Success: already f32, just collect into Vec
            Ok(view.iter().copied().collect())
        }
        Err(_) => {
            // Not f32: try to cast
            let tensor_f32 = tensor
                .cast_to::<f32>()
                .map_err(|e| QuantizeError::Calibration {
                    reason: format!("Failed to cast tensor to f32 for activation statistics: {e}"),
                })?;

            let view =
                tensor_f32
                    .to_array_view::<f32>()
                    .map_err(|e| QuantizeError::Calibration {
                        reason: format!("Tensor cast succeeded but array view failed: {e}"),
                    })?;

            Ok(view.iter().copied().collect())
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires ONNX model file on disk
    fn test_activation_estimator_real_inference() {
        // Run with: cargo test test_activation_estimator_real_inference -- --ignored --nocapture

        let model_paths = vec![
            "mnist.onnx",
            "test_models/mnist.onnx",
            "resnet18-v1-7.onnx",
            "test_models/resnet18-v1-7.onnx",
        ];

        let mut found_path = None;
        for path in model_paths {
            if std::path::Path::new(path).exists() {
                found_path = Some(path);
                break;
            }
        }

        let model_path = match found_path {
            Some(p) => p,
            None => {
                println!(
                    "No test model found. Place mnist.onnx or resnet18-v1-7.onnx in project root."
                );
                return;
            }
        };

        println!("Testing with model: {}", model_path);

        // Load model
        let model = OnnxModel::load(model_path).expect("Failed to load model");
        let info = model.info();
        println!("Model: {}, {} nodes", info.name, info.num_nodes);

        // Determine input shape (MNIST = [1, 28, 28], ResNet = [3, 224, 224])
        let input_shape = if model_path.contains("mnist") {
            vec![1, 28, 28]
        } else {
            vec![3, 224, 224]
        };

        // Create calibration dataset (just 5 samples for testing)
        let dataset = CalibrationDataset::random(input_shape, 5, (0.0, 1.0)).unwrap();

        // Run calibration
        let mut estimator = ActivationEstimator::new(model, model_path)
            .expect("Failed to create ActivationEstimator");

        estimator.calibrate(&dataset).expect("Calibration failed");

        // Verify we got stats
        let stats = estimator.get_layer_stats();
        assert!(!stats.is_empty(), "No activation statistics collected");

        println!("\nCollected stats for {} layers:", stats.len());
        for (name, stat) in stats.iter().take(5) {
            println!(
                "  {}: min={:.4}, max={:.4}, mean={:.4}",
                name,
                stat.min(),
                stat.max(),
                stat.mean()
            );
        }

        // Sanity check: activations should have reasonable ranges
        // (not all zeros, not all same value)
        for (name, stat) in stats.iter() {
            assert!(
                (stat.max() - stat.min()).abs() > 1e-6,
                "Layer {} has constant output (min={}, max={})",
                name,
                stat.min(),
                stat.max()
            );
        }
    }

    #[test]
    #[ignore]
    fn test_calibration_dataset_integration() {
        // This verifies the full pipeline: dataset → estimator → stats

        let model_path = "mnist.onnx";
        if !std::path::Path::new(model_path).exists() {
            println!("mnist.onnx not found, skipping integration test");
            return;
        }

        let model = OnnxModel::load(model_path).unwrap();
        let dataset = CalibrationDataset::random(vec![1, 28, 28], 10, (0.0, 1.0)).unwrap();
        let mut estimator = ActivationEstimator::new(model, model_path).unwrap();

        estimator.calibrate(&dataset).unwrap();

        let stats = estimator.get_layer_stats();
        assert!(!stats.is_empty());

        // All stats should have count = 10 samples
        for stat in stats.values() {
            // Each layer sees data from all samples (aggregated)
            assert!(stat.count() > 0);
        }
    }
}
