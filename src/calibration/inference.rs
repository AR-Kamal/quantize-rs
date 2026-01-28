use anyhow::Result;
use std::collections::HashMap;
use crate::onnx_utils::OnnxModel;
use crate::calibration::stats::ActivationStats;
use crate::calibration::CalibrationDataset;

#[derive(Debug, Clone)]
pub enum LayerType {
    Conv,
    Linear,
    BatchNorm,
    Activation,
}

#[derive(Debug, Clone)]
pub struct LayerActivation {
    pub name: String,
    pub layer_type: LayerType,
    pub input_stats: Option<ActivationStats>,
    pub output_stats: Option<ActivationStats>,
    pub weight_stats: Option<ActivationStats>,
}

/// Activation estimator - tracks data flow through network
pub struct ActivationEstimator {
    model: OnnxModel,
    layers: Vec<LayerActivation>,
    // layer_map: HashMap<String, usize>,
}

impl ActivationEstimator {
    pub fn new(model: OnnxModel) -> Self {
        let mut layers = Vec::new();
        let mut layer_map = HashMap::new();
        
        let weights = model.extract_weights();
        
        for (idx, weight) in weights.iter().enumerate() {
            let layer_type = infer_layer_type(&weight.name, &weight.shape);
            let weight_stats = ActivationStats::from_data(&weight.data);
            
            let layer = LayerActivation {
                name: weight.name.clone(),
                layer_type,
                input_stats: None,
                output_stats: None,
                weight_stats: Some(weight_stats),
            };
            
            layer_map.insert(weight.name.clone(), idx);
            layers.push(layer);
        }
        
        Self {
            model,
            layers,
            // layer_map,
        }
    }
    
    pub fn calibrate(&mut self, dataset: &CalibrationDataset) -> Result<()> {
        println!("Running calibration on {} samples...", dataset.len());
        
        let batch_size = 10;
        let num_batches = (dataset.len() + batch_size - 1) / batch_size;
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let batch = dataset.get_batch(start, batch_size);
            
            self.process_batch(batch)?;
            
            if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
                println!("  Processed {}/{} batches", batch_idx + 1, num_batches);
            }
        }
        
        println!("✓ Calibration complete");
        Ok(())
    }
    
    fn process_batch(&mut self, batch: &[Vec<f32>]) -> Result<()> {
        for sample in batch {
            self.forward_pass(sample)?;
        }
        Ok(())
    }
    
    /// Simulate forward pass to collect activation statistics
    fn forward_pass(&mut self, input: &[f32]) -> Result<()> {
        let mut current_stats = ActivationStats::from_data(input);
        
        // Propagate through layers
        for layer in &mut self.layers {
            // Update input stats for this layer
            if layer.input_stats.is_none() {
                layer.input_stats = Some(current_stats.clone());
            } else {
                // Merge with existing stats
                layer.input_stats.as_mut().unwrap().update(
                    &sample_from_stats(&current_stats, 1000)
                );
            }
            
            // Estimate output based on layer type and weights
            current_stats = estimate_output_stats(
                &current_stats,
                &layer.weight_stats,
                &layer.layer_type,
            );
            
            // Update output stats for this layer
            if layer.output_stats.is_none() {
                layer.output_stats = Some(current_stats.clone());
            } else {
                layer.output_stats.as_mut().unwrap().update(
                    &sample_from_stats(&current_stats, 1000)
                );
            }
        }
        
        Ok(())
    }
    
    /// Get activation statistics for all layers
    pub fn get_layer_stats(&self) -> HashMap<String, &ActivationStats> {
        let mut stats = HashMap::new();
        
        for layer in &self.layers {
            if let Some(ref output_stats) = layer.output_stats {
                stats.insert(layer.name.clone(), output_stats);
            }
        }
        
        stats
    }
    
    /// Get model reference
    pub fn model(&self) -> &OnnxModel {
        &self.model
    }

    pub fn into_model(self) -> OnnxModel {
        self.model
    }
    
}

fn infer_layer_type(name: &str, shape: &[usize]) -> LayerType {
    let name_lower = name.to_lowercase();
    
    if name_lower.contains("conv") {
        LayerType::Conv
    } else if name_lower.contains("dense") || name_lower.contains("linear") || name_lower.contains("fc") {
        LayerType::Linear
    } else if name_lower.contains("bn") || name_lower.contains("batchnorm") {
        LayerType::BatchNorm
    } else if shape.len() >= 2 {
        LayerType::Linear
    } else {
        LayerType::Activation
    }
}

fn estimate_output_stats(
    input_stats: &ActivationStats,
    weight_stats: &Option<ActivationStats>,
    layer_type: &LayerType,
) -> ActivationStats {
    match layer_type {
        LayerType::Conv | LayerType::Linear => {
            if let Some(w_stats) = weight_stats {
                
                // Estimate output range using weight and input statistics
                let out_min = w_stats.min * input_stats.max.abs().max(input_stats.min.abs());
                let out_max = w_stats.max * input_stats.max.abs().max(input_stats.min.abs());
                
                let range_factor = 0.7;
                let estimated_min = out_min * range_factor;
                let estimated_max = out_max * range_factor;
                
                let mut stats = ActivationStats::default();
                stats.min = estimated_min.min(estimated_max);
                stats.max = estimated_min.max(estimated_max);
                stats.mean = (stats.min + stats.max) / 2.0;
                stats.std = (stats.max - stats.min) / 4.0;
                stats.count = input_stats.count;
                
                stats
            } else {
                // No weights, pass through
                input_stats.clone()
            }
        }
        
        LayerType::BatchNorm => {
            // BatchNorm normalizes, so output is typically in [-3, 3] range
            let mut stats = ActivationStats::default();
            stats.min = -3.0;
            stats.max = 3.0;
            stats.mean = 0.0;
            stats.std = 1.0;
            stats.count = input_stats.count;
            stats
        }
        
        LayerType::Activation => {
            // For activations like ReLU, clip negative values
            let mut stats = input_stats.clone();
            if stats.min < 0.0 {
                stats.min = 0.0;
            }
            stats
        }
    }
}

/// Generate sample data from statistics (for merging)
fn sample_from_stats(stats: &ActivationStats, n: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Generate samples using normal distribution
    (0..n)
        .map(|_| {
            let sample = rng.gen::<f32>() * stats.std + stats.mean;
            sample.clamp(stats.min, stats.max)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Requires ONNX model
    fn test_activation_estimator() {
        // This test requires a real ONNX model
        // Run with: cargo test test_activation_estimator -- --ignored
        
        if let Ok(model) = OnnxModel::load("test_models/mnist.onnx") {
            let mut estimator = ActivationEstimator::new(model);
            
            // Create random calibration data
            let dataset = CalibrationDataset::random(vec![1, 28, 28], 10, (0.0, 1.0));
            
            estimator.calibrate(&dataset).unwrap();
            
            let stats = estimator.get_layer_stats();
            assert!(!stats.is_empty());
            
            println!("Collected stats for {} layers", stats.len());
        }
    }
}