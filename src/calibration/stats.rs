use std::collections::HashMap;
use crate::calibration::methods::CalibrationMethod;

#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub count: usize,
    
    histogram: Vec<(f32, usize)>,
}

impl ActivationStats {
    pub fn from_data(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }
        
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;
        
        let variance: f32 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        
        let mut histogram = build_histogram(data, min, max, 100);
        histogram.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        Self {
            min,
            max,
            mean,
            std,
            count: data.len(),
            histogram,
        }
    }
    
    pub fn update(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        
        let data_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let data_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        self.min = self.min.min(data_min);
        self.max = self.max.max(data_max);
        
        let data_sum: f32 = data.iter().sum();
        let new_count = self.count + data.len();
        self.mean = (self.mean * self.count as f32 + data_sum) / new_count as f32;
        
        self.count = new_count;
        
        let new_hist = build_histogram(data, self.min, self.max, 100);
        merge_histograms(&mut self.histogram, &new_hist);
    }
    
    pub fn percentile(&self, p: f32) -> f32 {
        if self.histogram.is_empty() {
            return self.min;
        }
        
        // ceil, not truncation: for 5 elements at p=50, target rank must be 3
        // (the actual median), not 2 (which would return the element below it).
        let target_count = (self.count as f32 * p / 100.0).ceil() as usize;
        let mut cumulative = 0;
        
        for &(value, count) in &self.histogram {
            cumulative += count;
            if cumulative >= target_count {
                return value;
            }
        }
        
        self.max
    }
}

impl Default for ActivationStats {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            count: 0,
            histogram: Vec::new(),
        }
    }
}

fn build_histogram(data: &[f32], min: f32, max: f32, num_bins: usize) -> Vec<(f32, usize)> {
    if (max - min).abs() < 1e-8 {
        return vec![(min, data.len())];
    }
    
    let bin_size = (max - min) / num_bins as f32;
    let mut bins = vec![0usize; num_bins];
    
    for &value in data {
        let bin_idx = ((value - min) / bin_size).floor() as usize;
        let bin_idx = bin_idx.min(num_bins - 1);
        bins[bin_idx] += 1;
    }
    
    bins.iter()
        .enumerate()
        .filter(|(_, &count)| count > 0)
        .map(|(i, &count)| {
            let value = min + (i as f32 + 0.5) * bin_size;
            (value, count)
        })
        .collect()
}

fn merge_histograms(hist1: &mut Vec<(f32, usize)>, hist2: &[(f32, usize)]) {
    let mut merged = HashMap::new();
    
    for &(value, count) in hist1.iter() {
        *merged.entry(value_to_key(value)).or_insert(0) += count;
    }
    
    for &(value, count) in hist2 {
        *merged.entry(value_to_key(value)).or_insert(0) += count;
    }
    
    *hist1 = merged
        .into_iter()
        .map(|(key, count)| (key_to_value(key), count))
        .collect();
    
    hist1.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
}

fn value_to_key(v: f32) -> i32 {
    (v * 1000.0).round() as i32
}

fn key_to_value(k: i32) -> f32 {
    k as f32 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_activation_stats() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let stats = ActivationStats::from_data(&data);
        
        assert_eq!(stats.min, -1.0);
        assert_eq!(stats.max, 1.0);
        assert!((stats.mean - 0.0).abs() < 0.01);
        
        let p50 = stats.percentile(50.0);
        assert!((p50 - 0.0).abs() < 0.3);
    }
}

pub fn calculate_optimal_range(
    data: &[f32],
    method: CalibrationMethod,
) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    match method {
        CalibrationMethod::MinMax => {
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min, max)
        }
        
        CalibrationMethod::Percentile(p) => {
            let stats = ActivationStats::from_data(data);
            let lower = stats.percentile(100.0 - p);
            let upper = stats.percentile(p);
            (lower, upper)
        }
        
        CalibrationMethod::Entropy => {
            optimize_kl_divergence(data)
        }
        
        CalibrationMethod::MSE => {
            optimize_mse(data)
        }
    }
}

/// Optimize range using KL divergence (entropy method)
fn optimize_kl_divergence(data: &[f32]) -> (f32, f32) {
    let stats = ActivationStats::from_data(data);
    
    // Try different percentile thresholds and find the one with minimum KL divergence
    let candidates = [99.0, 99.5, 99.9, 99.95, 99.99];
    let mut best_range = (stats.min, stats.max);
    let mut best_kl = f32::INFINITY;
    
    for &percentile in &candidates {
        let lower = stats.percentile(100.0 - percentile);
        let upper = stats.percentile(percentile);
        
        let kl = calculate_kl_divergence(data, lower, upper);
        
        if kl < best_kl {
            best_kl = kl;
            best_range = (lower, upper);
        }
    }
    
    best_range
}

/// Optimize range using MSE minimization
fn optimize_mse(data: &[f32]) -> (f32, f32) {
    let stats = ActivationStats::from_data(data);
    
    // Try different percentile thresholds and find the one with minimum MSE
    let candidates = [99.0, 99.5, 99.9, 99.95, 99.99];
    let mut best_range = (stats.min, stats.max);
    let mut best_mse = f32::INFINITY;
    
    for &percentile in &candidates {
        let lower = stats.percentile(100.0 - percentile);
        let upper = stats.percentile(percentile);
        
        let mse = calculate_quantization_mse(data, lower, upper);
        
        if mse < best_mse {
            best_mse = mse;
            best_range = (lower, upper);
        }
    }
    
    best_range
}

/// Calculate KL divergence between original and quantized distribution
fn calculate_kl_divergence(data: &[f32], min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-8 {
        return 0.0;
    }
    
    let orig_hist = build_histogram(data, min, max, 128);
    
    // Simulate quantized distribution (clipped and quantized)
    let quantized_data: Vec<f32> = data.iter()
        .map(|&v| {
            let clipped = v.clamp(min, max);
            // Simulate INT8 quantization
            let scale = (max - min) / 255.0;
            let q = ((clipped - min) / scale).round();
            min + q * scale
        })
        .collect();
    
    let quant_hist = build_histogram(&quantized_data, min, max, 128);
    
    // Calculate KL divergence
    let mut kl = 0.0;
    let epsilon = 1e-10;
    
    for (&(_, p_count), &(_, q_count)) in orig_hist.iter().zip(quant_hist.iter()) {
        let p = (p_count as f32 + epsilon) / (data.len() as f32 + epsilon);
        let q = (q_count as f32 + epsilon) / (quantized_data.len() as f32 + epsilon);
        kl += p * (p / q).ln();
    }
    
    kl
}

fn calculate_quantization_mse(data: &[f32], min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-8 {
        return 0.0;
    }
    
    let scale = (max - min) / 255.0;
    
    let mse: f32 = data.iter()
        .map(|&v| {
            let clipped = v.clamp(min, max);
            let q = ((clipped - min) / scale).round();
            let dequantized = min + q * scale;
            (v - dequantized).powi(2)
        })
        .sum::<f32>() / data.len() as f32;
    
    mse
}