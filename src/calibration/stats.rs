use crate::calibration::methods::CalibrationMethod;

const NUM_BINS: usize = 256;

#[derive(Debug, Clone)]
pub struct ActivationStats {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    count: usize,

    /// Running sum of squared deviations (Welford's M2) for incremental std.
    m2: f64,

    histogram_bins: Vec<usize>,
    hist_min: f32,
    hist_max: f32,
}

impl ActivationStats {
    /// Minimum observed value.
    pub fn min(&self) -> f32 { self.min }
    /// Maximum observed value.
    pub fn max(&self) -> f32 { self.max }
    /// Running mean.
    pub fn mean(&self) -> f32 { self.mean }
    /// Running standard deviation.
    pub fn std(&self) -> f32 { self.std }
    /// Number of observations.
    pub fn count(&self) -> usize { self.count }
}

impl ActivationStats {
    pub fn from_data(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let min = data.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);

        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;

        let m2: f64 = data.iter()
            .map(|&x| ((x - mean) as f64).powi(2))
            .sum();
        let std = (m2 / data.len() as f64).sqrt() as f32;

        let histogram_bins = build_histogram(data, min, max);

        Self {
            min,
            max,
            mean,
            std,
            count: data.len(),
            m2,
            histogram_bins,
            hist_min: min,
            hist_max: max,
        }
    }

    pub fn update(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let data_min = data.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
        let data_max = data.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);

        let new_min = self.min.min(data_min);
        let new_max = self.max.max(data_max);

        // Parallel/batch variant of Welford's online algorithm:
        // Merge two populations (existing stats + new batch) into combined stats.
        let old_count = self.count as f64;
        let new_count = data.len() as f64;
        let combined_count = old_count + new_count;

        let data_sum: f64 = data.iter().map(|&x| x as f64).sum();
        let data_mean = data_sum / new_count;

        let data_m2: f64 = data.iter()
            .map(|&x| ((x as f64) - data_mean).powi(2))
            .sum();

        // Chan's parallel algorithm for combining M2 values
        let delta = data_mean - self.mean as f64;
        self.m2 = self.m2 + data_m2 + delta * delta * old_count * new_count / combined_count;

        self.mean = ((self.mean as f64) * old_count + data_sum) as f32 / combined_count as f32;
        self.count = combined_count as usize;
        self.std = (self.m2 / combined_count).sqrt() as f32;

        // If range expanded, re-bin existing data into the new range
        if new_min < self.hist_min || new_max > self.hist_max {
            let mut rebinned = vec![0usize; NUM_BINS];
            rebin(&self.histogram_bins, self.hist_min, self.hist_max, &mut rebinned, new_min, new_max);
            self.histogram_bins = rebinned;
            self.hist_min = new_min;
            self.hist_max = new_max;
        }

        // Add new data into bins
        let new_hist = build_histogram(data, self.hist_min, self.hist_max);
        for (i, &c) in new_hist.iter().enumerate() {
            self.histogram_bins[i] += c;
        }

        self.min = new_min;
        self.max = new_max;
    }

    pub fn percentile(&self, p: f32) -> f32 {
        if self.histogram_bins.is_empty() {
            return self.min;
        }

        let total: usize = self.histogram_bins.iter().sum();
        if total == 0 {
            return self.min;
        }

        // ceil, not truncation: for 5 elements at p=50, target rank must be 3
        // (the actual median), not 2 (which would return the element below it).
        let target_count = (total as f32 * p / 100.0).ceil() as usize;
        let mut cumulative = 0;

        let bin_size = if (self.hist_max - self.hist_min).abs() < 1e-8 {
            0.0
        } else {
            (self.hist_max - self.hist_min) / NUM_BINS as f32
        };

        for (i, &count) in self.histogram_bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return self.hist_min + (i as f32 + 0.5) * bin_size;
            }
        }

        self.max
    }

    /// Return histogram data as (bin_center, count) pairs.
    pub fn histogram_data(&self) -> Vec<(f32, usize)> {
        if (self.hist_max - self.hist_min).abs() < 1e-8 {
            let total: usize = self.histogram_bins.iter().sum();
            if total > 0 {
                return vec![(self.hist_min, total)];
            }
            return Vec::new();
        }
        let bin_size = (self.hist_max - self.hist_min) / NUM_BINS as f32;
        self.histogram_bins.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(i, &count)| {
                let value = self.hist_min + (i as f32 + 0.5) * bin_size;
                (value, count)
            })
            .collect()
    }
}

impl Default for ActivationStats {
    fn default() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            std: 0.0,
            count: 0,
            m2: 0.0,
            histogram_bins: Vec::new(),
            hist_min: 0.0,
            hist_max: 0.0,
        }
    }
}

fn build_histogram(data: &[f32], min: f32, max: f32) -> Vec<usize> {
    let mut bins = vec![0usize; NUM_BINS];

    if (max - min).abs() < 1e-8 {
        // All values map to a single bin
        let finite_count = data.iter().filter(|v| v.is_finite()).count();
        if !bins.is_empty() {
            bins[0] = finite_count;
        }
        return bins;
    }

    let bin_size = (max - min) / NUM_BINS as f32;

    for &value in data {
        if !value.is_finite() { continue; }
        let bin_idx = ((value - min) / bin_size).floor() as usize;
        let bin_idx = bin_idx.min(NUM_BINS - 1);
        bins[bin_idx] += 1;
    }

    bins
}

/// Re-bin histogram data from one range to another.
fn rebin(
    old_bins: &[usize],
    old_min: f32,
    old_max: f32,
    new_bins: &mut [usize],
    new_min: f32,
    new_max: f32,
) {
    if old_bins.is_empty() || new_bins.is_empty() {
        return;
    }
    let old_range = old_max - old_min;
    let new_range = new_max - new_min;
    if old_range.abs() < 1e-8 || new_range.abs() < 1e-8 {
        // Everything goes into the closest bin in the new range
        let total: usize = old_bins.iter().sum();
        if total > 0 {
            let center = (old_min + old_max) * 0.5;
            let idx = ((center - new_min) / new_range * new_bins.len() as f32).floor() as usize;
            let idx = idx.min(new_bins.len() - 1);
            new_bins[idx] += total;
        }
        return;
    }
    let old_bin_size = old_range / old_bins.len() as f32;
    let new_bin_count = new_bins.len();
    for (i, &count) in old_bins.iter().enumerate() {
        if count == 0 { continue; }
        let center = old_min + (i as f32 + 0.5) * old_bin_size;
        let new_idx = ((center - new_min) / new_range * new_bin_count as f32).floor() as usize;
        let new_idx = new_idx.min(new_bin_count - 1);
        new_bins[new_idx] += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_stats() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let stats = ActivationStats::from_data(&data);

        assert_eq!(stats.min(), -1.0);
        assert_eq!(stats.max(), 1.0);
        assert!((stats.mean() - 0.0).abs() < 0.01);

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
            let min = data.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
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

/// Calculate KL divergence between original and quantized distribution.
///
/// Uses dense, aligned bins so every bin index in the original histogram
/// maps to the same value range in the quantized histogram.
fn calculate_kl_divergence(data: &[f32], min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-8 {
        return 0.0;
    }

    let num_bins = 128;
    let bin_size = (max - min) / num_bins as f32;
    let scale = (max - min) / 255.0;

    let mut orig_bins = vec![0usize; num_bins];
    let mut quant_bins = vec![0usize; num_bins];

    for &v in data {
        let clipped = v.clamp(min, max);

        // Original bin
        let bin = ((clipped - min) / bin_size).floor() as usize;
        let bin = bin.min(num_bins - 1);
        orig_bins[bin] += 1;

        // Simulated INT8 quantize -> dequantize, then bin
        let q = ((clipped - min) / scale).round();
        let dequant = min + q * scale;
        let qbin = ((dequant.clamp(min, max) - min) / bin_size).floor() as usize;
        let qbin = qbin.min(num_bins - 1);
        quant_bins[qbin] += 1;
    }

    let n = data.len() as f32;
    let epsilon = 1e-10_f32;
    let mut kl = 0.0_f32;

    for i in 0..num_bins {
        let p = (orig_bins[i] as f32 + epsilon) / (n + epsilon * num_bins as f32);
        let q = (quant_bins[i] as f32 + epsilon) / (n + epsilon * num_bins as f32);
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
