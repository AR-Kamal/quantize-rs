// src\quantization\mod.rs
//! Quantization algorithms

use anyhow::{bail, Result};

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub bits: u8,
    pub per_channel: bool,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            per_channel: false,
        }
    }
}

impl QuantConfig {
    pub fn int8() -> Self {
        Self {
            bits: 8,
            per_channel: false,
        }
    }

    pub fn with_per_channel(mut self, enabled: bool) -> Self {
        self.per_channel = enabled;
        self
    }
}

/// Quantization mode
#[derive(Debug, Clone, Copy)]
pub enum QuantMode {
    Int8,
    Int4,
}

/// Quantization parameters (scale and zero-point)
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i8,
    pub min: f32,
    pub max: f32,
}

impl QuantParams {
    /// Calculate quantization parameters from float data
    pub fn from_range(min: f32, max: f32) -> Self {
        // Ensure range includes zero
        let min = min.min(0.0);
        let max = max.max(0.0);

        // Handle edge case: if min == max, set a small range
        let (min, max) = if (max - min).abs() < 1e-8 {
            (min - 0.01, max + 0.01)
        } else {
            (min, max)
        };

        // INT8 range: -128 to 127
        let qmin = -128.0_f32;
        let qmax = 127.0_f32;

        // Calculate scale
        let scale = (max - min) / (qmax - qmin);

        // Handle edge case: ensure scale is not too small
        let scale = scale.max(1e-8);

        // Calculate zero point
        let initial_zero_point = qmin - min / scale;
        let zero_point = initial_zero_point.round().clamp(qmin, qmax) as i8;

        QuantParams {
            scale,
            zero_point,
            min,
            max,
        }
    }

    /// Quantize a float32 value to INT8
    pub fn quantize(&self, value: f32) -> i8 {
        let quantized = (value / self.scale).round() + (self.zero_point as f32);
        quantized.clamp(-128.0, 127.0) as i8
    }

    /// Dequantize an INT8 value back to float32
    pub fn dequantize(&self, value: i8) -> f32 {
        // (value - self.zero_point) as f32 * self.scale
        // Cast to i32 first to avoid overflow
        ((value as i32) - (self.zero_point as i32)) as f32 * self.scale
    }
}

/// Quantized tensor
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
    pub params: QuantParams,
}

impl QuantizedTensor {
    /// Quantize a float32 tensor
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        // Find min/max
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Calculate quantization parameters
        let params = QuantParams::from_range(min, max);

        // Quantize all values
        let quantized_data: Vec<i8> = data.iter().map(|&v| params.quantize(v)).collect();

        Ok(QuantizedTensor {
            data: quantized_data,
            shape,
            params,
        })
    }

    /// Dequantize back to float32
    pub fn to_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&v| self.params.dequantize(v))
            .collect()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i8>()
    }

    /// Calculate quantization error (MSE)
    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        if original.is_empty() {
            return 0.0;
        }

        let dequantized = self.to_f32();

        let sum: f32 = original
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let mse = sum / original.len() as f32;
        mse
    }
}

/// Main quantizer
pub struct Quantizer {
    config: QuantConfig,
}

impl Quantizer {
    pub fn new(config: QuantConfig) -> Self {
        Self { config }
    }

    /// Quantize a tensor
    pub fn quantize_tensor(&self, data: &[f32], shape: Vec<usize>) -> Result<QuantizedTensor> {
        match self.config.bits {
            8 => QuantizedTensor::from_f32(data, shape),
            _ => bail!("Only INT8 quantization supported currently"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_params() {
        let params = QuantParams::from_range(-1.0, 1.0);

        // Test quantization
        assert_eq!(params.quantize(0.0), params.zero_point);

        // Test round-trip
        let original = 0.5;
        let quantized = params.quantize(original);
        let dequantized = params.dequantize(quantized);

        // Should be close (within quantization error)
        assert!((original - dequantized).abs() < 0.01);
    }

    #[test]
    fn test_quantize_tensor() {
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let shape = vec![5];

        let quantized = QuantizedTensor::from_f32(&data, shape).unwrap();

        assert_eq!(quantized.data.len(), 5);
        assert_eq!(quantized.size_bytes(), 5); // 5 bytes (INT8)
    }
}
