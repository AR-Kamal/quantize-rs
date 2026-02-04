// src/quantization/mod.rs
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub bits: u8,
    pub per_channel: bool,
    pub calibration_method: Option<crate::calibration::methods::CalibrationMethod>,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            per_channel: false,
            calibration_method: None,
        }
    }
}

impl QuantConfig {
    pub fn int8() -> Self {
        Self {
            bits: 8,
            per_channel: false,
            calibration_method: None,
        }
    }

    pub fn with_per_channel(mut self, enabled: bool) -> Self {
        self.per_channel = enabled;
        self
    }

    pub fn with_calibration(mut self, method: crate::calibration::methods::CalibrationMethod) -> Self {
        self.calibration_method = Some(method);
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QuantMode {
    Int8,
    Int4,
}

#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i8,
    pub min: f32,
    pub max: f32,
}

impl QuantParams {
    pub fn from_range(min: f32, max: f32) -> Self {
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

        let scale = (max - min) / (qmax - qmin);

        let scale = scale.max(1e-8);

        let initial_zero_point = qmin - min / scale;
        let zero_point = initial_zero_point.round().clamp(qmin, qmax) as i8;

        QuantParams {
            scale,
            zero_point,
            min,
            max,
        }
    }

    pub fn quantize(&self, value: f32) -> i8 {
        let quantized = (value / self.scale).round() + (self.zero_point as f32);
        quantized.clamp(-128.0, 127.0) as i8
    }

    pub fn dequantize(&self, value: i8) -> f32 {
        ((value as i32) - (self.zero_point as i32)) as f32 * self.scale
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
    pub params: QuantParams,
    pub per_channel: bool,
    pub channel_params: Option<Vec<QuantParams>>,
}

impl QuantizedTensor {
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let params = QuantParams::from_range(min, max);

        let quantized_data: Vec<i8> = data.iter()
            .map(|&v| params.quantize(v))
            .collect();

        Ok(QuantizedTensor {
            data: quantized_data,
            shape,
            params,
            per_channel: false,
            channel_params: None,
        })
    }

    /// Quantize with explicit range (for calibration)
    pub fn from_f32_with_range(data: &[f32], shape: Vec<usize>, min: f32, max: f32) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        let params = QuantParams::from_range(min, max);

        let quantized_data: Vec<i8> = data.iter()
            .map(|&v| params.quantize(v))
            .collect();

        Ok(QuantizedTensor {
            data: quantized_data,
            shape,
            params,
            per_channel: false,
            channel_params: None,
        })
    }
    
    /// Quantize with per-channel
    pub fn from_f32_per_channel(
        data: &[f32],
        shape: Vec<usize>,
        channel_axis: usize,
    ) -> Result<Self> {

        
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        // Get number of channels
        if channel_axis >= shape.len() {
            bail!("Channel axis {} out of bounds for shape {:?}", channel_axis, shape);
        }

        let num_channels = shape[channel_axis];
        
        // Calculate elements per channel
        let total_elements: usize = shape.iter().product();
        let _elements_per_channel = total_elements / num_channels;

        // Quantize each channel separately
        let mut channel_params = Vec::new();
        let mut quantized_data = Vec::with_capacity(data.len());

        for channel_idx in 0..num_channels {
            let channel_data = extract_channel(data, &shape, channel_axis, channel_idx);

            let min = channel_data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = channel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let params = QuantParams::from_range(min, max);
            channel_params.push(params.clone());

            for &value in &channel_data {
                quantized_data.push(params.quantize(value));
            }
        }

        // Use first channel params as "representative" for backward compatibility
        let params = channel_params[0].clone();

        Ok(QuantizedTensor {
            data: quantized_data,
            shape,
            params,
            per_channel: true,
            channel_params: Some(channel_params),
        })
    }

    pub fn to_f32(&self) -> Vec<f32> {
        if self.per_channel {
            if let Some(ref channel_params) = self.channel_params {
                self.data.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let channel_idx = i / (self.data.len() / channel_params.len());
                        channel_params[channel_idx].dequantize(v)
                    })
                    .collect()
            } else {
                // Fallback to per-tensor
                self.data.iter()
                    .map(|&v| self.params.dequantize(v))
                    .collect()
            }
        } else {
            self.data.iter()
                .map(|&v| self.params.dequantize(v))
                .collect()
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i8>()
    }

    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        if original.is_empty() {
            return 0.0;
        }
        
        let dequantized = self.to_f32();
        
        let sum: f32 = original.iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        sum / original.len() as f32
    }
}

/// INT4 Quantized tensor with optional bit packing
#[derive(Debug, Clone)]
pub struct QuantizedTensorInt4 {
    pub data: Vec<i8>,
    pub packed_data: Option<Vec<u8>>,
    pub shape: Vec<usize>,
    pub params: QuantParamsInt4,
    pub per_channel: bool,
    pub channel_params: Option<Vec<QuantParamsInt4>>,
}

impl QuantizedTensorInt4 {
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let params = QuantParamsInt4::from_range(min, max);

        let quantized_data: Vec<i8> = data.iter()
            .map(|&v| params.quantize(v))
            .collect();

        Ok(QuantizedTensorInt4 {
            data: quantized_data,
            packed_data: None,
            shape,
            params,
            per_channel: false,
            channel_params: None,
        })
    }

    pub fn from_f32_with_range(data: &[f32], shape: Vec<usize>, min: f32, max: f32) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        // Use provided range instead of computing from data
        let params = QuantParamsInt4::from_range(min, max);

        let quantized_data: Vec<i8> = data.iter()
            .map(|&v| params.quantize(v))
            .collect();

        Ok(QuantizedTensorInt4 {
            data: quantized_data,
            packed_data: None,
            shape,
            params,
            per_channel: false,
            channel_params: None,
        })
    }

    pub fn from_f32_per_channel(
        data: &[f32],
        shape: Vec<usize>,
        channel_axis: usize,
    ) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        if channel_axis >= shape.len() {
            bail!("Channel axis {} out of bounds for shape {:?}", channel_axis, shape);
        }

        let num_channels = shape[channel_axis];
        
        let total_elements: usize = shape.iter().product();
        let _elements_per_channel = total_elements / num_channels;

        let mut channel_params = Vec::new();
        let mut quantized_data = Vec::with_capacity(data.len());

        for channel_idx in 0..num_channels {
            let channel_data = extract_channel(data, &shape, channel_axis, channel_idx);

            let min = channel_data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = channel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let params = QuantParamsInt4::from_range(min, max);
            channel_params.push(params.clone());

            for &value in &channel_data {
                quantized_data.push(params.quantize(value));
            }
        }

        let params = channel_params[0].clone();

        Ok(QuantizedTensorInt4 {
            data: quantized_data,
            packed_data: None,
            shape,
            params,
            per_channel: true,
            channel_params: Some(channel_params),
        })
    }

    pub fn pack(&mut self) {
        self.packed_data = Some(pack_int4(&self.data));
    }

    pub fn ensure_unpacked(&self) -> Vec<i8> {
        if let Some(ref packed) = self.packed_data {
            unpack_int4(packed, self.data.len())
        } else {
            self.data.clone()
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let data = self.ensure_unpacked();
        
        if self.per_channel {
            if let Some(ref channel_params) = self.channel_params {
                data.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let channel_idx = i / (data.len() / channel_params.len());
                        channel_params[channel_idx].dequantize(v)
                    })
                    .collect()
            } else {
                data.iter()
                    .map(|&v| self.params.dequantize(v))
                    .collect()
            }
        } else {
            data.iter()
                .map(|&v| self.params.dequantize(v))
                .collect()
        }
    }

    pub fn size_bytes(&self) -> usize {
        if let Some(ref packed) = self.packed_data {
            packed.len() 
        } else {
            self.data.len() * std::mem::size_of::<i8>() 
        }
    }

    pub fn unpacked_size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i8>()
    }

    pub fn packed_size_bytes(&self) -> usize {
        if let Some(ref packed) = self.packed_data {
            packed.len()
        } else {
            (self.data.len() + 1) / 2
        }
    }

    pub fn is_packed(&self) -> bool {
        self.packed_data.is_some()
    }

    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        if original.is_empty() {
            return 0.0;
        }
        
        let dequantized = self.to_f32();
        
        let sum: f32 = original.iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        sum / original.len() as f32
    }
}


fn pack_int4_pair(val1: i8, val2: i8) -> u8 {
    debug_assert!(val1 >= -8 && val1 <= 7, "val1 out of INT4 range: {}", val1);
    debug_assert!(val2 >= -8 && val2 <= 7, "val2 out of INT4 range: {}", val2);
    
    // Convert to 4-bit representation
    let nibble1 = (val1 & 0x0F) as u8;
    let nibble2 = (val2 & 0x0F) as u8;
    
    // Pack: high 4 bits = val1, low 4 bits = val2
    (nibble1 << 4) | nibble2
}

fn unpack_int4_pair(byte: u8) -> (i8, i8) {
    let nibble1 = (byte >> 4) & 0x0F;
    let nibble2 = byte & 0x0F;
    
    // Convert from 4-bit to signed i8
    let val1 = if nibble1 >= 8 {
        (nibble1 as i8) | !0x0F  
    } else {
        nibble1 as i8
    };
    
    let val2 = if nibble2 >= 8 {
        (nibble2 as i8) | !0x0F  
    } else {
        nibble2 as i8
    };
    
    (val1, val2)
}

pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    
    for chunk in values.chunks(2) {
        let val1 = chunk[0];
        let val2 = if chunk.len() > 1 { chunk[1] } else { 0 };
        
        packed.push(pack_int4_pair(val1, val2));
    }
    
    packed
}

pub fn unpack_int4(packed: &[u8], num_values: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(num_values);
    
    for &byte in packed {
        let (val1, val2) = unpack_int4_pair(byte);
        values.push(val1);
        if values.len() < num_values {
            values.push(val2);
        }
    }
    
    // Truncate to exact size (removes padding)
    values.truncate(num_values);
    values
}

// Helper function
fn extract_channel(
    data: &[f32],
    shape: &[usize],
    channel_axis: usize,
    channel_idx: usize,
) -> Vec<f32> {
    let num_channels = shape[channel_axis];
    let elements_per_channel = data.len() / num_channels;
    
    let start = channel_idx * elements_per_channel;
    let end = start + elements_per_channel;
    
    data[start..end].to_vec()
}

#[derive(Debug, Clone)]
pub struct QuantParamsInt4 {
    pub scale: f32,
    pub zero_point: i8,
    pub min: f32,
    pub max: f32,
}

impl QuantParamsInt4 {
    pub fn from_range(min: f32, max: f32) -> Self {
        let min = min.min(0.0);
        let max = max.max(0.0);

        let (min, max) = if (max - min).abs() < 1e-8 {
            (min - 0.01, max + 0.01)
        } else {
            (min, max)
        };

        let qmin = -8.0_f32;
        let qmax = 7.0_f32;

        let scale = (max - min) / (qmax - qmin);

        let scale = scale.max(1e-8);

        let initial_zero_point = qmin - min / scale;
        let zero_point = initial_zero_point.round().clamp(qmin, qmax) as i8;

        QuantParamsInt4 {
            scale,
            zero_point,
            min,
            max,
        }
    }

    pub fn quantize(&self, value: f32) -> i8 {
        let quantized = (value / self.scale).round() + (self.zero_point as f32);
        quantized.clamp(-8.0, 7.0) as i8
    }

    pub fn dequantize(&self, value: i8) -> f32 {
        ((value as i32) - (self.zero_point as i32)) as f32 * self.scale
    }
}

#[derive(Debug, Clone)]
pub enum QuantizedTensorType {
    Int8(QuantizedTensor),
    Int4(QuantizedTensorInt4),
}

impl QuantizedTensorType {
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            QuantizedTensorType::Int8(t) => t.to_f32(),
            QuantizedTensorType::Int4(t) => t.to_f32(),
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            QuantizedTensorType::Int8(t) => t.size_bytes(),
            QuantizedTensorType::Int4(t) => t.size_bytes(),
        }
    }

    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        match self {
            QuantizedTensorType::Int8(t) => t.quantization_error(original),
            QuantizedTensorType::Int4(t) => t.quantization_error(original),
        }
    }
    
    pub fn data(&self) -> Vec<i8> {
        match self {
            QuantizedTensorType::Int8(t) => t.data.clone(),
            QuantizedTensorType::Int4(t) => t.ensure_unpacked(),
        }
    }

        pub fn get_scale_zero_point(&self) -> (f32, i8) {
        match self {
            QuantizedTensorType::Int8(t) => (t.params.scale, t.params.zero_point),
            QuantizedTensorType::Int4(t) => (t.params.scale, t.params.zero_point),
        }
    }
    
    pub fn bits(&self) -> u8 {
        match self {
            QuantizedTensorType::Int8(_) => 8,
            QuantizedTensorType::Int4(_) => 4,
        }
    }
    
    pub fn is_int8(&self) -> bool {
        matches!(self, QuantizedTensorType::Int8(_))
    }
    
    pub fn is_int4(&self) -> bool {
        matches!(self, QuantizedTensorType::Int4(_))
    }
}

pub struct Quantizer {
    config: QuantConfig,
    calibration_stats: Option<std::collections::HashMap<String, crate::calibration::stats::ActivationStats>>,
}

impl Quantizer {
    pub fn new(config: QuantConfig) -> Self {
        Self { 
            config,
            calibration_stats: None,
        }
    }
    
    pub fn with_calibration(
        config: QuantConfig,
        stats: std::collections::HashMap<String, crate::calibration::stats::ActivationStats>,
    ) -> Self {
        Self {
            config,
            calibration_stats: Some(stats),
        }
    }
    
    /// Quantize a tensor with optional calibration
    pub fn quantize_tensor_with_name(
        &self,
        name: &str,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensorType> {
        let (min, max) = if let Some(ref stats_map) = self.calibration_stats {
            if let Some(stats) = stats_map.get(name) {
                if let Some(method) = self.config.calibration_method {
                    use crate::calibration::stats::calculate_optimal_range;
                    
                    let sample_data = sample_from_activation_stats(stats, 1000);
                    calculate_optimal_range(&sample_data, method)
                } else {
                    (stats.min, stats.max)
                }
            } else {
                // No stats for this layer, use data min/max
                let min = data.iter().copied().fold(f32::INFINITY, f32::min);
                let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            }
        } else {
            // No calibration, use data min/max
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (min, max)
        };
        
        // Quantize with optimal range
        self.quantize_with_range(data, shape, min, max)
    }
    
    pub fn quantize_tensor(&self, data: &[f32], shape: Vec<usize>) -> Result<QuantizedTensorType> {
        match self.config.bits {
            8 => {
                let tensor = if self.config.per_channel && shape.len() >= 2 {
                    QuantizedTensor::from_f32_per_channel(data, shape, 0)?
                } else {
                    QuantizedTensor::from_f32(data, shape)?
                };
                Ok(QuantizedTensorType::Int8(tensor))
            },
            4 => {
                let mut tensor = if self.config.per_channel && shape.len() >= 2 {
                    QuantizedTensorInt4::from_f32_per_channel(data, shape, 0)?
                } else {
                    QuantizedTensorInt4::from_f32(data, shape)?
                };
                tensor.pack();
                Ok(QuantizedTensorType::Int4(tensor))
            },
            _ => bail!("Only INT8 and INT4 quantization supported"),
        }
    }
    
    /// Quantize with specific range (for calibration)
    fn quantize_with_range(
        &self,
        data: &[f32],
        shape: Vec<usize>,
        min: f32,
        max: f32,
    ) -> Result<QuantizedTensorType> {
        match self.config.bits {
            8 => {
                let tensor = if self.config.per_channel && shape.len() >= 2 {
                    QuantizedTensor::from_f32_per_channel(data, shape, 0)?
                } else {
                    QuantizedTensor::from_f32_with_range(data, shape, min, max)?
                };
                Ok(QuantizedTensorType::Int8(tensor))
            },
            4 => {
                let mut tensor = if self.config.per_channel && shape.len() >= 2 {
                    QuantizedTensorInt4::from_f32_per_channel(data, shape, 0)?
                } else {
                    QuantizedTensorInt4::from_f32_with_range(data, shape, min, max)?
                };
                tensor.pack();
                Ok(QuantizedTensorType::Int4(tensor))
            },
            _ => bail!("Only INT8 and INT4 quantization supported"),
        }
    }
}

/// Helper: Sample data from activation statistics
fn sample_from_activation_stats(stats: &crate::calibration::stats::ActivationStats, n: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
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
    fn test_quant_params() {
        let params = QuantParams::from_range(-1.0, 1.0);

        assert_eq!(params.quantize(0.0), params.zero_point);

        let original = 0.5;
        let quantized = params.quantize(original);
        let dequantized = params.dequantize(quantized);

        assert!((original - dequantized).abs() < 0.01);
    }

    #[test]
    fn test_quantize_tensor() {
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let shape = vec![5];

        let quantized = QuantizedTensor::from_f32(&data, shape).unwrap();

        assert_eq!(quantized.data.len(), 5);
        assert_eq!(quantized.size_bytes(), 5);
    }

    #[test]
    fn test_per_channel_quantization() {
        let mut data = vec![];
        for _ in 0..100 {
            data.push(0.5); // Channel 0
        }
        for _ in 0..100 {
            data.push(5.0); // Channel 1
        }
        
        let shape = vec![2, 100];
        
        let quantized = QuantizedTensor::from_f32_per_channel(&data, shape, 0).unwrap();
        
        assert!(quantized.per_channel);
        assert!(quantized.channel_params.is_some());
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 2);
        
        let dequantized = quantized.to_f32();
        let error: f32 = data.iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / data.len() as f32;
        
        println!("Per-channel MSE: {}", error);
        assert!(error < 0.1);
    }

    #[test]
    fn test_per_channel_vs_per_tensor() {
        let mut data = vec![];
    
        for _ in 0..1000 {
            data.push(0.01);
        }
        
        for _ in 0..1000 {
            data.push(10.0);
        }
        
        let shape = vec![2, 1000];
        
        // Per-tensor quantization
        let per_tensor = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let per_tensor_error = per_tensor.quantization_error(&data);
        
        // Per-channel quantization
        let per_channel = QuantizedTensor::from_f32_per_channel(&data, shape, 0).unwrap();
        let per_channel_error = per_channel.quantization_error(&data);
        
        println!("Per-tensor error:  {:.8}", per_tensor_error);
        println!("Per-channel error: {:.8}", per_channel_error);
        
        // Per-channel
        assert!(per_channel_error < per_tensor_error);
        assert!(per_channel_error < per_tensor_error * 0.5);
    }

    #[test]
    fn test_per_channel_benefit() {
        let mut data = vec![];
        
        for i in 0..1000 {
            data.push(-0.1 + (i as f32 / 1000.0) * 0.2);
        }
        
        for i in 0..1000 {
            data.push(-10.0 + (i as f32 / 1000.0) * 20.0);
        }
        
        let shape = vec![2, 1000];
        
        let per_tensor = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let per_tensor_error = per_tensor.quantization_error(&data);
        
        let per_channel = QuantizedTensor::from_f32_per_channel(&data, shape, 0).unwrap();
        let per_channel_error = per_channel.quantization_error(&data);
        
        println!("Per-tensor MSE:  {:.8}", per_tensor_error);
        println!("Per-channel MSE: {:.8}", per_channel_error);
        
        assert!(per_channel_error < per_tensor_error, 
                "Per-channel ({:.8}) should be better than per-tensor ({:.8})", 
                per_channel_error, per_tensor_error);
    }

    #[test]
    fn test_int4_quant_params() {
        let params = QuantParamsInt4::from_range(-1.0, 1.0);
        
        assert!(params.quantize(-10.0) >= -8);
        assert!(params.quantize(-10.0) <= 7);
        assert!(params.quantize(10.0) >= -8);
        assert!(params.quantize(10.0) <= 7);
        
        let zero_quant = params.quantize(0.0);
        assert!(zero_quant >= -8 && zero_quant <= 7);
        
        for &original in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let quantized = params.quantize(original);
            let dequantized = params.dequantize(quantized);
            
            println!("Original: {:.2}, Quantized: {}, Dequantized: {:.2}, Error: {:.4}", 
                     original, quantized, dequantized, (original - dequantized).abs());
            
            assert!((original - dequantized).abs() < params.scale * 2.0);
        }
    }

    #[test]
    fn test_int4_extreme_values() {
        // Test with extreme value ranges
        let params = QuantParamsInt4::from_range(-100.0, 100.0);
        
        let q_neg = params.quantize(-100.0);
        let q_pos = params.quantize(100.0);
        
        assert_eq!(q_neg, -8);
        assert_eq!(q_pos, 7);
    }

    #[test]
    fn test_int4_vs_int8_error() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        
        let params_int8 = QuantParams::from_range(-1.0, 1.0);
        let error_int8: f32 = data.iter()
            .map(|&v| {
                let q = params_int8.quantize(v);
                let dq = params_int8.dequantize(q);
                (v - dq).powi(2)
            })
            .sum::<f32>() / data.len() as f32;
        
        let params_int4 = QuantParamsInt4::from_range(-1.0, 1.0);
        let error_int4: f32 = data.iter()
            .map(|&v| {
                let q = params_int4.quantize(v);
                let dq = params_int4.dequantize(q);
                (v - dq).powi(2)
            })
            .sum::<f32>() / data.len() as f32;
        
        println!("INT8 MSE: {:.8}", error_int8);
        println!("INT4 MSE: {:.8}", error_int4);
        
        assert!(error_int4 > error_int8);
        
        assert!(error_int4 < error_int8 * 500.0, 
                "INT4 error ({:.8}) is too high compared to INT8 ({:.8})", 
                error_int4, error_int8);

        assert!(error_int4.is_finite());
        assert!(error_int4 < 0.01);
        
    }

    #[test]
    fn test_int4_range() {
        let params = QuantParamsInt4::from_range(-1.0, 1.0);
        
        assert!(params.quantize(-10.0) == -8);
        assert!(params.quantize(10.0) == 7);
        
        // Test quantization within range
        for i in -8..=7 {
            let value = i as f32 * params.scale;
            let quantized = params.quantize(value);
            assert!(quantized >= -8 && quantized <= 7);
        }
    }

    #[test]
    fn test_int4_optimal_precision() {
        let params = QuantParamsInt4::from_range(-1.0, 1.0);
        
        let mut unique_values = std::collections::HashSet::new();
        
        // Sample across the range
        for i in 0..1000 {
            let value = -1.0 + (i as f32 / 1000.0) * 2.0;
            unique_values.insert(params.quantize(value));
        }
     
        println!("Unique quantized values: {}", unique_values.len());
        assert!(unique_values.len() >= 14);
    }

    #[test]
    fn test_int4_tensor_quantization() {
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let shape = vec![5];

        let quantized = QuantizedTensorInt4::from_f32(&data, shape).unwrap();

        assert_eq!(quantized.data.len(), 5);
        assert_eq!(quantized.size_bytes(), 5);
        assert_eq!(quantized.packed_size_bytes(), 3);
        
        for &val in &quantized.data {
            assert!(val >= -8 && val <= 7, "Value {} out of INT4 range", val);
        }
    }

    #[test]
    fn test_int4_round_trip() {
        let original = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let shape = vec![5];

        let quantized = QuantizedTensorInt4::from_f32(&original, shape).unwrap();
        let dequantized = quantized.to_f32();

        println!("Original:    {:?}", original);
        println!("Quantized:   {:?}", quantized.data);
        println!("Dequantized: {:?}", dequantized);

        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            println!("  {:.2} -> {:.2}, error: {:.4}", orig, deq, error);
            assert!(error < 0.15, "Error too large: {}", error);
        }
    }

    #[test]
    fn test_int4_per_channel() {
        let mut data = vec![];
        
        // Channel 0: small range [-0.1, 0.1]
        for i in 0..100 {
            data.push(-0.1 + (i as f32 / 100.0) * 0.2);
        }
        
        // Channel 1: large range [-10.0, 10.0]
        for i in 0..100 {
            data.push(-10.0 + (i as f32 / 100.0) * 20.0);
        }
        
        let shape = vec![2, 100];
        
        let quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape, 0).unwrap();
        
        assert!(quantized.per_channel);
        assert!(quantized.channel_params.is_some());
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 2);
        
        let error = quantized.quantization_error(&data);
        println!("INT4 per-channel MSE: {:.8}", error);
        
        assert!(error < 1.0, "Error too high: {}", error);
    }

    #[test]
    fn test_int4_vs_int8_compression() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 2.0 - 1.0).collect();
        let shape = vec![1000];
        
        let int8_quantized = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let int8_size = int8_quantized.size_bytes();
        let int8_error = int8_quantized.quantization_error(&data);
        
        let int4_quantized = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        let int4_size = int4_quantized.size_bytes();
        let int4_packed_size = int4_quantized.packed_size_bytes();
        let int4_error = int4_quantized.quantization_error(&data);
        
        println!("INT8: {} bytes, MSE: {:.8}", int8_size, int8_error);
        println!("INT4 (unpacked): {} bytes, MSE: {:.8}", int4_size, int4_error);
        println!("INT4 (packed): {} bytes, MSE: {:.8}", int4_packed_size, int4_error);
        
        assert_eq!(int4_size, int8_size);
        
        assert!(int4_packed_size <= int8_size / 2 + 1);
        
        assert!(int4_error > int8_error);
        
        assert!(int4_error < 0.01, "INT4 error too high: {}", int4_error);
    }

    #[test]
    fn test_int4_large_tensor() {
        let size = 64 * 3 * 3 * 3; // 64 filters, 3x3x3 kernels
        let data: Vec<f32> = (0..size).map(|i| {
            ((i as f32 / size as f32) * 2.0 - 1.0) * 0.5
        }).collect();
        
        let shape = vec![64, 3, 3, 3];
        
        let quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape, 0).unwrap();
        
        assert_eq!(quantized.data.len(), size);
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 64);
        
        let error = quantized.quantization_error(&data);
        println!("Large tensor INT4 error: {:.8}", error);
        
        assert!(error < 0.01, "Error too high for large tensor: {}", error);
    }

    #[test]
    fn test_int4_extreme_ranges() {
        let test_cases = vec![
            (vec![-0.001, 0.0, 0.001], "tiny range"),
            (vec![-100.0, 0.0, 100.0], "large range"),
            (vec![0.0, 0.0, 0.0], "all zeros"),
            (vec![1.0, 1.0, 1.0], "all same"),
        ];
        
        for (data, desc) in test_cases {
            println!("\nTesting: {}", desc);
            let shape = vec![data.len()];
            
            let result = QuantizedTensorInt4::from_f32(&data, shape);
            assert!(result.is_ok(), "Failed on {}", desc);
            
            let quantized = result.unwrap();
            let dequantized = quantized.to_f32();
            
            println!("  Original:    {:?}", data);
            println!("  Dequantized: {:?}", dequantized);
            
            for &val in &quantized.data {
                assert!(val >= -8 && val <= 7, "Value {} out of range for {}", val, desc);
            }
        }
    }

    #[test]
    fn test_int4_pack_unpack_pair() {
        let test_cases = vec![
            (-8, 7),
            (-8, -8),
            (7, 7),
            (0, 0),
            (-1, 0),
            (0, -1),
            (-5, 3),
            (6, -4),
        ];
        
        for (val1, val2) in test_cases {
            println!("\nTesting: ({}, {})", val1, val2);
            
            let packed = pack_int4_pair(val1, val2);
            let (unpacked1, unpacked2) = unpack_int4_pair(packed);
            
            println!("  Packed: 0x{:02X} (binary: {:08b})", packed, packed);
            println!("  Unpacked: ({}, {})", unpacked1, unpacked2);
            
            assert_eq!(val1, unpacked1, "First value mismatch");
            assert_eq!(val2, unpacked2, "Second value mismatch");
        }
    }

    #[test]
    fn test_int4_pack_unpack_vector() {
        let values = vec![-8, -7, -1, 0, 1, 7];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        
        println!("\nEven length:");
        println!("  Original: {:?}", values);
        println!("  Packed:   {:?} ({} bytes)", packed, packed.len());
        println!("  Unpacked: {:?}", unpacked);
        
        assert_eq!(values, unpacked);
        assert_eq!(packed.len(), (values.len() + 1) / 2);
    }

    #[test]
    fn test_int4_pack_unpack_odd_length() {
        let values = vec![-8, -5, 0, 5, 7];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        
        println!("\nOdd length:");
        println!("  Original: {:?}", values);
        println!("  Packed:   {:?} ({} bytes)", packed, packed.len());
        println!("  Unpacked: {:?}", unpacked);
        
        assert_eq!(values, unpacked);
        assert_eq!(packed.len(), (values.len() + 1) / 2);
    }

    #[test]
    fn test_int4_pack_all_values() {
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        
        println!("\nAll INT4 values:");
        println!("  Original: {:?}", values);
        println!("  Packed:   {} bytes", packed.len());
        println!("  Unpacked: {:?}", unpacked);
        
        assert_eq!(values, unpacked);
        assert_eq!(packed.len(), 8);
    }

    #[test]
    fn test_int4_pack_large_vector() {
        let values: Vec<i8> = (0..1000).map(|i| ((i % 16) - 8) as i8).collect();
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        
        assert_eq!(values, unpacked);
        assert_eq!(packed.len(), 500);
        
        println!("\nLarge vector:");
        println!("  Original: {} values", values.len());
        println!("  Packed:   {} bytes ({}x compression)", packed.len(), 
                values.len() / packed.len());
        println!("  Unpacked: {} values", unpacked.len());
    }

    #[test]
    fn test_int4_compression_ratio() {
        let size = 10000;
        let values: Vec<i8> = (0..size).map(|i| ((i % 16) - 8) as i8).collect();
        
        let unpacked_size = values.len() * std::mem::size_of::<i8>();
        
        let packed = pack_int4(&values);
        let packed_size = packed.len();
        
        let compression_ratio = unpacked_size as f32 / packed_size as f32;
        
        println!("\nCompression test:");
        println!("  Values:      {}", size);
        println!("  Unpacked:    {} bytes", unpacked_size);
        println!("  Packed:      {} bytes", packed_size);
        println!("  Compression: {:.2}x", compression_ratio);
        
        assert!((compression_ratio - 2.0).abs() < 0.01, 
                "Expected ~2x compression, got {:.2}x", compression_ratio);
    }

    #[test]
    fn test_int4_tensor_packing() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 2.0 - 1.0).collect();
        let shape = vec![1000];
        
        let mut quantized = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        
        println!("Before packing:");
        println!("  Unpacked size: {} bytes", quantized.unpacked_size_bytes());
        println!("  Is packed: {}", quantized.is_packed());
        
        assert!(!quantized.is_packed());
        assert_eq!(quantized.size_bytes(), 1000);
        
        quantized.pack();
        
        println!("\nAfter packing:");
        println!("  Packed size: {} bytes", quantized.size_bytes());
        println!("  Is packed: {}", quantized.is_packed());
        println!("  Compression: {}x", quantized.unpacked_size_bytes() / quantized.size_bytes());
        
        assert!(quantized.is_packed());
        assert_eq!(quantized.size_bytes(), 500);
        
        let dequantized = quantized.to_f32();
        assert_eq!(dequantized.len(), 1000);
        
        let error = quantized.quantization_error(&data);
        println!("  MSE after packing: {:.8}", error);
        assert!(error < 0.01);
    }

    #[test]
    fn test_int4_packed_vs_unpacked_error() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
        let shape = vec![100];
        
        let unpacked = QuantizedTensorInt4::from_f32(&data, shape.clone()).unwrap();
        let error_unpacked = unpacked.quantization_error(&data);
        
        let mut packed = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        packed.pack();
        let error_packed = packed.quantization_error(&data);
        
        println!("Unpacked error: {:.8}", error_unpacked);
        println!("Packed error:   {:.8}", error_packed);
        
        assert!((error_unpacked - error_packed).abs() < 1e-6);
    }

    #[test]
    fn test_int4_per_channel_packing() {
        let mut data = vec![];
        for i in 0..500 {
            data.push((i as f32 / 500.0) * 0.2 - 0.1); // Channel 0
        }
        for i in 0..500 {
            data.push((i as f32 / 500.0) * 20.0 - 10.0); // Channel 1
        }
        
        let shape = vec![2, 500];
        
        let mut quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape, 0).unwrap();
        
        let error_before = quantized.quantization_error(&data);
        println!("Error before packing: {:.8}", error_before);
        
        quantized.pack();
        
        let error_after = quantized.quantization_error(&data);
        println!("Error after packing:  {:.8}", error_after);
        println!("Size: {} bytes (packed from {} bytes)", 
                quantized.size_bytes(), 
                quantized.unpacked_size_bytes());
        
        assert!((error_before - error_after).abs() < 1e-6);
        
        assert_eq!(quantized.size_bytes(), 500);
    }

    #[test]
    fn test_int4_compression_comparison() {
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| {
            ((i as f32 / size as f32) * 2.0 - 1.0) * 0.5
        }).collect();
        let shape = vec![size];
        
        let fp32_size = size * std::mem::size_of::<f32>();
        
        let int8 = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let int8_size = int8.size_bytes();
        
        let int4_unpacked = QuantizedTensorInt4::from_f32(&data, shape.clone()).unwrap();
        let int4_unpacked_size = int4_unpacked.size_bytes();
        
        let mut int4_packed = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        int4_packed.pack();
        let int4_packed_size = int4_packed.size_bytes();
        
        println!("\nCompression Comparison:");
        println!("  FP32:          {} bytes", fp32_size);
        println!("  INT8:          {} bytes ({:.1}x)", int8_size, fp32_size as f32 / int8_size as f32);
        println!("  INT4 unpacked: {} bytes ({:.1}x)", int4_unpacked_size, fp32_size as f32 / int4_unpacked_size as f32);
        println!("  INT4 packed:   {} bytes ({:.1}x)", int4_packed_size, fp32_size as f32 / int4_packed_size as f32);
        
        assert_eq!(fp32_size / int8_size, 4); // 4x compression
        assert_eq!(fp32_size / int4_packed_size, 8); // 8x compression!
    }

    #[test]
    #[ignore] // Run manually with: cargo test test_int4_real_model -- --ignored --nocapture
    fn test_int4_real_model() {
        use crate::onnx_utils::OnnxModel;
        
        println!("\n{}", "=".repeat(60));
        println!("INT4 Real Model Test");
        println!("\n{}", "=".repeat(60));
        
        let model_paths = vec![
            "test_models/mnist.onnx",
            "mnist.onnx",
            "test_models/resnet18-v1-7.onnx",
            "resnet18-v1-7.onnx",
        ];
        
        let mut model = None;
        for path in &model_paths {
            if std::path::Path::new(path).exists() {
                println!("Loading model: {}", path);
                match OnnxModel::load(path) {
                    Ok(m) => {
                        model = Some(m);
                        break;
                    }
                    Err(e) => println!("  Failed: {}", e),
                }
            }
        }
        
        let model = match model {
            Some(m) => m,
            None => {
                println!("No test models found. Skipping test.");
                println!("Place mnist.onnx or resnet18-v1-7.onnx in current directory.");
                return;
            }
        };
        
        let info = model.info();
        println!("✓ Model loaded: {}", info.name);
        println!("  Nodes: {}", info.num_nodes);
        println!();
        
        println!("Extracting weights...");
        let weights = model.extract_weights();
        println!("✓ Found {} weight tensors", weights.len());
        
        if weights.is_empty() {
            println!("No weights to quantize!");
            return;
        }
        
        println!();
        println!("\n{}", "=".repeat(60));
        println!("Testing Per-Tensor Quantization");
        println!("\n{}", "=".repeat(60));
        
        let test_weights: Vec<_> = weights.iter()
            .filter(|w| w.data.len() > 1000) 
            .take(5)
            .collect();
        
        println!("Testing {} large layers:\n", test_weights.len());
        
        for (idx, weight) in test_weights.iter().enumerate() {
            let name = if weight.name.len() > 40 {
                format!("{}...", &weight.name[..37])
            } else {
                weight.name.clone()
            };
            
            println!("[{}] {}", idx + 1, name);
            println!("    Shape: {:?}, Elements: {}", weight.shape, weight.data.len());
            
            let fp32_size = weight.data.len() * 4;
            
            let int8_result = QuantizedTensor::from_f32(&weight.data, weight.shape.clone());
            let (int8_size, int8_error) = if let Ok(q) = int8_result {
                (q.size_bytes(), q.quantization_error(&weight.data))
            } else {
                println!("    INT8 failed!");
                continue;
            };
            
            let int4_result = QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone());
            let (int4_unpacked_size, int4_error) = if let Ok(q) = int4_result {
                (q.size_bytes(), q.quantization_error(&weight.data))
            } else {
                println!("    INT4 failed!");
                continue;
            };
            
            let mut int4_packed = QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone()).unwrap();
            int4_packed.pack();
            let int4_packed_size = int4_packed.size_bytes();
            let int4_packed_error = int4_packed.quantization_error(&weight.data);
            
            println!("    FP32:          {:7} bytes", fp32_size);
            println!("    INT8:          {:7} bytes ({:.1}x) MSE: {:.8}", 
                    int8_size, fp32_size as f32 / int8_size as f32, int8_error);
            println!("    INT4 unpacked: {:7} bytes ({:.1}x) MSE: {:.8}", 
                    int4_unpacked_size, fp32_size as f32 / int4_unpacked_size as f32, int4_error);
            println!("    INT4 packed:   {:7} bytes ({:.1}x) MSE: {:.8}", 
                    int4_packed_size, fp32_size as f32 / int4_packed_size as f32, int4_packed_error);
            
            assert_eq!(int4_error, int4_packed_error, "Packing changed error!");
            
            let int8_ratio = fp32_size as f32 / int8_size as f32;
            let int4_ratio = fp32_size as f32 / int4_packed_size as f32;
            
            assert!((int8_ratio - 4.0).abs() < 0.1, "INT8 compression should be ~4x");
            assert!((int4_ratio - 8.0).abs() < 0.1, "INT4 compression should be ~8x");
            
            println!();
        }
        
        println!("\n{}", "=".repeat(60));
        println!("Testing Per-Channel Quantization");
        println!("\n{}", "=".repeat(60));
        
        // Test per-channel on Conv layers (multi-dimensional)
        let conv_weights: Vec<_> = weights.iter()
            .filter(|w| w.shape.len() >= 2 && w.shape[0] > 1) 
            .take(3)
            .collect();
        
        if conv_weights.is_empty() {
            println!("No multi-channel layers found for per-channel test.");
        } else {
            println!("Testing {} conv layers:\n", conv_weights.len());
            
            for (idx, weight) in conv_weights.iter().enumerate() {
                let name = if weight.name.len() > 40 {
                    format!("{}...", &weight.name[..37])
                } else {
                    weight.name.clone()
                };
                
                println!("[{}] {}", idx + 1, name);
                println!("    Shape: {:?}, Channels: {}", weight.shape, weight.shape[0]);
                
                let per_tensor = QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone()).unwrap();
                let per_tensor_error = per_tensor.quantization_error(&weight.data);
                
                let per_channel_result = QuantizedTensorInt4::from_f32_per_channel(
                    &weight.data, 
                    weight.shape.clone(), 
                    0
                );
                
                if let Ok(per_channel) = per_channel_result {
                    let per_channel_error = per_channel.quantization_error(&weight.data);
                    
                    let improvement = ((per_tensor_error - per_channel_error) / per_tensor_error) * 100.0;
                    
                    println!("    Per-tensor:  MSE: {:.8}", per_tensor_error);
                    println!("    Per-channel: MSE: {:.8} ({:.1}% better)", 
                            per_channel_error, improvement);
                    
                    assert!(per_channel_error <= per_tensor_error * 1.1, 
                        "Per-channel should not be significantly worse");
                } else {
                    println!("    Per-channel failed!");
                }
                
                println!();
            }
        }
        
        println!("\n{}", "=".repeat(60));
        println!("Summary");
        println!("\n{}", "=".repeat(60));
        
        println!("✓ INT4 quantization works on real model weights");
        println!("✓ Compression ratios correct (4x INT8, 8x INT4)");
        println!("✓ Bit packing is lossless");
        println!("✓ Per-channel quantization works");
        println!("\nINT4 implementation is ready for CLI integration!");
    }
}