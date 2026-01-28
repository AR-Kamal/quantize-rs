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
    pub per_channel: bool,
    pub channel_params: Option<Vec<QuantParams>>,
}

impl QuantizedTensor {
    /// Quantize a float32 tensor (per-tensor)
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

    /// Quantize with per-channel support
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
            // Extract channel data
            let channel_data = extract_channel(data, &shape, channel_axis, channel_idx);

            // Find min/max for this channel
            let min = channel_data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = channel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Calculate params for this channel
            let params = QuantParams::from_range(min, max);
            channel_params.push(params.clone());

            // Quantize channel
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

    /// Dequantize back to float32
    pub fn to_f32(&self) -> Vec<f32> {
        if self.per_channel {
            if let Some(ref channel_params) = self.channel_params {
                // Per-channel dequantization
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
            // Per-tensor dequantization
            self.data.iter()
                .map(|&v| self.params.dequantize(v))
                .collect()
        }
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
    /// Quantize a float32 tensor to INT4 (per-tensor)
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        if data.is_empty() {
            bail!("Cannot quantize empty tensor");
        }

        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Calculate INT4 quantization parameters
        let params = QuantParamsInt4::from_range(min, max);

        // Quantize all values
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

    /// Quantize with per-channel support
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

            // Calculate INT4 params for this channel
            let params = QuantParamsInt4::from_range(min, max);
            channel_params.push(params.clone());

            // Quantize channel
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

    /// Pack the INT4 data into compact format (2 values per byte)
    pub fn pack(&mut self) {
        self.packed_data = Some(pack_int4(&self.data));
    }

    /// Unpack data if it's packed
    pub fn ensure_unpacked(&self) -> Vec<i8> {
        if let Some(ref packed) = self.packed_data {
            unpack_int4(packed, self.data.len())
        } else {
            self.data.clone()
        }
    }

    /// Dequantize back to float32
    pub fn to_f32(&self) -> Vec<f32> {
        let data = self.ensure_unpacked();
        
        if self.per_channel {
            if let Some(ref channel_params) = self.channel_params {
                // Per-channel dequantization
                data.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let channel_idx = i / (data.len() / channel_params.len());
                        channel_params[channel_idx].dequantize(v)
                    })
                    .collect()
            } else {
                // Fallback to per-tensor
                data.iter()
                    .map(|&v| self.params.dequantize(v))
                    .collect()
            }
        } else {
            // Per-tensor dequantization
            data.iter()
                .map(|&v| self.params.dequantize(v))
                .collect()
        }
    }

    /// Get size in bytes (current storage)
    pub fn size_bytes(&self) -> usize {
        if let Some(ref packed) = self.packed_data {
            packed.len() 
        } else {
            self.data.len() * std::mem::size_of::<i8>() 
        }
    }

    /// Get unpacked size in bytes
    pub fn unpacked_size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i8>()
    }

    /// Get packed size in bytes
    pub fn packed_size_bytes(&self) -> usize {
        if let Some(ref packed) = self.packed_data {
            packed.len()
        } else {
            (self.data.len() + 1) / 2
        }
    }

    /// Check if data is currently packed
    pub fn is_packed(&self) -> bool {
        self.packed_data.is_some()
    }

    /// Calculate quantization error (MSE)
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


/// Pack two INT4 values into one byte
fn pack_int4_pair(val1: i8, val2: i8) -> u8 {
    debug_assert!(val1 >= -8 && val1 <= 7, "val1 out of INT4 range: {}", val1);
    debug_assert!(val2 >= -8 && val2 <= 7, "val2 out of INT4 range: {}", val2);
    
    // Convert to 4-bit representation
    let nibble1 = (val1 & 0x0F) as u8;
    let nibble2 = (val2 & 0x0F) as u8;
    
    // Pack: high 4 bits = val1, low 4 bits = val2
    (nibble1 << 4) | nibble2
}

/// Unpack one byte into two INT4 values
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

/// Pack a vector of INT4 values into packed bytes
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    
    for chunk in values.chunks(2) {
        let val1 = chunk[0];
        let val2 = if chunk.len() > 1 { chunk[1] } else { 0 };
        
        packed.push(pack_int4_pair(val1, val2));
    }
    
    packed
}

/// Unpack packed bytes into INT4 values
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

// Helper function to extract channel data
fn extract_channel(
    data: &[f32],
    shape: &[usize],
    channel_axis: usize,
    channel_idx: usize,
) -> Vec<f32> {
    // Simplified: assumes NCHW format for now
    let num_channels = shape[channel_axis];
    let elements_per_channel = data.len() / num_channels;
    
    let start = channel_idx * elements_per_channel;
    let end = start + elements_per_channel;
    
    data[start..end].to_vec()
}

/// INT4 quantization parameters (range: -8 to 7, asymmetric)
#[derive(Debug, Clone)]
pub struct QuantParamsInt4 {
    pub scale: f32,
    pub zero_point: i8,
    pub min: f32,
    pub max: f32,
}

impl QuantParamsInt4 {
    /// Calculate INT4 quantization parameters from float data
    /// Uses asymmetric range [-8, 7] for optimal precision
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

        // INT4 range: -8 to 7 (asymmetric, 16 values - maximum precision)
        let qmin = -8.0_f32;
        let qmax = 7.0_f32;

        // Calculate scale
        let scale = (max - min) / (qmax - qmin);

        // Handle edge case: ensure scale is not too small
        let scale = scale.max(1e-8);

        // Calculate zero point
        let initial_zero_point = qmin - min / scale;
        let zero_point = initial_zero_point.round().clamp(qmin, qmax) as i8;

        QuantParamsInt4 {
            scale,
            zero_point,
            min,
            max,
        }
    }

    /// Quantize a float32 value to INT4 range [-8, 7]
    pub fn quantize(&self, value: f32) -> i8 {
        let quantized = (value / self.scale).round() + (self.zero_point as f32);
        quantized.clamp(-8.0, 7.0) as i8
    }

    /// Dequantize an INT4 value back to float32
    pub fn dequantize(&self, value: i8) -> f32 {
        // Cast to i32 first to avoid overflow
        ((value as i32) - (self.zero_point as i32)) as f32 * self.scale
    }
}

/// Enum to hold either INT8 or INT4 quantized tensor
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

/// Main quantizer
pub struct Quantizer {
    config: QuantConfig,
}

impl Quantizer {
    pub fn new(config: QuantConfig) -> Self {
        Self { config }
    }

    /// Quantize a tensor
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

    #[test]
    fn test_per_channel_quantization() {
        let mut data = vec![];
        for _ in 0..100 {
            data.push(0.5); // Channel 0
        }
        for _ in 0..100 {
            data.push(5.0); // Channel 1
        }
        
        let shape = vec![2, 100]; // 2 channels, 100 elements each
        
        let quantized = QuantizedTensor::from_f32_per_channel(&data, shape, 0).unwrap();
        
        assert!(quantized.per_channel);
        assert!(quantized.channel_params.is_some());
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 2);
        
        // Dequantize and check error
        let dequantized = quantized.to_f32();
        let error: f32 = data.iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / data.len() as f32;
        
        println!("Per-channel MSE: {}", error);
        assert!(error < 0.1); // Should be very low
    }

    #[test]
    fn test_per_channel_vs_per_tensor() {
        // Create data with very different channel ranges (like real Conv layers)
        let mut data = vec![];
        
        // Channel 0: small values
        for _ in 0..1000 {
            data.push(0.01);
        }
        
        // Channel 1: large values
        for _ in 0..1000 {
            data.push(10.0);
        }
        
        let shape = vec![2, 1000]; // 2 channels, 1000 elements each
        
        // Per-tensor quantization
        let per_tensor = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let per_tensor_error = per_tensor.quantization_error(&data);
        
        // Per-channel quantization
        let per_channel = QuantizedTensor::from_f32_per_channel(&data, shape, 0).unwrap();
        let per_channel_error = per_channel.quantization_error(&data);
        
        println!("Per-tensor error:  {:.8}", per_tensor_error);
        println!("Per-channel error: {:.8}", per_channel_error);
        
        // Per-channel should be MUCH better
        assert!(per_channel_error < per_tensor_error);
        assert!(per_channel_error < per_tensor_error * 0.5); // At least 2x better
    }

    #[test]
    fn test_per_channel_benefit() {
        // Create data with very different channel ranges
        let mut data = vec![];
        
        // Channel 0: small values (-0.1 to 0.1)
        for i in 0..1000 {
            data.push(-0.1 + (i as f32 / 1000.0) * 0.2);
        }
        
        // Channel 1: large values (-10.0 to 10.0)
        for i in 0..1000 {
            data.push(-10.0 + (i as f32 / 1000.0) * 20.0);
        }
        
        let shape = vec![2, 1000];
        
        // Per-tensor quantization
        let per_tensor = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let per_tensor_error = per_tensor.quantization_error(&data);
        
        // Per-channel quantization
        let per_channel = QuantizedTensor::from_f32_per_channel(&data, shape, 0).unwrap();
        let per_channel_error = per_channel.quantization_error(&data);
        
        println!("Per-tensor MSE:  {:.8}", per_tensor_error);
        println!("Per-channel MSE: {:.8}", per_channel_error);
        
        // Per-channel should be significantly better
        assert!(per_channel_error < per_tensor_error, 
                "Per-channel ({:.8}) should be better than per-tensor ({:.8})", 
                per_channel_error, per_tensor_error);
    }

    #[test]
    fn test_int4_quant_params() {
        let params = QuantParamsInt4::from_range(-1.0, 1.0);
        
        // Test quantization range
        assert!(params.quantize(-10.0) >= -8);
        assert!(params.quantize(-10.0) <= 7);
        assert!(params.quantize(10.0) >= -8);
        assert!(params.quantize(10.0) <= 7);
        
        // Test zero point
        let zero_quant = params.quantize(0.0);
        assert!(zero_quant >= -8 && zero_quant <= 7);
        
        // Test round-trip for a few values
        for &original in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let quantized = params.quantize(original);
            let dequantized = params.dequantize(quantized);
            
            println!("Original: {:.2}, Quantized: {}, Dequantized: {:.2}, Error: {:.4}", 
                     original, quantized, dequantized, (original - dequantized).abs());
            
            // Error should be reasonable (less than scale)
            assert!((original - dequantized).abs() < params.scale * 2.0);
        }
    }

    #[test]
    fn test_int4_extreme_values() {
        // Test with extreme value ranges
        let params = QuantParamsInt4::from_range(-100.0, 100.0);
        
        let q_neg = params.quantize(-100.0);
        let q_pos = params.quantize(100.0);
        
        assert_eq!(q_neg, -8);  // Should clamp to -8
        assert_eq!(q_pos, 7);   // Should clamp to 7
    }

    #[test]
    fn test_int4_vs_int8_error() {
        // Compare error between INT4 and INT8
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        
        // INT8 quantization
        let params_int8 = QuantParams::from_range(-1.0, 1.0);
        let error_int8: f32 = data.iter()
            .map(|&v| {
                let q = params_int8.quantize(v);
                let dq = params_int8.dequantize(q);
                (v - dq).powi(2)
            })
            .sum::<f32>() / data.len() as f32;
        
        // INT4 quantization
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
        
        // INT4 should have higher error (fewer values)
        assert!(error_int4 > error_int8);
        
        // But not astronomically higher (should be reasonable)
        // INT4 has 16x fewer values, so error can be ~256-500x higher (squared effect)
        assert!(error_int4 < error_int8 * 500.0, 
                "INT4 error ({:.8}) is too high compared to INT8 ({:.8})", 
                error_int4, error_int8);

        // Verify INT4 error is still usable
        assert!(error_int4.is_finite());
        assert!(error_int4 < 0.01);
        
    }

    #[test]
    fn test_int4_range() {
        let params = QuantParamsInt4::from_range(-1.0, 1.0);
        
        // Test that we use full range
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
        // Verify we're using all 16 values
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
        assert_eq!(quantized.size_bytes(), 5); // Unpacked: 5 bytes
        assert_eq!(quantized.packed_size_bytes(), 3); // Packed: 3 bytes (will implement later)
        
        // Verify all values are in INT4 range
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

        // Check round-trip error
        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            println!("  {:.2} -> {:.2}, error: {:.4}", orig, deq, error);
            assert!(error < 0.15, "Error too large: {}", error);
        }
    }

    #[test]
    fn test_int4_per_channel() {
        // Create 2-channel data with different ranges
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
        
        // Per-channel quantization
        let quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape, 0).unwrap();
        
        assert!(quantized.per_channel);
        assert!(quantized.channel_params.is_some());
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 2);
        
        // Calculate error
        let error = quantized.quantization_error(&data);
        println!("INT4 per-channel MSE: {:.8}", error);
        
        // Should be reasonable
        assert!(error < 1.0, "Error too high: {}", error);
    }

    #[test]
    fn test_int4_vs_int8_compression() {
        let data: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) * 2.0 - 1.0).collect();
        let shape = vec![1000];
        
        // INT8 quantization
        let int8_quantized = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let int8_size = int8_quantized.size_bytes();
        let int8_error = int8_quantized.quantization_error(&data);
        
        // INT4 quantization
        let int4_quantized = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        let int4_size = int4_quantized.size_bytes();
        let int4_packed_size = int4_quantized.packed_size_bytes();
        let int4_error = int4_quantized.quantization_error(&data);
        
        println!("INT8: {} bytes, MSE: {:.8}", int8_size, int8_error);
        println!("INT4 (unpacked): {} bytes, MSE: {:.8}", int4_size, int4_error);
        println!("INT4 (packed): {} bytes, MSE: {:.8}", int4_packed_size, int4_error);
        
        // INT4 unpacked should be same size as INT8
        assert_eq!(int4_size, int8_size);
        
        // INT4 packed should be half the size
        assert!(int4_packed_size <= int8_size / 2 + 1);
        
        // INT4 should have higher error (as expected)
        assert!(int4_error > int8_error);
        
        // But error should still be reasonable
        assert!(int4_error < 0.01, "INT4 error too high: {}", int4_error);
    }

    #[test]
    fn test_int4_large_tensor() {
        // Test with realistic Conv layer size
        let size = 64 * 3 * 3 * 3; // 64 filters, 3x3x3 kernels
        let data: Vec<f32> = (0..size).map(|i| {
            ((i as f32 / size as f32) * 2.0 - 1.0) * 0.5
        }).collect();
        
        let shape = vec![64, 3, 3, 3];
        
        // Per-channel quantization
        let quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape, 0).unwrap();
        
        assert_eq!(quantized.data.len(), size);
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 64);
        
        let error = quantized.quantization_error(&data);
        println!("Large tensor INT4 error: {:.8}", error);
        
        assert!(error < 0.01, "Error too high for large tensor: {}", error);
    }

    #[test]
    fn test_int4_extreme_ranges() {
        // Test with very different value ranges
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
            
            // All values should be in valid range
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
        // Test even-length vector
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
        // Test odd-length vector (requires padding)
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
        // Test all possible INT4 values
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        
        println!("\nAll INT4 values:");
        println!("  Original: {:?}", values);
        println!("  Packed:   {} bytes", packed.len());
        println!("  Unpacked: {:?}", unpacked);
        
        assert_eq!(values, unpacked);
        assert_eq!(packed.len(), 8); // 16 values / 2 = 8 bytes
    }

    #[test]
    fn test_int4_pack_large_vector() {
        // Test with realistic size
        let values: Vec<i8> = (0..1000).map(|i| ((i % 16) - 8) as i8).collect();
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed, values.len());
        
        assert_eq!(values, unpacked);
        assert_eq!(packed.len(), 500); // 1000 values / 2 = 500 bytes
        
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
        
        // Unpacked size (1 byte per value)
        let unpacked_size = values.len() * std::mem::size_of::<i8>();
        
        // Packed size (0.5 bytes per value)
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
        
        // Pack the data
        quantized.pack();
        
        println!("\nAfter packing:");
        println!("  Packed size: {} bytes", quantized.size_bytes());
        println!("  Is packed: {}", quantized.is_packed());
        println!("  Compression: {}x", quantized.unpacked_size_bytes() / quantized.size_bytes());
        
        assert!(quantized.is_packed());
        assert_eq!(quantized.size_bytes(), 500); // 2x compression
        
        // Verify dequantization still works
        let dequantized = quantized.to_f32();
        assert_eq!(dequantized.len(), 1000);
        
        // Verify error is same whether packed or not
        let error = quantized.quantization_error(&data);
        println!("  MSE after packing: {:.8}", error);
        assert!(error < 0.01);
    }

    #[test]
    fn test_int4_packed_vs_unpacked_error() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 2.0 - 1.0).collect();
        let shape = vec![100];
        
        // Create unpacked version
        let unpacked = QuantizedTensorInt4::from_f32(&data, shape.clone()).unwrap();
        let error_unpacked = unpacked.quantization_error(&data);
        
        // Create packed version
        let mut packed = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        packed.pack();
        let error_packed = packed.quantization_error(&data);
        
        println!("Unpacked error: {:.8}", error_unpacked);
        println!("Packed error:   {:.8}", error_packed);
        
        // Error should be identical (packing is lossless)
        assert!((error_unpacked - error_packed).abs() < 1e-6);
    }

    #[test]
    fn test_int4_per_channel_packing() {
        // Create 2-channel data
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
        
        // Pack
        quantized.pack();
        
        let error_after = quantized.quantization_error(&data);
        println!("Error after packing:  {:.8}", error_after);
        println!("Size: {} bytes (packed from {} bytes)", 
                quantized.size_bytes(), 
                quantized.unpacked_size_bytes());
        
        // Error should be same
        assert!((error_before - error_after).abs() < 1e-6);
        
        // Size should be half
        assert_eq!(quantized.size_bytes(), 500);
    }

    #[test]
    fn test_int4_compression_comparison() {
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|i| {
            ((i as f32 / size as f32) * 2.0 - 1.0) * 0.5
        }).collect();
        let shape = vec![size];
        
        // Original FP32 size
        let fp32_size = size * std::mem::size_of::<f32>();
        
        // INT8 quantization
        let int8 = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
        let int8_size = int8.size_bytes();
        
        // INT4 unpacked
        let int4_unpacked = QuantizedTensorInt4::from_f32(&data, shape.clone()).unwrap();
        let int4_unpacked_size = int4_unpacked.size_bytes();
        
        // INT4 packed
        let mut int4_packed = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
        int4_packed.pack();
        let int4_packed_size = int4_packed.size_bytes();
        
        println!("\nCompression Comparison:");
        println!("  FP32:          {} bytes", fp32_size);
        println!("  INT8:          {} bytes ({:.1}x)", int8_size, fp32_size as f32 / int8_size as f32);
        println!("  INT4 unpacked: {} bytes ({:.1}x)", int4_unpacked_size, fp32_size as f32 / int4_unpacked_size as f32);
        println!("  INT4 packed:   {} bytes ({:.1}x)", int4_packed_size, fp32_size as f32 / int4_packed_size as f32);
        
        // Verify compression ratios
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
        
        // Try to load a real model
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
        
        // Extract weights
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
        
        // Test a few representative layers
        let test_weights: Vec<_> = weights.iter()
            .filter(|w| w.data.len() > 1000) // Only test larger layers
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
            
            // Original size
            let fp32_size = weight.data.len() * 4;
            
            // INT8 quantization
            let int8_result = QuantizedTensor::from_f32(&weight.data, weight.shape.clone());
            let (int8_size, int8_error) = if let Ok(q) = int8_result {
                (q.size_bytes(), q.quantization_error(&weight.data))
            } else {
                println!("    INT8 failed!");
                continue;
            };
            
            // INT4 per-tensor
            let int4_result = QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone());
            let (int4_unpacked_size, int4_error) = if let Ok(q) = int4_result {
                (q.size_bytes(), q.quantization_error(&weight.data))
            } else {
                println!("    INT4 failed!");
                continue;
            };
            
            // INT4 packed
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
            
            // Verify packing is lossless
            assert_eq!(int4_error, int4_packed_error, "Packing changed error!");
            
            // Verify compression ratios
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
            .filter(|w| w.shape.len() >= 2 && w.shape[0] > 1) // Multi-channel
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
                
                // INT4 per-tensor
                let per_tensor = QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone()).unwrap();
                let per_tensor_error = per_tensor.quantization_error(&weight.data);
                
                // INT4 per-channel
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
                    
                    // Per-channel should generally be better or equal
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