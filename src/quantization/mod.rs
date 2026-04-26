//! Core quantization logic for INT8 and INT4.
//!
//! Provides tensor-level quantization (per-tensor and per-channel),
//! INT4 bit-packing, and the high-level [`Quantizer`] that combines
//! a [`QuantConfig`] with optional calibration statistics.

use crate::errors::{QuantizeError, Result};

/// Configuration for a quantization pass.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Bit width: `4` for INT4 or `8` for INT8.
    pub bits: u8,
    /// When `true`, compute separate scale/zero-point per output channel (axis 0).
    pub per_channel: bool,
    /// When `true`, force `zero_point == 0` (symmetric quantization) — required
    /// by most ONNX Runtime / TensorRT INT8 matmul kernels for per-channel
    /// weight quantization.  Defaults to `false` (asymmetric).
    pub symmetric: bool,
    /// Optional calibration method used for range optimization.
    pub calibration_method: Option<crate::calibration::methods::CalibrationMethod>,
    /// Layer names to skip entirely (exact match against the initializer name).
    pub excluded_layers: Vec<String>,
    /// Per-layer bit-width overrides.  Key = initializer name, value = 4 or 8.
    pub layer_bits: std::collections::HashMap<String, u8>,
    /// Minimum number of elements a tensor must have to be quantized.
    /// Tensors with fewer elements are left in FP32.  Defaults to 0 (no minimum).
    pub min_elements: usize,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            per_channel: false,
            symmetric: false,
            calibration_method: None,
            excluded_layers: Vec::new(),
            layer_bits: std::collections::HashMap::new(),
            min_elements: 0,
        }
    }
}

impl QuantConfig {
    /// Create a default INT8 per-tensor configuration.
    pub fn int8() -> Self {
        Self::default()
    }

    /// Enable or disable per-channel quantization.
    pub fn with_per_channel(mut self, enabled: bool) -> Self {
        self.per_channel = enabled;
        self
    }

    /// Enable or disable symmetric quantization (`zero_point == 0`).
    pub fn with_symmetric(mut self, enabled: bool) -> Self {
        self.symmetric = enabled;
        self
    }

    /// Set the calibration method for range optimization.
    pub fn with_calibration(
        mut self,
        method: crate::calibration::methods::CalibrationMethod,
    ) -> Self {
        self.calibration_method = Some(method);
        self
    }

    /// Return `true` if the layer should be quantized.
    ///
    /// A layer is skipped when:
    /// - its name appears in [`excluded_layers`], or
    /// - `num_elements` is below [`min_elements`] (and `min_elements > 0`).
    pub fn should_quantize(&self, name: &str, num_elements: usize) -> bool {
        if self.excluded_layers.iter().any(|e| e == name) {
            return false;
        }
        if self.min_elements > 0 && num_elements < self.min_elements {
            return false;
        }
        true
    }

    /// Return the effective bit width for a layer.
    ///
    /// If the layer name has an entry in [`layer_bits`], that value is used;
    /// otherwise the global [`bits`] is returned.
    pub fn bits_for_layer(&self, name: &str) -> u8 {
        self.layer_bits.get(name).copied().unwrap_or(self.bits)
    }
}

// ---------------------------------------------------------------------------
// QuantRange trait and marker types
// ---------------------------------------------------------------------------

/// Marker trait that supplies the clamp constants for a quantization bit-width.
pub trait QuantRange: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Minimum quantized value (inclusive).
    const QMIN: f32;
    /// Maximum quantized value (inclusive).
    const QMAX: f32;
    /// Bit width (4 or 8).
    const BITS: u8;
}

/// Marker for INT8 quantization (`-128 … 127`).
#[derive(Debug, Clone)]
pub struct Int8Range;
impl QuantRange for Int8Range {
    const QMIN: f32 = -128.0;
    const QMAX: f32 = 127.0;
    const BITS: u8 = 8;
}

/// Marker for INT4 quantization (`-8 … 7`).
#[derive(Debug, Clone)]
pub struct Int4Range;
impl QuantRange for Int4Range {
    const QMIN: f32 = -8.0;
    const QMAX: f32 = 7.0;
    const BITS: u8 = 4;
}

// ---------------------------------------------------------------------------
// QuantParamsGeneric<R>
// ---------------------------------------------------------------------------

/// Affine quantization parameters (scale and zero-point), generic over bit-width.
///
/// - INT8: `q = clamp(round(x / scale) + zero_point, -128, 127)`
/// - INT4: `q = clamp(round(x / scale) + zero_point, -8, 7)`
/// - Dequantization: `x = (q - zero_point) * scale`
#[derive(Debug, Clone)]
pub struct QuantParamsGeneric<R: QuantRange> {
    scale: f32,
    zero_point: i8,
    _marker: std::marker::PhantomData<R>,
}

/// INT8 affine quantization parameters — `clamp(-128, 127)`.
pub type QuantParams = QuantParamsGeneric<Int8Range>;
/// INT4 affine quantization parameters — `clamp(-8, 7)`.
pub type QuantParamsInt4 = QuantParamsGeneric<Int4Range>;

impl<R: QuantRange> QuantParamsGeneric<R> {
    /// Quantization scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }
    /// Quantization zero point.
    pub fn zero_point(&self) -> i8 {
        self.zero_point
    }

    /// Compute asymmetric quantization parameters from a floating-point range.
    ///
    /// The resulting zero-point is in `[QMIN, QMAX]` and the full integer
    /// range `[QMIN, QMAX]` is used to represent the input.  For ONNX Runtime /
    /// TensorRT per-channel weight quantization, prefer
    /// [`from_range_symmetric`](Self::from_range_symmetric) instead — those
    /// kernels assume `zero_point == 0`.
    pub fn from_range(min: f32, max: f32) -> Self {
        let min = min.min(0.0);
        let max = max.max(0.0);

        // Handle constant-value tensors: when min ≈ max the data is (near-)constant.
        // Use unit scale centred on zero so that the constant dequantizes accurately.
        let (min, max) = if (max - min).abs() < 1e-8 {
            let abs = min.abs().max(max.abs()).max(1e-8);
            (-abs, abs)
        } else {
            (min, max)
        };

        let scale = (max - min) / (R::QMAX - R::QMIN);
        let scale = scale.max(1e-8);

        let initial_zero_point = R::QMIN - min / scale;
        // Guard against NaN — if min/scale produced NaN (degenerate input),
        // fall back to 0 to avoid undefined behaviour on the `as i8` cast.
        let zero_point = if initial_zero_point.is_finite() {
            initial_zero_point.round().clamp(R::QMIN, R::QMAX) as i8
        } else {
            0i8
        };

        QuantParamsGeneric {
            scale,
            zero_point,
            _marker: std::marker::PhantomData,
        }
    }

    /// Compute symmetric quantization parameters: `zero_point = 0`, `scale`
    /// chosen so that the positive half of the quantized range covers
    /// `max(|min|, |max|)`.
    ///
    /// This is the convention ONNX Runtime, TensorRT, and most accelerated
    /// INT8 matmul kernels expect for per-channel weight quantization.
    /// For INT8, the effective representable range is `[-127*scale, 127*scale]`;
    /// any input value mapped to `-128` after rounding is clamped back into
    /// range.  Zero always dequantizes to exactly 0.0.
    pub fn from_range_symmetric(min: f32, max: f32) -> Self {
        let abs_max = min.abs().max(max.abs()).max(1e-8);
        // R::QMAX is the positive maximum (127 for INT8, 7 for INT4).  Dividing
        // by QMAX — not by (QMAX - QMIN) — is what makes the result symmetric
        // and keeps the zero-point at 0.
        let scale = (abs_max / R::QMAX).max(1e-8);
        QuantParamsGeneric {
            scale,
            zero_point: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Quantize a single float to the target integer type.
    pub fn quantize(&self, value: f32) -> i8 {
        if !value.is_finite() {
            return self.zero_point;
        }
        let quantized = (value / self.scale).round() + (self.zero_point as f32);
        quantized.clamp(R::QMIN, R::QMAX) as i8
    }

    /// Dequantize a single integer value back to float.
    pub fn dequantize(&self, value: i8) -> f32 {
        ((value as i32) - (self.zero_point as i32)) as f32 * self.scale
    }
}

// ---------------------------------------------------------------------------
// QuantizedTensorGeneric<R>
// ---------------------------------------------------------------------------

/// Generic quantized tensor, parameterized by bit-width marker.
///
/// For INT4 tensors, call [`QuantizedTensorGeneric::pack`] to compress two
/// values per byte for 2× storage savings.
#[derive(Debug, Clone)]
pub struct QuantizedTensorGeneric<R: QuantRange> {
    pub(crate) data: Vec<i8>,
    /// Bit-packed storage — always `None` for INT8, set by `.pack()` for INT4.
    pub(crate) packed_data: Option<Vec<u8>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) params: QuantParamsGeneric<R>,
    pub(crate) per_channel: bool,
    pub(crate) channel_params: Option<Vec<QuantParamsGeneric<R>>>,
}

/// An INT8 quantized tensor with optional per-channel parameters.
pub type QuantizedTensor = QuantizedTensorGeneric<Int8Range>;

/// An INT4 quantized tensor with optional per-channel parameters and bit packing.
///
/// Values are stored in the range `[-8, 7]`. Call [`pack`](QuantizedTensorInt4::pack) to
/// compress two values into one byte for 2× storage savings.
pub type QuantizedTensorInt4 = QuantizedTensorGeneric<Int4Range>;

// ---------------------------------------------------------------------------
// Shared impl for all bit-widths
// ---------------------------------------------------------------------------

impl<R: QuantRange> QuantizedTensorGeneric<R> {
    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Per-tensor quantization parameters (channel-0 if per-channel).
    pub fn params(&self) -> &QuantParamsGeneric<R> {
        &self.params
    }
    /// Whether per-channel quantization was used.
    pub fn is_per_channel(&self) -> bool {
        self.per_channel
    }

    /// Quantize FP32 data, computing the range from the data (asymmetric).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::InvalidTensor`] if `data` is empty or shape mismatches.
    pub fn from_f32(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        Self::from_f32_with_mode(data, shape, false)
    }

    /// Quantize FP32 data using symmetric quantization (`zero_point == 0`).
    ///
    /// Required by most ONNX Runtime / TensorRT INT8 kernels for per-channel
    /// weight quantization.  See [`QuantParamsGeneric::from_range_symmetric`].
    pub fn from_f32_symmetric(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        Self::from_f32_with_mode(data, shape, true)
    }

    fn from_f32_with_mode(data: &[f32], shape: Vec<usize>, symmetric: bool) -> Result<Self> {
        if data.is_empty() {
            return Err(QuantizeError::InvalidTensor {
                reason: "Cannot quantize empty tensor".into(),
            });
        }

        let expected_len: usize = shape.iter().product();
        if expected_len != data.len() {
            return Err(QuantizeError::InvalidTensor {
                reason: format!(
                    "Shape {:?} expects {} elements but got {}",
                    shape,
                    expected_len,
                    data.len()
                ),
            });
        }

        let min = data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min);
        let max = data
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);

        if !min.is_finite() || !max.is_finite() {
            return Err(QuantizeError::InvalidTensor {
                reason: "Tensor contains only non-finite values (NaN/Inf)".into(),
            });
        }

        let params = if symmetric {
            QuantParamsGeneric::<R>::from_range_symmetric(min, max)
        } else {
            QuantParamsGeneric::<R>::from_range(min, max)
        };

        let quantized_data: Vec<i8> = data.iter().map(|&v| params.quantize(v)).collect();

        Ok(QuantizedTensorGeneric {
            data: quantized_data,
            packed_data: None,
            shape,
            params,
            per_channel: false,
            channel_params: None,
        })
    }

    /// Quantize FP32 data using an explicit range (for calibration; asymmetric).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::InvalidTensor`] if `data` is empty or shape mismatches.
    pub fn from_f32_with_range(
        data: &[f32],
        shape: Vec<usize>,
        min: f32,
        max: f32,
    ) -> Result<Self> {
        Self::from_f32_with_range_and_mode(data, shape, min, max, false)
    }

    /// Same as [`from_f32_with_range`](Self::from_f32_with_range) but produces
    /// symmetric parameters (`zero_point == 0`).
    pub fn from_f32_with_range_symmetric(
        data: &[f32],
        shape: Vec<usize>,
        min: f32,
        max: f32,
    ) -> Result<Self> {
        Self::from_f32_with_range_and_mode(data, shape, min, max, true)
    }

    fn from_f32_with_range_and_mode(
        data: &[f32],
        shape: Vec<usize>,
        min: f32,
        max: f32,
        symmetric: bool,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(QuantizeError::InvalidTensor {
                reason: "Cannot quantize empty tensor".into(),
            });
        }

        let expected_len: usize = shape.iter().product();
        if expected_len != data.len() {
            return Err(QuantizeError::InvalidTensor {
                reason: format!(
                    "Shape {:?} expects {} elements but got {}",
                    shape,
                    expected_len,
                    data.len()
                ),
            });
        }

        let params = if symmetric {
            QuantParamsGeneric::<R>::from_range_symmetric(min, max)
        } else {
            QuantParamsGeneric::<R>::from_range(min, max)
        };

        let quantized_data: Vec<i8> = data.iter().map(|&v| params.quantize(v)).collect();

        Ok(QuantizedTensorGeneric {
            data: quantized_data,
            packed_data: None,
            shape,
            params,
            per_channel: false,
            channel_params: None,
        })
    }

    /// Quantize FP32 data with per-channel ranges (axis 0 only, asymmetric).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::InvalidTensor`] if `data` is empty, shape
    /// mismatches, or the tensor is scalar.
    pub fn from_f32_per_channel(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        Self::from_f32_per_channel_with_mode(data, shape, false)
    }

    /// Same as [`from_f32_per_channel`](Self::from_f32_per_channel) but emits
    /// symmetric parameters (`zero_point == 0` for every channel).  Required
    /// by most INT8 per-channel matmul kernels.
    pub fn from_f32_per_channel_symmetric(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        Self::from_f32_per_channel_with_mode(data, shape, true)
    }

    fn from_f32_per_channel_with_mode(
        data: &[f32],
        shape: Vec<usize>,
        symmetric: bool,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(QuantizeError::InvalidTensor {
                reason: "Cannot quantize empty tensor".into(),
            });
        }

        if shape.is_empty() {
            return Err(QuantizeError::InvalidTensor {
                reason: "Cannot do per-channel quantization on scalar".into(),
            });
        }

        let expected_len: usize = shape.iter().product();
        if expected_len != data.len() {
            return Err(QuantizeError::InvalidTensor {
                reason: format!(
                    "Shape {:?} expects {} elements but got {}",
                    shape,
                    expected_len,
                    data.len()
                ),
            });
        }

        let num_channels = shape[0];
        if num_channels == 0 {
            return Err(QuantizeError::InvalidTensor {
                reason: "Number of channels is 0".into(),
            });
        }
        if !data.len().is_multiple_of(num_channels) {
            return Err(QuantizeError::InvalidTensor {
                reason: format!(
                    "Data length {} not evenly divisible by {} channels",
                    data.len(),
                    num_channels
                ),
            });
        }
        let elements_per_channel = data.len() / num_channels;

        let mut channel_params = Vec::with_capacity(num_channels);
        let mut quantized_data = Vec::with_capacity(data.len());

        // Walk the data channel-by-channel with a borrowed slice — no Vec alloc
        // per channel.  For typical Conv weights this avoids hundreds of small
        // allocations that used to dominate the per-channel hot path.
        for (channel_idx, channel_slice) in data.chunks_exact(elements_per_channel).enumerate() {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in channel_slice {
                if v.is_finite() {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
            }

            if !min.is_finite() || !max.is_finite() {
                return Err(QuantizeError::InvalidTensor {
                    reason: format!(
                        "Channel {} contains only non-finite values (NaN/Inf)",
                        channel_idx
                    ),
                });
            }

            let params = if symmetric {
                QuantParamsGeneric::<R>::from_range_symmetric(min, max)
            } else {
                QuantParamsGeneric::<R>::from_range(min, max)
            };

            quantized_data.extend(channel_slice.iter().map(|&v| params.quantize(v)));
            channel_params.push(params);
        }

        // Use first channel params as "representative" for backward compatibility
        let params = channel_params[0].clone();

        Ok(QuantizedTensorGeneric {
            data: quantized_data,
            packed_data: None,
            shape,
            params,
            per_channel: true,
            channel_params: Some(channel_params),
        })
    }

    /// Dequantize all values back to FP32.
    pub fn to_f32(&self) -> Vec<f32> {
        // Borrow data directly when unpacked; allocate only for the packed INT4 path.
        let data_owned;
        let data: &[i8] = if let Some(ref packed) = self.packed_data {
            data_owned = unpack_int4(packed, self.data.len());
            &data_owned
        } else {
            &self.data
        };

        if self.per_channel {
            if let Some(ref channel_params) = self.channel_params {
                if channel_params.is_empty() {
                    return data.iter().map(|&v| self.params.dequantize(v)).collect();
                }
                // Chunk the data by elements_per_channel and zip with channel params.
                // This replaces a per-element division (`i / elements_per_channel`)
                // with outer-loop iteration over contiguous channel slices.
                let elements_per_channel = data.len() / channel_params.len();
                let mut out = Vec::with_capacity(data.len());
                if elements_per_channel == 0 {
                    // Degenerate: fewer values than channels.  Fall back to the
                    // representative params so we don't panic on the zip below.
                    return data.iter().map(|&v| self.params.dequantize(v)).collect();
                }
                for (chunk, params) in data.chunks(elements_per_channel).zip(channel_params.iter())
                {
                    out.extend(chunk.iter().map(|&v| params.dequantize(v)));
                }
                out
            } else {
                data.iter().map(|&v| self.params.dequantize(v)).collect()
            }
        } else {
            data.iter().map(|&v| self.params.dequantize(v)).collect()
        }
    }

    /// Size of the quantized data in bytes (packed if available, unpacked otherwise).
    pub fn size_bytes(&self) -> usize {
        if let Some(ref packed) = self.packed_data {
            packed.len()
        } else {
            self.data.len() * std::mem::size_of::<i8>()
        }
    }

    /// Mean squared error between the original data and the dequantized values.
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

        sum / original.len() as f32
    }
}

// ---------------------------------------------------------------------------
// INT4-specific methods
// ---------------------------------------------------------------------------

impl QuantizedTensorGeneric<Int4Range> {
    /// Pack two INT4 values per byte for 2× compression.
    pub fn pack(&mut self) {
        self.packed_data = Some(pack_int4(&self.data));
    }

    /// Return unpacked i8 data, decompressing from packed storage if needed.
    pub fn ensure_unpacked(&self) -> Vec<i8> {
        if let Some(ref packed) = self.packed_data {
            unpack_int4(packed, self.data.len())
        } else {
            self.data.clone()
        }
    }

    /// Whether the data is currently bit-packed.
    pub fn is_packed(&self) -> bool {
        self.packed_data.is_some()
    }

    /// Size that the packed representation would occupy (or already occupies).
    pub fn packed_size_bytes(&self) -> usize {
        if let Some(ref packed) = self.packed_data {
            packed.len()
        } else {
            self.data.len().div_ceil(2)
        }
    }

    /// Size of the unpacked representation in bytes.
    pub fn unpacked_size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i8>()
    }
}

// ---------------------------------------------------------------------------
// INT4 bit-packing helpers
// ---------------------------------------------------------------------------

fn pack_int4_pair(val1: i8, val2: i8) -> u8 {
    debug_assert!((-8..=7).contains(&val1), "val1 out of INT4 range: {}", val1);
    debug_assert!((-8..=7).contains(&val2), "val2 out of INT4 range: {}", val2);

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

/// Pack a slice of INT4 values (two per byte, high nibble first).
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));

    for chunk in values.chunks(2) {
        let val1 = chunk[0];
        let val2 = if chunk.len() > 1 { chunk[1] } else { 0 };

        packed.push(pack_int4_pair(val1, val2));
    }

    packed
}

/// Unpack INT4 values from packed bytes, returning exactly `num_values` i8s.
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

// ---------------------------------------------------------------------------
// QuantizedTensorType
// ---------------------------------------------------------------------------

/// Type-erased wrapper over [`QuantizedTensor`] (INT8) and [`QuantizedTensorInt4`] (INT4).
#[derive(Debug, Clone)]
pub enum QuantizedTensorType {
    Int8(QuantizedTensor),
    Int4(QuantizedTensorInt4),
}

impl QuantizedTensorType {
    /// Dequantize all values back to FP32.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            QuantizedTensorType::Int8(t) => t.to_f32(),
            QuantizedTensorType::Int4(t) => t.to_f32(),
        }
    }

    /// Size of the quantized data in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            QuantizedTensorType::Int8(t) => t.size_bytes(),
            QuantizedTensorType::Int4(t) => t.size_bytes(),
        }
    }

    #[must_use]
    pub fn quantization_error(&self, original: &[f32]) -> f32 {
        match self {
            QuantizedTensorType::Int8(t) => t.quantization_error(original),
            QuantizedTensorType::Int4(t) => t.quantization_error(original),
        }
    }

    #[must_use]
    pub fn data(&self) -> Vec<i8> {
        match self {
            QuantizedTensorType::Int8(t) => t.data.clone(),
            QuantizedTensorType::Int4(t) => t.ensure_unpacked(),
        }
    }

    /// Per-tensor scale and zero-point.
    pub fn get_scale_zero_point(&self) -> (f32, i8) {
        match self {
            QuantizedTensorType::Int8(t) => (t.params.scale, t.params.zero_point),
            QuantizedTensorType::Int4(t) => (t.params.scale, t.params.zero_point),
        }
    }

    /// Return all per-channel scales and zero-points.
    ///
    /// For per-tensor quantization, returns single-element vectors.
    /// For per-channel, returns one entry per channel.
    pub fn get_all_scales_zero_points(&self) -> (Vec<f32>, Vec<i8>) {
        match self {
            QuantizedTensorType::Int8(t) => {
                if let Some(ref cp) = t.channel_params {
                    (
                        cp.iter().map(|p| p.scale).collect(),
                        cp.iter().map(|p| p.zero_point).collect(),
                    )
                } else {
                    (vec![t.params.scale], vec![t.params.zero_point])
                }
            }
            QuantizedTensorType::Int4(t) => {
                if let Some(ref cp) = t.channel_params {
                    (
                        cp.iter().map(|p| p.scale).collect(),
                        cp.iter().map(|p| p.zero_point).collect(),
                    )
                } else {
                    (vec![t.params.scale], vec![t.params.zero_point])
                }
            }
        }
    }

    /// Whether per-channel quantization was used.
    pub fn is_per_channel(&self) -> bool {
        match self {
            QuantizedTensorType::Int8(t) => t.per_channel,
            QuantizedTensorType::Int4(t) => t.per_channel,
        }
    }

    #[must_use]
    pub fn bits(&self) -> u8 {
        match self {
            QuantizedTensorType::Int8(_) => 8,
            QuantizedTensorType::Int4(_) => 4,
        }
    }

    /// `true` if this is an INT8 tensor.
    pub fn is_int8(&self) -> bool {
        matches!(self, QuantizedTensorType::Int8(_))
    }

    /// `true` if this is an INT4 tensor.
    pub fn is_int4(&self) -> bool {
        matches!(self, QuantizedTensorType::Int4(_))
    }

    /// Borrow quantized data without cloning.
    ///
    /// Returns `None` for packed INT4 tensors (must use `data()` which unpacks).
    pub fn data_ref(&self) -> Option<&[i8]> {
        match self {
            QuantizedTensorType::Int8(t) => Some(&t.data),
            QuantizedTensorType::Int4(t) => {
                if t.packed_data.is_some() {
                    None // packed: caller must use data() to unpack
                } else {
                    Some(&t.data)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantizer
// ---------------------------------------------------------------------------

/// High-level quantizer that combines configuration with optional calibration.
pub struct Quantizer {
    config: QuantConfig,
    calibration_stats:
        Option<std::collections::HashMap<String, crate::calibration::stats::ActivationStats>>,
}

impl std::fmt::Debug for Quantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats_count = self.calibration_stats.as_ref().map(|m| m.len());
        f.debug_struct("Quantizer")
            .field("config", &self.config)
            .field("calibration_stats_count", &stats_count)
            .finish()
    }
}

impl Quantizer {
    /// Create a quantizer with the given configuration (no calibration).
    pub fn new(config: QuantConfig) -> Self {
        Self {
            config,
            calibration_stats: None,
        }
    }

    /// Create a quantizer with configuration and pre-collected activation statistics.
    pub fn with_calibration(
        config: QuantConfig,
        stats: std::collections::HashMap<String, crate::calibration::stats::ActivationStats>,
    ) -> Self {
        Self {
            config,
            calibration_stats: Some(stats),
        }
    }

    /// Quantize a tensor with optional calibration.
    pub fn quantize_tensor_with_name(
        &self,
        name: &str,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<QuantizedTensorType> {
        let (min, max) = if let Some(ref stats_map) = self.calibration_stats {
            if let Some(stats) = stats_map.get(name) {
                if let Some(method) = self.config.calibration_method {
                    // Compute the range directly from the histogram — deterministic,
                    // no sample regeneration, no RNG.
                    use crate::calibration::stats::calculate_optimal_range_from_stats;
                    calculate_optimal_range_from_stats(stats, method)
                } else {
                    (stats.min(), stats.max())
                }
            } else {
                finite_min_max(data, name)?
            }
        } else {
            finite_min_max(data, name)?
        };

        self.quantize_with_range(data, shape, min, max)
    }

    /// Quantize a tensor using the configured bit width and per-channel setting.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::InvalidTensor`] or [`QuantizeError::UnsupportedConfig`].
    pub fn quantize_tensor(&self, data: &[f32], shape: Vec<usize>) -> Result<QuantizedTensorType> {
        self.build_tensor_with_optional_range(data, shape, None)
    }

    /// Quantize with specific range (for calibration).
    ///
    /// When `per_channel` is enabled, the provided `min`/`max` are ignored
    /// because per-channel quantization computes separate ranges from the
    /// weight data for each channel.  The calibration range (derived from
    /// activation statistics) applies to per-tensor mode only.
    fn quantize_with_range(
        &self,
        data: &[f32],
        shape: Vec<usize>,
        min: f32,
        max: f32,
    ) -> Result<QuantizedTensorType> {
        self.build_tensor_with_optional_range(data, shape, Some((min, max)))
    }

    /// Shared core: build a [`QuantizedTensorType`] for any bit-width and range mode.
    fn build_tensor_with_optional_range(
        &self,
        data: &[f32],
        shape: Vec<usize>,
        range: Option<(f32, f32)>,
    ) -> Result<QuantizedTensorType> {
        let pc = self.config.per_channel && shape.len() >= 2;
        let sym = self.config.symmetric;
        match self.config.bits {
            8 => {
                let t = match (pc, range, sym) {
                    (true, _, true) => {
                        QuantizedTensor::from_f32_per_channel_symmetric(data, shape)?
                    }
                    (true, _, false) => QuantizedTensor::from_f32_per_channel(data, shape)?,
                    (false, Some((min, max)), true) => {
                        QuantizedTensor::from_f32_with_range_symmetric(data, shape, min, max)?
                    }
                    (false, Some((min, max)), false) => {
                        QuantizedTensor::from_f32_with_range(data, shape, min, max)?
                    }
                    (false, None, true) => QuantizedTensor::from_f32_symmetric(data, shape)?,
                    (false, None, false) => QuantizedTensor::from_f32(data, shape)?,
                };
                Ok(QuantizedTensorType::Int8(t))
            }
            4 => {
                let mut t = match (pc, range, sym) {
                    (true, _, true) => {
                        QuantizedTensorInt4::from_f32_per_channel_symmetric(data, shape)?
                    }
                    (true, _, false) => QuantizedTensorInt4::from_f32_per_channel(data, shape)?,
                    (false, Some((min, max)), true) => {
                        QuantizedTensorInt4::from_f32_with_range_symmetric(data, shape, min, max)?
                    }
                    (false, Some((min, max)), false) => {
                        QuantizedTensorInt4::from_f32_with_range(data, shape, min, max)?
                    }
                    (false, None, true) => QuantizedTensorInt4::from_f32_symmetric(data, shape)?,
                    (false, None, false) => QuantizedTensorInt4::from_f32(data, shape)?,
                };
                t.pack();
                Ok(QuantizedTensorType::Int4(t))
            }
            b => Err(QuantizeError::UnsupportedConfig {
                reason: format!("bits must be 4 or 8, got {b}"),
            }),
        }
    }

    /// Quantize every weight in `model` that passes
    /// [`QuantConfig::should_quantize`].  Honours per-layer bit-width overrides.
    ///
    /// When this quantizer was built with calibration, activation-based
    /// range optimization is used for the default bit-width; layers whose
    /// bit-width is overridden fall back to weight-only quantization
    /// (the calibration stats are keyed by the default configuration).
    ///
    /// Skipped weights do not appear in the returned vector.
    pub fn quantize_model(
        &self,
        model: &crate::onnx_utils::OnnxModel,
    ) -> Result<Vec<QuantizedWeightOutput>> {
        use rayon::prelude::*;

        let weights = model.extract_weights();
        let to_quantize: Vec<_> = weights
            .iter()
            .filter(|w| self.config.should_quantize(&w.name, w.num_elements()))
            .collect();

        to_quantize
            .par_iter()
            .map(|w| self.quantize_weight_to_output(w))
            .collect()
    }

    fn quantize_weight_to_output(
        &self,
        weight: &crate::onnx_utils::WeightTensor,
    ) -> Result<QuantizedWeightOutput> {
        let layer_bits = self.config.bits_for_layer(&weight.name);

        // For the default bit-width, use the shared (possibly calibrated)
        // quantizer.  For per-layer bit-width overrides, build a layer-local
        // quantizer: calibration stats are keyed by the default configuration
        // and re-applying them at a different bit-width is ill-defined.
        let quantized = if layer_bits == self.config.bits {
            self.quantize_tensor_with_name(&weight.name, &weight.data, weight.shape.clone())?
        } else {
            let layer_config = QuantConfig {
                bits: layer_bits,
                per_channel: self.config.per_channel,
                symmetric: self.config.symmetric,
                ..Default::default()
            };
            Quantizer::new(layer_config).quantize_tensor(&weight.data, weight.shape.clone())?
        };

        let mse = quantized.quantization_error(&weight.data);
        let (scales, zero_points) = quantized.get_all_scales_zero_points();
        let is_per_channel = quantized.is_per_channel();
        let bits_used = quantized.bits();
        let quantized_size_bytes = quantized.size_bytes();

        Ok(QuantizedWeightOutput {
            qdq: crate::onnx_utils::graph_builder::QdqWeightInput {
                original_name: weight.name.clone(),
                quantized_values: quantized.data(),
                scales,
                zero_points,
                bits: bits_used,
                axis: if is_per_channel { Some(0) } else { None },
            },
            quantized_size_bytes,
            mse,
        })
    }
}

/// Per-weight result of [`Quantizer::quantize_model`].
///
/// Bundles the QDQ block ready for
/// [`OnnxModel::save_quantized_with_options`](crate::onnx_utils::OnnxModel::save_quantized_with_options)
/// with the two telemetry values callers typically want to print: on-disk
/// size and round-trip MSE.
#[derive(Debug, Clone)]
pub struct QuantizedWeightOutput {
    /// QDQ block for `save_quantized_with_options`.
    pub qdq: crate::onnx_utils::graph_builder::QdqWeightInput,
    /// Size of the quantized payload in bytes (INT8 = 1 byte/elem;
    /// packed INT4 = ceil(elem/2)).
    pub quantized_size_bytes: usize,
    /// MSE between the original FP32 values and the dequantized output.
    pub mse: f32,
}

// ---------------------------------------------------------------------------
// Calibration helper
// ---------------------------------------------------------------------------

/// Compute the finite min/max of `data`, returning an error if all values are NaN/Inf.
fn finite_min_max(data: &[f32], name: &str) -> Result<(f32, f32)> {
    let min = data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let max = data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || !max.is_finite() {
        return Err(QuantizeError::InvalidTensor {
            reason: format!(
                "Tensor '{}' contains only non-finite values (NaN/Inf)",
                name
            ),
        });
    }
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // QuantConfig per-layer selection
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_quantize_no_restrictions() {
        let config = QuantConfig::default();
        assert!(config.should_quantize("any.layer", 1));
        assert!(config.should_quantize("any.layer", 1_000_000));
    }

    #[test]
    fn test_should_quantize_excluded_layer() {
        let config = QuantConfig {
            excluded_layers: vec!["head.weight".to_string()],
            ..Default::default()
        };
        assert!(!config.should_quantize("head.weight", 1024));
        assert!(config.should_quantize("body.weight", 1024));
    }

    #[test]
    fn test_should_quantize_min_elements() {
        let config = QuantConfig {
            min_elements: 512,
            ..Default::default()
        };
        assert!(!config.should_quantize("small.bias", 4));
        assert!(!config.should_quantize("small.bias", 511));
        assert!(config.should_quantize("large.weight", 512));
        assert!(config.should_quantize("large.weight", 1024));
    }

    #[test]
    fn test_should_quantize_excluded_takes_priority_over_min_elements() {
        let config = QuantConfig {
            excluded_layers: vec!["head.weight".to_string()],
            min_elements: 1,
            ..Default::default()
        };
        // excluded → skipped regardless of size
        assert!(!config.should_quantize("head.weight", 1_000_000));
    }

    #[test]
    fn test_bits_for_layer_default() {
        let config = QuantConfig {
            bits: 8,
            ..Default::default()
        };
        assert_eq!(config.bits_for_layer("any.weight"), 8);
    }

    #[test]
    fn test_bits_for_layer_override() {
        let mut layer_bits = std::collections::HashMap::new();
        layer_bits.insert("head.weight".to_string(), 4u8);
        let config = QuantConfig {
            bits: 8,
            layer_bits,
            ..Default::default()
        };
        assert_eq!(config.bits_for_layer("head.weight"), 4);
        assert_eq!(config.bits_for_layer("body.weight"), 8);
    }

    // -----------------------------------------------------------------------
    // Existing tests below
    // -----------------------------------------------------------------------

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

        let quantized = QuantizedTensor::from_f32_per_channel(&data, shape).unwrap();

        assert!(quantized.per_channel);
        assert!(quantized.channel_params.is_some());
        assert_eq!(quantized.channel_params.as_ref().unwrap().len(), 2);

        let dequantized = quantized.to_f32();
        let error: f32 = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

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
        let per_channel = QuantizedTensor::from_f32_per_channel(&data, shape).unwrap();
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

        let per_channel = QuantizedTensor::from_f32_per_channel(&data, shape).unwrap();
        let per_channel_error = per_channel.quantization_error(&data);

        println!("Per-tensor MSE:  {:.8}", per_tensor_error);
        println!("Per-channel MSE: {:.8}", per_channel_error);

        assert!(
            per_channel_error < per_tensor_error,
            "Per-channel ({:.8}) should be better than per-tensor ({:.8})",
            per_channel_error,
            per_tensor_error
        );
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

            println!(
                "Original: {:.2}, Quantized: {}, Dequantized: {:.2}, Error: {:.4}",
                original,
                quantized,
                dequantized,
                (original - dequantized).abs()
            );

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
        let error_int8: f32 = data
            .iter()
            .map(|&v| {
                let q = params_int8.quantize(v);
                let dq = params_int8.dequantize(q);
                (v - dq).powi(2)
            })
            .sum::<f32>()
            / data.len() as f32;

        let params_int4 = QuantParamsInt4::from_range(-1.0, 1.0);
        let error_int4: f32 = data
            .iter()
            .map(|&v| {
                let q = params_int4.quantize(v);
                let dq = params_int4.dequantize(q);
                (v - dq).powi(2)
            })
            .sum::<f32>()
            / data.len() as f32;

        println!("INT8 MSE: {:.8}", error_int8);
        println!("INT4 MSE: {:.8}", error_int4);

        assert!(error_int4 > error_int8);

        assert!(
            error_int4 < error_int8 * 500.0,
            "INT4 error ({:.8}) is too high compared to INT8 ({:.8})",
            error_int4,
            error_int8
        );

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

        let quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape).unwrap();

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
        println!(
            "INT4 (unpacked): {} bytes, MSE: {:.8}",
            int4_size, int4_error
        );
        println!(
            "INT4 (packed): {} bytes, MSE: {:.8}",
            int4_packed_size, int4_error
        );

        assert_eq!(int4_size, int8_size);

        assert!(int4_packed_size <= int8_size / 2 + 1);

        assert!(int4_error > int8_error);

        assert!(int4_error < 0.01, "INT4 error too high: {}", int4_error);
    }

    #[test]
    fn test_int4_large_tensor() {
        let size = 64 * 3 * 3 * 3; // 64 filters, 3x3x3 kernels
        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32 / size as f32) * 2.0 - 1.0) * 0.5)
            .collect();

        let shape = vec![64, 3, 3, 3];

        let quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape).unwrap();

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
                assert!(
                    val >= -8 && val <= 7,
                    "Value {} out of range for {}",
                    val,
                    desc
                );
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
        println!(
            "  Packed:   {} bytes ({}x compression)",
            packed.len(),
            values.len() / packed.len()
        );
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

        assert!(
            (compression_ratio - 2.0).abs() < 0.01,
            "Expected ~2x compression, got {:.2}x",
            compression_ratio
        );
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
        println!(
            "  Compression: {}x",
            quantized.unpacked_size_bytes() / quantized.size_bytes()
        );

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

        let mut quantized = QuantizedTensorInt4::from_f32_per_channel(&data, shape).unwrap();

        let error_before = quantized.quantization_error(&data);
        println!("Error before packing: {:.8}", error_before);

        quantized.pack();

        let error_after = quantized.quantization_error(&data);
        println!("Error after packing:  {:.8}", error_after);
        println!(
            "Size: {} bytes (packed from {} bytes)",
            quantized.size_bytes(),
            quantized.unpacked_size_bytes()
        );

        assert!((error_before - error_after).abs() < 1e-6);

        assert_eq!(quantized.size_bytes(), 500);
    }

    #[test]
    fn test_int4_compression_comparison() {
        let size = 10000;
        let data: Vec<f32> = (0..size)
            .map(|i| ((i as f32 / size as f32) * 2.0 - 1.0) * 0.5)
            .collect();
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
        println!(
            "  INT8:          {} bytes ({:.1}x)",
            int8_size,
            fp32_size as f32 / int8_size as f32
        );
        println!(
            "  INT4 unpacked: {} bytes ({:.1}x)",
            int4_unpacked_size,
            fp32_size as f32 / int4_unpacked_size as f32
        );
        println!(
            "  INT4 packed:   {} bytes ({:.1}x)",
            int4_packed_size,
            fp32_size as f32 / int4_packed_size as f32
        );

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

        let test_weights: Vec<_> = weights
            .iter()
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
            println!(
                "    Shape: {:?}, Elements: {}",
                weight.shape,
                weight.data.len()
            );

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

            let mut int4_packed =
                QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone()).unwrap();
            int4_packed.pack();
            let int4_packed_size = int4_packed.size_bytes();
            let int4_packed_error = int4_packed.quantization_error(&weight.data);

            println!("    FP32:          {:7} bytes", fp32_size);
            println!(
                "    INT8:          {:7} bytes ({:.1}x) MSE: {:.8}",
                int8_size,
                fp32_size as f32 / int8_size as f32,
                int8_error
            );
            println!(
                "    INT4 unpacked: {:7} bytes ({:.1}x) MSE: {:.8}",
                int4_unpacked_size,
                fp32_size as f32 / int4_unpacked_size as f32,
                int4_error
            );
            println!(
                "    INT4 packed:   {:7} bytes ({:.1}x) MSE: {:.8}",
                int4_packed_size,
                fp32_size as f32 / int4_packed_size as f32,
                int4_packed_error
            );

            assert_eq!(int4_error, int4_packed_error, "Packing changed error!");

            let int8_ratio = fp32_size as f32 / int8_size as f32;
            let int4_ratio = fp32_size as f32 / int4_packed_size as f32;

            assert!(
                (int8_ratio - 4.0).abs() < 0.1,
                "INT8 compression should be ~4x"
            );
            assert!(
                (int4_ratio - 8.0).abs() < 0.1,
                "INT4 compression should be ~8x"
            );

            println!();
        }

        println!("\n{}", "=".repeat(60));
        println!("Testing Per-Channel Quantization");
        println!("\n{}", "=".repeat(60));

        // Test per-channel on Conv layers (multi-dimensional)
        let conv_weights: Vec<_> = weights
            .iter()
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
                println!(
                    "    Shape: {:?}, Channels: {}",
                    weight.shape, weight.shape[0]
                );

                let per_tensor =
                    QuantizedTensorInt4::from_f32(&weight.data, weight.shape.clone()).unwrap();
                let per_tensor_error = per_tensor.quantization_error(&weight.data);

                let per_channel_result =
                    QuantizedTensorInt4::from_f32_per_channel(&weight.data, weight.shape.clone());

                if let Ok(per_channel) = per_channel_result {
                    let per_channel_error = per_channel.quantization_error(&weight.data);

                    let improvement =
                        ((per_tensor_error - per_channel_error) / per_tensor_error) * 100.0;

                    println!("    Per-tensor:  MSE: {:.8}", per_tensor_error);
                    println!(
                        "    Per-channel: MSE: {:.8} ({:.1}% better)",
                        per_channel_error, improvement
                    );

                    assert!(
                        per_channel_error <= per_tensor_error * 1.1,
                        "Per-channel should not be significantly worse"
                    );
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

    // -----------------------------------------------------------------------
    // All-NaN / all-Inf edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_nan_returns_error() {
        let data = vec![f32::NAN, f32::NAN, f32::NAN];
        let result = QuantizedTensor::from_f32(&data, vec![3]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("non-finite"),
            "error should mention non-finite: {}",
            err
        );
    }

    #[test]
    fn test_all_inf_returns_error() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY];
        let result = QuantizedTensor::from_f32(&data, vec![2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_nan_int4_returns_error() {
        let data = vec![f32::NAN; 4];
        let result = QuantizedTensorInt4::from_f32(&data, vec![4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_nan_per_channel_returns_error() {
        let data = vec![f32::NAN; 6];
        let result = QuantizedTensor::from_f32_per_channel(&data, vec![2, 3]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Channel 0"),
            "error should mention channel: {}",
            err
        );
    }

    #[test]
    fn test_mixed_nan_finite_succeeds() {
        // Some NaN, some finite — should succeed using finite range
        let data = vec![f32::NAN, 1.0, -1.0, f32::NAN];
        let result = QuantizedTensor::from_f32(&data, vec![4]);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // Symmetric quantization
    // -----------------------------------------------------------------------

    #[test]
    fn test_int8_symmetric_params_zero_point_is_zero() {
        let params = QuantParams::from_range_symmetric(-0.5, 2.0);
        assert_eq!(params.zero_point(), 0, "symmetric must have zp=0");
        // scale = abs_max / QMAX = 2.0 / 127
        let expected_scale = 2.0_f32 / 127.0;
        assert!(
            (params.scale() - expected_scale).abs() < 1e-6,
            "scale {} vs expected {}",
            params.scale(),
            expected_scale
        );
    }

    #[test]
    fn test_int4_symmetric_params_zero_point_is_zero() {
        let params = QuantParamsInt4::from_range_symmetric(-3.0, 1.0);
        assert_eq!(params.zero_point(), 0);
        // scale = 3.0 / 7
        let expected_scale = 3.0_f32 / 7.0;
        assert!((params.scale() - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_symmetric_zero_dequantizes_to_zero() {
        // The defining property of symmetric quantization: 0.0 → 0 → 0.0 exactly.
        let params = QuantParams::from_range_symmetric(-10.0, 10.0);
        let q = params.quantize(0.0);
        assert_eq!(q, 0);
        let dq = params.dequantize(q);
        assert_eq!(dq, 0.0);
    }

    #[test]
    fn test_symmetric_asymmetric_produce_different_scales() {
        // For a skewed range, asymmetric gives a tighter scale than symmetric.
        let asym = QuantParams::from_range(0.0, 10.0);
        let sym = QuantParams::from_range_symmetric(0.0, 10.0);
        assert_ne!(asym.zero_point(), sym.zero_point());
        // Asymmetric packs [0, 10] into [-128, 127] → scale = 10/255 ≈ 0.039
        // Symmetric uses abs_max/127 → scale = 10/127 ≈ 0.079
        assert!(
            sym.scale() > asym.scale(),
            "symmetric scale {} should exceed asymmetric {}",
            sym.scale(),
            asym.scale()
        );
    }

    #[test]
    fn test_symmetric_constant_tensor_handled() {
        // All-zero tensor would give abs_max = 0 → scale must be clamped away from 0.
        let params = QuantParams::from_range_symmetric(0.0, 0.0);
        assert!(params.scale() > 0.0);
        assert_eq!(params.zero_point(), 0);
    }

    #[test]
    fn test_from_f32_symmetric_tensor_has_zero_zp() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let tensor = QuantizedTensor::from_f32_symmetric(&data, vec![100]).unwrap();
        assert_eq!(tensor.params().zero_point(), 0);
    }

    #[test]
    fn test_from_f32_per_channel_symmetric_every_channel_zp_zero() {
        // 4 channels with different ranges.
        let mut data = Vec::new();
        for ch in 0..4 {
            let scale = (ch + 1) as f32;
            for i in 0..16 {
                data.push((i as f32 - 8.0) * 0.1 * scale);
            }
        }
        let tensor = QuantizedTensor::from_f32_per_channel_symmetric(&data, vec![4, 16]).unwrap();

        let channel_params = tensor
            .channel_params
            .as_ref()
            .expect("per-channel expected");
        assert_eq!(channel_params.len(), 4);
        for (i, p) in channel_params.iter().enumerate() {
            assert_eq!(p.zero_point(), 0, "channel {} zp should be 0", i);
            assert!(p.scale() > 0.0, "channel {} scale must be positive", i);
        }
    }

    #[test]
    fn test_symmetric_round_trip_error_bounded() {
        let data: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) / 250.0).collect();
        let tensor = QuantizedTensor::from_f32_symmetric(&data, vec![500]).unwrap();
        let mse = tensor.quantization_error(&data);
        // Symmetric INT8 on [-1, 1] should still be very accurate.
        assert!(mse < 1e-3, "symmetric MSE unexpectedly high: {}", mse);
    }

    #[test]
    fn test_int4_symmetric_round_trip_error_bounded() {
        let data: Vec<f32> = (0..500).map(|i| (i as f32 - 250.0) / 250.0).collect();
        let tensor = QuantizedTensorInt4::from_f32_symmetric(&data, vec![500]).unwrap();
        let mse = tensor.quantization_error(&data);
        // INT4 symmetric has ~15 levels over [-1, 1] → expected MSE < ~0.005
        assert!(mse < 0.01, "INT4 symmetric MSE too high: {}", mse);
    }

    #[test]
    fn test_quantizer_symmetric_config_routes_correctly() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let config = QuantConfig {
            bits: 8,
            per_channel: true,
            symmetric: true,
            ..Default::default()
        };
        let q = Quantizer::new(config)
            .quantize_tensor(&data, vec![4, 16])
            .unwrap();
        let (_, zero_points) = q.get_all_scales_zero_points();
        assert!(
            zero_points.iter().all(|&z| z == 0),
            "all zero_points must be 0 under symmetric config, got {:?}",
            zero_points
        );
    }
}
