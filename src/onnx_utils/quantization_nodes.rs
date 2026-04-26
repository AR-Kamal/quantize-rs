//! Low-level builders for ONNX QDQ (Quantize-Dequantize) graph primitives.
//!
//! Each quantized weight becomes four graph elements:
//!
//! ```text
//! Initializers:
//!   "{name}_quantized"  — INT8 tensor, same shape as original
//!   "{name}_scale"      — FP32 scalar
//!   "{name}_zp"         — INT8 scalar
//!
//! Node:
//!   DequantizeLinear
//!     inputs:  ["{name}_quantized", "{name}_scale", "{name}_zp"]
//!     outputs: ["{name}"]          ← original name; downstream graph untouched
//! ```
//!
//! The DequantizeLinear op runs at inference time:
//!   `output = (input - zero_point) × scale`
//! which matches the dequantize formula already used in `QuantParams` and
//! `QuantParamsInt4`.

use crate::onnx_proto::{attribute_proto, tensor_proto, AttributeProto, NodeProto, TensorProto};

// ---------------------------------------------------------------------------
// Name generation
// ---------------------------------------------------------------------------

/// Canonical names for the four graph elements that replace one FP32 initializer.
#[derive(Debug, Clone)]
pub struct DequantLinearNames {
    /// `"{original}_quantized"` — the INT8 weight tensor
    pub quantized_name: String,
    /// `"{original}_scale"` — FP32 scale scalar
    pub scale_name: String,
    /// `"{original}_zp"` — INT8 zero-point scalar
    pub zp_name: String,
    /// `"DequantizeLinear_{original}"` — the node name
    pub node_name: String,
    /// The original tensor name — becomes the DequantizeLinear *output*,
    /// so every downstream node (Conv, MatMul, …) sees no change.
    pub output_name: String,
}

impl DequantLinearNames {
    /// Derive all four names from the original weight tensor name.
    pub fn from_original(original_name: &str) -> Self {
        Self {
            quantized_name: format!("{}_quantized", original_name),
            scale_name: format!("{}_scale", original_name),
            zp_name: format!("{}_zp", original_name),
            node_name: format!("DequantizeLinear_{}", original_name),
            output_name: original_name.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Node builder
// ---------------------------------------------------------------------------

/// Build a DequantizeLinear `NodeProto`.
///
/// ONNX spec (opset ≥ 10):
///   inputs  = [x (INT8), x_scale (FP32), x_zero_point (INT8)]
///   outputs = [y (FP32)]
///   y = (x - x_zero_point) × x_scale
///
/// When `axis` is `Some(a)`, the `axis` attribute is set on the node,
/// enabling per-channel dequantization (opset ≥ 13).
pub fn build_dequantize_linear_node(names: &DequantLinearNames, axis: Option<usize>) -> NodeProto {
    let attribute = match axis {
        Some(a) => vec![AttributeProto {
            name: "axis".to_string(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: a as i64,
            ..Default::default()
        }],
        None => vec![],
    };

    NodeProto {
        op_type: "DequantizeLinear".to_string(),
        name: names.node_name.clone(),
        input: vec![
            names.quantized_name.clone(),
            names.scale_name.clone(),
            names.zp_name.clone(),
        ],
        output: vec![names.output_name.clone()],
        attribute,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// On-disk storage format
// ---------------------------------------------------------------------------

/// How quantized values are stored on disk inside the ONNX initializer.
///
/// ONNX `DequantizeLinear` accepted only INT8 inputs before opset 21.  From
/// opset 21 it also accepts native `INT4` (and `UINT4`), which is 2× smaller
/// on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFormat {
    /// INT4 values widened to INT8 bytes.  Compatible with opset 10+ — the
    /// default for backward compatibility, but gives only 4× compression.
    Int8Widened,
    /// Native `DataType::Int4` with two values packed per byte.  Requires
    /// opset 21.  Gives the full 8× compression for INT4 models.
    NativeInt4,
}

/// Pack INT4 values in ONNX wire-format layout: the element at the **even**
/// index goes into the **low** nibble, odd index into the high nibble.
///
/// See the ONNX spec for `TensorProto` `DataType::Int4`.  This is the opposite
/// nibble order from [`crate::pack_int4`], which uses the library's internal
/// layout (val1 in high nibble).
pub(crate) fn pack_int4_onnx(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for chunk in values.chunks(2) {
        let lo = (chunk[0] & 0x0F) as u8;
        let hi = if chunk.len() > 1 {
            (chunk[1] & 0x0F) as u8
        } else {
            0
        };
        packed.push((hi << 4) | lo);
    }
    packed
}

/// Unpack INT4 values stored in ONNX wire-format layout.  Returns exactly
/// `num_values` `i8`s, sign-extended from 4 bits.
pub(crate) fn unpack_int4_onnx(packed: &[u8], num_values: usize) -> Vec<i8> {
    let mut values = Vec::with_capacity(num_values);
    for &byte in packed {
        let lo = byte & 0x0F;
        let hi = (byte >> 4) & 0x0F;
        values.push(sign_extend_nibble(lo));
        if values.len() < num_values {
            values.push(sign_extend_nibble(hi));
        }
    }
    values.truncate(num_values);
    values
}

#[inline]
fn sign_extend_nibble(nibble: u8) -> i8 {
    if nibble >= 8 {
        (nibble as i8) | !0x0F
    } else {
        nibble as i8
    }
}

// ---------------------------------------------------------------------------
// Initializer builders
// ---------------------------------------------------------------------------

/// Tensor holding the quantized weight values.
///
/// Shape (`dims`) always matches the **logical** element count of the original
/// FP32 tensor.  With [`StorageFormat::Int8Widened`] each element occupies one
/// byte; with [`StorageFormat::NativeInt4`] two elements share a byte so
/// `raw_data.len() == dims.product().div_ceil(2)`.
pub fn build_quantized_weight_tensor(
    names: &DequantLinearNames,
    values: &[i8],
    shape: &[i64],
    format: StorageFormat,
) -> TensorProto {
    match format {
        StorageFormat::Int8Widened => TensorProto {
            name: names.quantized_name.clone(),
            data_type: tensor_proto::DataType::Int8 as i32,
            dims: shape.to_vec(),
            // Each i8 value → one byte.  Reinterpret cast, not value conversion.
            raw_data: values.iter().map(|&v| v as u8).collect(),
            ..Default::default()
        },
        StorageFormat::NativeInt4 => TensorProto {
            name: names.quantized_name.clone(),
            data_type: tensor_proto::DataType::Int4 as i32,
            dims: shape.to_vec(),
            raw_data: pack_int4_onnx(values),
            ..Default::default()
        },
    }
}

/// FP32 scale tensor.
///
/// For per-tensor quantization, `scales` has one element and the tensor
/// is rank-0 (scalar).  For per-channel, `scales` has one entry per
/// channel and the tensor is rank-1 with shape `[num_channels]`.
pub fn build_scale_tensor(names: &DequantLinearNames, scales: &[f32]) -> TensorProto {
    let mut t = TensorProto {
        name: names.scale_name.clone(),
        data_type: tensor_proto::DataType::Float as i32,
        float_data: scales.to_vec(),
        ..Default::default()
    };
    if scales.len() > 1 {
        // rank-1: [num_channels]
        t.dims = vec![scales.len() as i64];
    }
    // For scalar (len == 1), dims remains empty (rank-0 scalar).
    t
}

/// Zero-point tensor.  Data type matches the quantized weight:
///   - [`StorageFormat::Int8Widened`]: `DataType::Int8`, one byte per value.
///   - [`StorageFormat::NativeInt4`]: `DataType::Int4`, packed two per byte.
///
/// For per-tensor, `zps` has one element → rank-0 scalar.
/// For per-channel, `zps` has one per channel → rank-1 `[num_channels]`.
pub fn build_zero_point_tensor(
    names: &DequantLinearNames,
    zps: &[i8],
    format: StorageFormat,
) -> TensorProto {
    let (data_type, raw_data) = match format {
        StorageFormat::Int8Widened => (
            tensor_proto::DataType::Int8 as i32,
            zps.iter().map(|&v| v as u8).collect(),
        ),
        StorageFormat::NativeInt4 => (tensor_proto::DataType::Int4 as i32, pack_int4_onnx(zps)),
    };

    let mut t = TensorProto {
        name: names.zp_name.clone(),
        data_type,
        raw_data,
        ..Default::default()
    };
    if zps.len() > 1 {
        // rank-1: [num_channels]
        t.dims = vec![zps.len() as i64];
    }
    // For scalar (len == 1), dims remains empty (rank-0 scalar).
    t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::tensor_proto;

    #[test]
    fn test_names_from_simple_weight() {
        let n = DequantLinearNames::from_original("conv1.weight");
        assert_eq!(n.quantized_name, "conv1.weight_quantized");
        assert_eq!(n.scale_name, "conv1.weight_scale");
        assert_eq!(n.zp_name, "conv1.weight_zp");
        assert_eq!(n.node_name, "DequantizeLinear_conv1.weight");
        assert_eq!(n.output_name, "conv1.weight");
    }

    #[test]
    fn test_names_from_dotted_path() {
        // Real ResNet-18 weight names look like this
        let n = DequantLinearNames::from_original("layer1.0.conv1.weight");
        assert_eq!(n.quantized_name, "layer1.0.conv1.weight_quantized");
        assert_eq!(n.output_name, "layer1.0.conv1.weight");
    }

    #[test]
    fn test_dequantize_linear_node_inputs_outputs() {
        let names = DequantLinearNames::from_original("fc.weight");
        let node = build_dequantize_linear_node(&names, None);

        assert_eq!(node.op_type, "DequantizeLinear");
        assert_eq!(node.name, "DequantizeLinear_fc.weight");

        assert_eq!(node.input.len(), 3);
        assert_eq!(node.input[0], "fc.weight_quantized");
        assert_eq!(node.input[1], "fc.weight_scale");
        assert_eq!(node.input[2], "fc.weight_zp");

        assert_eq!(node.output.len(), 1);
        assert_eq!(node.output[0], "fc.weight");
        assert!(node.attribute.is_empty());
    }

    #[test]
    fn test_dequantize_linear_node_with_axis() {
        let names = DequantLinearNames::from_original("conv.weight");
        let node = build_dequantize_linear_node(&names, Some(0));

        assert_eq!(node.attribute.len(), 1);
        assert_eq!(node.attribute[0].name, "axis");
        assert_eq!(node.attribute[0].i, 0);
    }

    #[test]
    fn test_quantized_weight_tensor_shape_and_data() {
        let names = DequantLinearNames::from_original("w");
        let values = vec![1i8, -2, 3, -4, 5, 6];
        let shape = vec![2i64, 3];
        let t = build_quantized_weight_tensor(&names, &values, &shape, StorageFormat::Int8Widened);

        assert_eq!(t.name, "w_quantized");
        assert_eq!(t.data_type, tensor_proto::DataType::Int8 as i32);
        assert_eq!(t.dims.len(), 2);
        assert_eq!(t.dims[0], 2);
        assert_eq!(t.dims[1], 3);

        // Verify byte-level round-trip
        let recovered: Vec<i8> = t.raw_data.iter().map(|&b| b as i8).collect();
        assert_eq!(recovered, values);
    }

    #[test]
    fn test_scale_tensor_scalar() {
        let names = DequantLinearNames::from_original("w");
        let t = build_scale_tensor(&names, &[0.003921]);

        assert_eq!(t.name, "w_scale");
        assert_eq!(t.data_type, tensor_proto::DataType::Float as i32);
        assert_eq!(t.dims.len(), 0, "single scale must be rank-0 scalar");
        assert!((t.float_data[0] - 0.003921).abs() < 1e-6);
    }

    #[test]
    fn test_scale_tensor_per_channel() {
        let names = DequantLinearNames::from_original("w");
        let t = build_scale_tensor(&names, &[0.01, 0.02, 0.03]);

        assert_eq!(t.dims.len(), 1);
        assert_eq!(t.dims[0], 3);
        assert_eq!(t.float_data.len(), 3);
    }

    #[test]
    fn test_zero_point_tensor_scalar() {
        let names = DequantLinearNames::from_original("w");
        let t = build_zero_point_tensor(&names, &[-3], StorageFormat::Int8Widened);

        assert_eq!(t.name, "w_zp");
        assert_eq!(t.data_type, tensor_proto::DataType::Int8 as i32);
        assert_eq!(t.dims.len(), 0, "single zp must be rank-0 scalar");
        assert_eq!(t.raw_data[0], (-3i8) as u8);
    }

    #[test]
    fn test_zero_point_tensor_per_channel() {
        let names = DequantLinearNames::from_original("w");
        let t = build_zero_point_tensor(&names, &[-3, 0, 5], StorageFormat::Int8Widened);

        assert_eq!(t.dims.len(), 1);
        assert_eq!(t.dims[0], 3);
        assert_eq!(t.raw_data.len(), 3);
    }

    #[test]
    fn test_int4_range_values_round_trip() {
        // INT4 signed range: [-8, 7].  These arrive as i8; we store them as-is.
        let names = DequantLinearNames::from_original("w");
        let values = vec![-8i8, -1, 0, 7];
        let shape = vec![4i64];
        let t = build_quantized_weight_tensor(&names, &values, &shape, StorageFormat::Int8Widened);

        let recovered: Vec<i8> = t.raw_data.iter().map(|&b| b as i8).collect();
        assert_eq!(recovered, values);
    }

    // -----------------------------------------------------------------------
    // Native INT4 (ONNX opset 21) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_onnx_pack_layout_even_index_in_low_nibble() {
        // ONNX spec: element at even index goes in the low nibble.
        // [0x1, 0x2] → byte = (0x2 << 4) | 0x1 = 0x21
        let packed = pack_int4_onnx(&[1, 2]);
        assert_eq!(packed, vec![0x21]);

        let packed = pack_int4_onnx(&[0, 0x7]);
        assert_eq!(packed, vec![0x70]);
    }

    #[test]
    fn test_onnx_pack_negative_values() {
        // -1 in 4-bit two's complement is 0xF.
        // [-1, -1] → byte = (0xF << 4) | 0xF = 0xFF
        assert_eq!(pack_int4_onnx(&[-1, -1]), vec![0xFF]);

        // [-8, 7] → byte = (0x7 << 4) | 0x8 = 0x78
        assert_eq!(pack_int4_onnx(&[-8, 7]), vec![0x78]);
    }

    #[test]
    fn test_onnx_pack_odd_length_zero_pads_high_nibble() {
        // Single value in the low nibble, high nibble zero.
        assert_eq!(pack_int4_onnx(&[0x3]), vec![0x03]);
        assert_eq!(pack_int4_onnx(&[-1]), vec![0x0F]);
    }

    #[test]
    fn test_onnx_pack_unpack_round_trip_all_values() {
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_int4_onnx(&values);
        let unpacked = unpack_int4_onnx(&packed, values.len());
        assert_eq!(unpacked, values);
        assert_eq!(packed.len(), 8, "16 values must pack to exactly 8 bytes");
    }

    #[test]
    fn test_onnx_pack_unpack_round_trip_odd_length() {
        let values: Vec<i8> = vec![-8, -1, 0, 7, -3];
        let packed = pack_int4_onnx(&values);
        let unpacked = unpack_int4_onnx(&packed, values.len());
        assert_eq!(unpacked, values);
        assert_eq!(packed.len(), 3, "5 values must pack to ceil(5/2) = 3 bytes");
    }

    #[test]
    fn test_native_int4_weight_tensor_uses_int4_data_type() {
        let names = DequantLinearNames::from_original("w");
        let values = vec![-8i8, -1, 0, 7];
        let shape = vec![4i64];
        let t = build_quantized_weight_tensor(&names, &values, &shape, StorageFormat::NativeInt4);

        assert_eq!(t.data_type, tensor_proto::DataType::Int4 as i32);
        assert_eq!(t.dims, vec![4], "dims should be logical element count");
        assert_eq!(t.raw_data.len(), 2, "4 values → 2 packed bytes");

        let recovered = unpack_int4_onnx(&t.raw_data, values.len());
        assert_eq!(recovered, values);
    }

    #[test]
    fn test_native_int4_zero_point_scalar() {
        let names = DequantLinearNames::from_original("w");
        let t = build_zero_point_tensor(&names, &[-3], StorageFormat::NativeInt4);

        assert_eq!(t.data_type, tensor_proto::DataType::Int4 as i32);
        assert_eq!(t.dims.len(), 0, "scalar zp has rank 0");
        assert_eq!(t.raw_data.len(), 1);

        let recovered = unpack_int4_onnx(&t.raw_data, 1);
        assert_eq!(recovered, vec![-3]);
    }

    #[test]
    fn test_native_int4_zero_point_per_channel() {
        let names = DequantLinearNames::from_original("w");
        let zps = vec![-3, 0, 5, -1, 7];
        let t = build_zero_point_tensor(&names, &zps, StorageFormat::NativeInt4);

        assert_eq!(t.data_type, tensor_proto::DataType::Int4 as i32);
        assert_eq!(t.dims, vec![5], "per-channel zp has rank 1");
        assert_eq!(t.raw_data.len(), 3, "5 values → 3 packed bytes");

        let recovered = unpack_int4_onnx(&t.raw_data, zps.len());
        assert_eq!(recovered, zps);
    }
}
