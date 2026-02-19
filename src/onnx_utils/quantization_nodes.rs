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

use crate::onnx_proto::{
    AttributeProto, NodeProto, TensorProto,
    attribute_proto, tensor_proto,
};

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
            scale_name:     format!("{}_scale", original_name),
            zp_name:        format!("{}_zp", original_name),
            node_name:      format!("DequantizeLinear_{}", original_name),
            output_name:    original_name.to_string(),
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
pub fn build_dequantize_linear_node(
    names: &DequantLinearNames,
    axis: Option<usize>,
) -> NodeProto {
    let attribute = match axis {
        Some(a) => vec![AttributeProto {
            name:   "axis".to_string(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i:      a as i64,
            ..Default::default()
        }],
        None => vec![],
    };

    NodeProto {
        op_type:   "DequantizeLinear".to_string(),
        name:      names.node_name.clone(),
        input:     vec![
            names.quantized_name.clone(),
            names.scale_name.clone(),
            names.zp_name.clone(),
        ],
        output:    vec![names.output_name.clone()],
        attribute,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Initializer builders
// ---------------------------------------------------------------------------

/// INT8 tensor holding the quantized weight values.
///
/// Shape matches the original FP32 tensor exactly.  For INT4-quantized values
/// (range [-8, 7]), the i8 bytes are stored directly — see the INT4 note in
/// `graph_builder::apply_qdq_transform`.
pub fn build_quantized_weight_tensor(
    names: &DequantLinearNames,
    values: &[i8],
    shape: &[i64],
) -> TensorProto {
    TensorProto {
        name:      names.quantized_name.clone(),
        data_type: tensor_proto::DataType::Int8 as i32,
        dims:      shape.to_vec(),
        // Each i8 value → one byte.  Reinterpret cast, not value conversion.
        raw_data:  values.iter().map(|&v| v as u8).collect(),
        ..Default::default()
    }
}

/// FP32 scale tensor.
///
/// For per-tensor quantization, `scales` has one element and the tensor
/// is rank-0 (scalar).  For per-channel, `scales` has one entry per
/// channel and the tensor is rank-1 with shape `[num_channels]`.
pub fn build_scale_tensor(names: &DequantLinearNames, scales: &[f32]) -> TensorProto {
    let mut t = TensorProto {
        name:       names.scale_name.clone(),
        data_type:  tensor_proto::DataType::Float as i32,
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

/// INT8 zero-point tensor.
///
/// For per-tensor, `zps` has one element → rank-0 scalar.
/// For per-channel, `zps` has one per channel → rank-1 `[num_channels]`.
pub fn build_zero_point_tensor(names: &DequantLinearNames, zps: &[i8]) -> TensorProto {
    let mut t = TensorProto {
        name:      names.zp_name.clone(),
        data_type: tensor_proto::DataType::Int8 as i32,
        raw_data:  zps.iter().map(|&v| v as u8).collect(),
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
        assert_eq!(n.scale_name,     "conv1.weight_scale");
        assert_eq!(n.zp_name,        "conv1.weight_zp");
        assert_eq!(n.node_name,      "DequantizeLinear_conv1.weight");
        assert_eq!(n.output_name,    "conv1.weight");
    }

    #[test]
    fn test_names_from_dotted_path() {
        // Real ResNet-18 weight names look like this
        let n = DequantLinearNames::from_original("layer1.0.conv1.weight");
        assert_eq!(n.quantized_name, "layer1.0.conv1.weight_quantized");
        assert_eq!(n.output_name,    "layer1.0.conv1.weight");
    }

    #[test]
    fn test_dequantize_linear_node_inputs_outputs() {
        let names = DequantLinearNames::from_original("fc.weight");
        let node = build_dequantize_linear_node(&names, None);

        assert_eq!(node.op_type, "DequantizeLinear");
        assert_eq!(node.name,    "DequantizeLinear_fc.weight");

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
        let names  = DequantLinearNames::from_original("w");
        let values = vec![1i8, -2, 3, -4, 5, 6];
        let shape  = vec![2i64, 3];
        let t = build_quantized_weight_tensor(&names, &values, &shape);

        assert_eq!(t.name,      "w_quantized");
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

        assert_eq!(t.name,      "w_scale");
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
        let t = build_zero_point_tensor(&names, &[-3]);

        assert_eq!(t.name,      "w_zp");
        assert_eq!(t.data_type, tensor_proto::DataType::Int8 as i32);
        assert_eq!(t.dims.len(), 0, "single zp must be rank-0 scalar");
        assert_eq!(t.raw_data[0], (-3i8) as u8);
    }

    #[test]
    fn test_zero_point_tensor_per_channel() {
        let names = DequantLinearNames::from_original("w");
        let t = build_zero_point_tensor(&names, &[-3, 0, 5]);

        assert_eq!(t.dims.len(), 1);
        assert_eq!(t.dims[0], 3);
        assert_eq!(t.raw_data.len(), 3);
    }

    #[test]
    fn test_int4_range_values_round_trip() {
        // INT4 signed range: [-8, 7].  These arrive as i8; we store them as-is.
        let names  = DequantLinearNames::from_original("w");
        let values = vec![-8i8, -1, 0, 7];
        let shape  = vec![4i64];
        let t = build_quantized_weight_tensor(&names, &values, &shape);

        let recovered: Vec<i8> = t.raw_data.iter().map(|&b| b as i8).collect();
        assert_eq!(recovered, values);
    }
}
