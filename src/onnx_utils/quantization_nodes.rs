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
pub fn build_dequantize_linear_node(names: &DequantLinearNames) -> onnx::onnx::NodeProto {
    let mut node = onnx::onnx::NodeProto::new();
    node.set_op_type("DequantizeLinear".to_string());
    node.set_name(names.node_name.clone());

    node.mut_input().push(names.quantized_name.clone());
    node.mut_input().push(names.scale_name.clone());
    node.mut_input().push(names.zp_name.clone());

    node.mut_output().push(names.output_name.clone());

    node
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
) -> onnx::onnx::TensorProto {
    let mut t = onnx::onnx::TensorProto::new();
    t.set_name(names.quantized_name.clone());
    t.set_data_type(onnx::onnx::TensorProto_DataType::INT8);

    for &d in shape {
        t.mut_dims().push(d);
    }

    // Each i8 value → one byte.  Reinterpret cast, not value conversion.
    t.set_raw_data(values.iter().map(|&v| v as u8).collect());
    t
}

/// FP32 scalar: the quantization scale.
///
/// Rank-0 (no dims) per ONNX scalar convention.
pub fn build_scale_tensor(names: &DequantLinearNames, scale: f32) -> onnx::onnx::TensorProto {
    let mut t = onnx::onnx::TensorProto::new();
    t.set_name(names.scale_name.clone());
    t.set_data_type(onnx::onnx::TensorProto_DataType::FLOAT);
    // rank-0: no dims pushed
    t.mut_float_data().push(scale);
    t
}

/// INT8 scalar: the zero point.
///
/// Rank-0 (no dims).  Stored as a single raw byte (i8 reinterpret cast).
pub fn build_zero_point_tensor(names: &DequantLinearNames, zp: i8) -> onnx::onnx::TensorProto {
    let mut t = onnx::onnx::TensorProto::new();
    t.set_name(names.zp_name.clone());
    t.set_data_type(onnx::onnx::TensorProto_DataType::INT8);
    // rank-0: no dims pushed; single byte raw_data
    t.set_raw_data(vec![zp as u8]);
    t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        let node = build_dequantize_linear_node(&names);

        assert_eq!(node.get_op_type(), "DequantizeLinear");
        assert_eq!(node.get_name(),    "DequantizeLinear_fc.weight");

        let inputs = node.get_input();
        assert_eq!(inputs.len(), 3);
        assert_eq!(inputs[0], "fc.weight_quantized");
        assert_eq!(inputs[1], "fc.weight_scale");
        assert_eq!(inputs[2], "fc.weight_zp");

        let outputs = node.get_output();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], "fc.weight");
    }

    #[test]
    fn test_quantized_weight_tensor_shape_and_data() {
        let names  = DequantLinearNames::from_original("w");
        let values = vec![1i8, -2, 3, -4, 5, 6];
        let shape  = vec![2i64, 3];
        let t = build_quantized_weight_tensor(&names, &values, &shape);

        assert_eq!(t.get_name(),      "w_quantized");
        assert_eq!(t.get_data_type(), onnx::onnx::TensorProto_DataType::INT8);
        assert_eq!(t.get_dims().len(), 2);
        assert_eq!(t.get_dims()[0], 2);
        assert_eq!(t.get_dims()[1], 3);

        // Verify byte-level round-trip
        let recovered: Vec<i8> = t.get_raw_data().iter().map(|&b| b as i8).collect();
        assert_eq!(recovered, values);
    }

    #[test]
    fn test_scale_tensor_is_rank0_float() {
        let names = DequantLinearNames::from_original("w");
        let t = build_scale_tensor(&names, 0.003921);

        assert_eq!(t.get_name(),      "w_scale");
        assert_eq!(t.get_data_type(), onnx::onnx::TensorProto_DataType::FLOAT);
        assert_eq!(t.get_dims().len(), 0, "scale must be rank-0 scalar");
        assert!((t.get_float_data()[0] - 0.003921).abs() < 1e-6);
    }

    #[test]
    fn test_zero_point_tensor_is_rank0_int8() {
        let names = DequantLinearNames::from_original("w");
        let t = build_zero_point_tensor(&names, -3);

        assert_eq!(t.get_name(),      "w_zp");
        assert_eq!(t.get_data_type(), onnx::onnx::TensorProto_DataType::INT8);
        assert_eq!(t.get_dims().len(), 0, "zp must be rank-0 scalar");
        assert_eq!(t.get_raw_data()[0], (-3i8) as u8);
    }

    #[test]
    fn test_int4_range_values_round_trip() {
        // INT4 signed range: [-8, 7].  These arrive as i8; we store them as-is.
        let names  = DequantLinearNames::from_original("w");
        let values = vec![-8i8, -1, 0, 7];
        let shape  = vec![4i64];
        let t = build_quantized_weight_tensor(&names, &values, &shape);

        let recovered: Vec<i8> = t.get_raw_data().iter().map(|&b| b as i8).collect();
        assert_eq!(recovered, values);
    }
}