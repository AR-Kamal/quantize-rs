//! End-to-end integration tests for quantize-rs.
//!
//! These tests construct ONNX models in memory (no model files checked into
//! the repo) and exercise the full quantization pipeline.

use protobuf::Message;
use quantize_rs::*;
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use quantize_rs::quantization::{
    QuantConfig, QuantizedTensor, QuantizedTensorInt4, Quantizer,
};

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a minimal ONNX ModelProto with one Conv node and one weight tensor.
fn build_minimal_model(weight_data: &[f32], weight_shape: &[i64]) -> onnx::onnx::ModelProto {
    let mut model = onnx::onnx::ModelProto::new();

    // Opset
    let mut opset = onnx::onnx::OperatorSetIdProto::new();
    opset.set_version(13);
    model.mut_opset_import().push(opset);

    let graph = model.mut_graph();
    graph.set_name("test_graph".to_string());

    // Graph input
    let mut inp = onnx::onnx::ValueInfoProto::new();
    inp.set_name("input".to_string());
    graph.mut_input().push(inp);

    // Graph output
    let mut out = onnx::onnx::ValueInfoProto::new();
    out.set_name("output".to_string());
    graph.mut_output().push(out);

    // Weight initializer
    let mut w = onnx::onnx::TensorProto::new();
    w.set_name("weight".to_string());
    w.set_data_type(onnx::onnx::TensorProto_DataType::FLOAT);
    for &d in weight_shape {
        w.mut_dims().push(d);
    }
    for &v in weight_data {
        w.mut_float_data().push(v);
    }
    graph.mut_initializer().push(w);

    // Conv node
    let mut conv = onnx::onnx::NodeProto::new();
    conv.set_op_type("Conv".to_string());
    conv.set_name("conv0".to_string());
    conv.mut_input().push("input".to_string());
    conv.mut_input().push("weight".to_string());
    conv.mut_output().push("output".to_string());
    graph.mut_node().push(conv);

    model
}

/// Build a two-weight ONNX ModelProto with two Conv nodes chained.
fn build_two_weight_model(
    w1_data: &[f32], w1_shape: &[i64],
    w2_data: &[f32], w2_shape: &[i64],
) -> onnx::onnx::ModelProto {
    let mut model = onnx::onnx::ModelProto::new();

    let mut opset = onnx::onnx::OperatorSetIdProto::new();
    opset.set_version(13);
    model.mut_opset_import().push(opset);

    let graph = model.mut_graph();
    graph.set_name("test_two_weight".to_string());

    let mut inp = onnx::onnx::ValueInfoProto::new();
    inp.set_name("input".to_string());
    graph.mut_input().push(inp);

    let mut out = onnx::onnx::ValueInfoProto::new();
    out.set_name("output".to_string());
    graph.mut_output().push(out);

    // Weight 1
    let mut w1 = onnx::onnx::TensorProto::new();
    w1.set_name("w1".to_string());
    w1.set_data_type(onnx::onnx::TensorProto_DataType::FLOAT);
    for &d in w1_shape { w1.mut_dims().push(d); }
    for &v in w1_data { w1.mut_float_data().push(v); }
    graph.mut_initializer().push(w1);

    // Weight 2
    let mut w2 = onnx::onnx::TensorProto::new();
    w2.set_name("w2".to_string());
    w2.set_data_type(onnx::onnx::TensorProto_DataType::FLOAT);
    for &d in w2_shape { w2.mut_dims().push(d); }
    for &v in w2_data { w2.mut_float_data().push(v); }
    graph.mut_initializer().push(w2);

    // Conv1
    let mut conv1 = onnx::onnx::NodeProto::new();
    conv1.set_op_type("Conv".to_string());
    conv1.set_name("conv1".to_string());
    conv1.mut_input().push("input".to_string());
    conv1.mut_input().push("w1".to_string());
    conv1.mut_output().push("mid".to_string());
    graph.mut_node().push(conv1);

    // Conv2
    let mut conv2 = onnx::onnx::NodeProto::new();
    conv2.set_op_type("Conv".to_string());
    conv2.set_name("conv2".to_string());
    conv2.mut_input().push("mid".to_string());
    conv2.mut_input().push("w2".to_string());
    conv2.mut_output().push("output".to_string());
    graph.mut_node().push(conv2);

    model
}

/// Serialize a ModelProto to a tempfile and return the path.
fn write_model_to_tempfile(
    model: &onnx::onnx::ModelProto,
    dir: &tempfile::TempDir,
    name: &str,
) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut file = std::fs::File::create(&path).unwrap();
    model.write_to_writer(&mut file).unwrap();
    path
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn test_quantize_simple_model_int8() {
    // Build model in memory
    let weight_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let model_proto = build_minimal_model(&weight_data, &[4, 4]);

    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    // Load
    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].name, "weight");

    // Quantize INT8
    let quantizer = Quantizer::new(QuantConfig { bits: 8, per_channel: false, calibration_method: None });
    let quantized = quantizer.quantize_tensor(&weights[0].data, weights[0].shape.clone()).unwrap();
    assert!(quantized.is_int8());

    let (scales, zero_points) = quantized.get_all_scales_zero_points();
    let is_pc = quantized.is_per_channel();

    let qdq_data = vec![QdqWeightInput {
        original_name: weights[0].name.clone(),
        quantized_values: quantized.data(),
        scales,
        zero_points,
        bits: 8,
        axis: if is_pc { Some(0) } else { None },
    }];

    // Save
    let output_path = dir.path().join("model_int8.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    // Reload and validate
    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken: {:?}", report.broken_refs);

    // Check QDQ info
    let qinfo = reloaded.load_quantized_info();
    assert_eq!(qinfo.len(), 1);
    assert_eq!(qinfo[0].name, "weight");
    assert_eq!(qinfo[0].bits, 8);
    assert!(qinfo[0].scale > 0.0);
}

#[test]
fn test_quantize_simple_model_int4() {
    let weight_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let model_proto = build_minimal_model(&weight_data, &[4, 4]);

    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();

    // Quantize INT4
    let quantizer = Quantizer::new(QuantConfig { bits: 4, per_channel: false, calibration_method: None });
    let quantized = quantizer.quantize_tensor(&weights[0].data, weights[0].shape.clone()).unwrap();
    assert!(quantized.is_int4());
    assert_eq!(quantized.bits(), 4);

    let (scales, zero_points) = quantized.get_all_scales_zero_points();
    let qdq_data = vec![QdqWeightInput {
        original_name: weights[0].name.clone(),
        quantized_values: quantized.data(),
        scales,
        zero_points,
        bits: 4,
        axis: None,
    }];

    let output_path = dir.path().join("model_int4.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken: {:?}", report.broken_refs);

    let qinfo = reloaded.load_quantized_info();
    assert_eq!(qinfo.len(), 1);
    assert_eq!(qinfo[0].bits, 4);
}

#[test]
fn test_quantize_per_channel() {
    // Two channels with very different ranges
    let mut w1_data = Vec::new();
    for i in 0..8 { w1_data.push(i as f32 * 0.01); }  // channel 0: small
    for i in 0..8 { w1_data.push(i as f32 * 1.0); }    // channel 1: large

    let mut w2_data = Vec::new();
    for i in 0..8 { w2_data.push(i as f32 * 0.5); }
    for i in 0..8 { w2_data.push(i as f32 * 2.0); }

    let model_proto = build_two_weight_model(
        &w1_data, &[2, 8],
        &w2_data, &[2, 8],
    );

    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();
    assert_eq!(weights.len(), 2);

    // Per-channel INT8
    let quantizer = Quantizer::new(QuantConfig { bits: 8, per_channel: true, calibration_method: None });

    let mut qdq_data = Vec::new();
    for w in &weights {
        let quantized = quantizer.quantize_tensor(&w.data, w.shape.clone()).unwrap();
        assert!(quantized.is_per_channel());

        let (scales, zero_points) = quantized.get_all_scales_zero_points();
        // Per-channel should have 2 scale/zp values
        assert_eq!(scales.len(), 2);
        assert_eq!(zero_points.len(), 2);

        qdq_data.push(QdqWeightInput {
            original_name: w.name.clone(),
            quantized_values: quantized.data(),
            scales,
            zero_points,
            bits: 8,
            axis: Some(0),
        });
    }

    let output_path = dir.path().join("model_pc.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken: {:?}", report.broken_refs);
}

#[test]
fn test_round_trip_quantization_accuracy() {
    // Known data: linear ramp
    let data: Vec<f32> = (0..1000).map(|i| (i as f32 / 999.0) * 2.0 - 1.0).collect();
    let shape = vec![1000];

    // INT8 round-trip
    let q8 = QuantizedTensor::from_f32(&data, shape.clone()).unwrap();
    let mse_8 = q8.quantization_error(&data);
    assert!(mse_8 < 1e-4, "INT8 MSE too high: {}", mse_8);

    // INT4 round-trip
    let q4 = QuantizedTensorInt4::from_f32(&data, shape).unwrap();
    let mse_4 = q4.quantization_error(&data);
    assert!(mse_4 < 0.01, "INT4 MSE too high: {}", mse_4);

    // INT4 should have higher error than INT8
    assert!(mse_4 > mse_8, "INT4 error should exceed INT8 error");

    // Dequantized data should have the correct length
    assert_eq!(q8.to_f32().len(), 1000);
    assert_eq!(q4.to_f32().len(), 1000);
}

#[test]
fn test_error_variants_are_correct() {
    // Empty tensor → InvalidTensor
    let result = QuantizedTensor::from_f32(&[], vec![0]);
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), QuantizeError::InvalidTensor { .. }),
        "expected InvalidTensor for empty tensor"
    );

    // Shape mismatch → InvalidTensor
    let result = QuantizedTensor::from_f32(&[1.0, 2.0], vec![3]);
    assert!(matches!(result.unwrap_err(), QuantizeError::InvalidTensor { .. }));

    // Per-channel on scalar → InvalidTensor
    let result = QuantizedTensor::from_f32_per_channel(&[1.0], vec![]);
    assert!(matches!(result.unwrap_err(), QuantizeError::InvalidTensor { .. }));

    // Unsupported bits → UnsupportedConfig
    let quantizer = Quantizer::new(QuantConfig { bits: 3, per_channel: false, calibration_method: None });
    let result = quantizer.quantize_tensor(&[1.0, 2.0], vec![2]);
    assert!(matches!(result.unwrap_err(), QuantizeError::UnsupportedConfig { .. }));

    // Model load non-existent → ModelLoad
    let result = OnnxModel::load("/nonexistent/path/model.onnx");
    assert!(matches!(result.unwrap_err(), QuantizeError::ModelLoad { .. }));

    // CalibrationMethod parse error → Config
    let result: Result<quantize_rs::calibration::methods::CalibrationMethod, _> = "invalid".parse();
    assert!(matches!(result.unwrap_err(), QuantizeError::Config { .. }));

    // Config validation error → Config
    let cfg = Config::from_yaml("bits: 3").unwrap();
    assert!(matches!(cfg.validate().unwrap_err(), QuantizeError::Config { .. }));
}

#[test]
fn test_config_round_trip() {
    // Valid YAML
    let yaml = r#"
bits: 8
per_channel: true
models:
  - input: model.onnx
    output: model_int8.onnx
"#;
    let config = Config::from_yaml(yaml).unwrap();
    assert_eq!(config.bits, 8);
    assert!(config.per_channel);
    config.validate().unwrap();

    // Valid TOML
    let toml_str = r#"
bits = 4
per_channel = false

[[models]]
input = "a.onnx"
output = "b.onnx"
"#;
    let config = Config::from_toml(toml_str).unwrap();
    assert_eq!(config.bits, 4);
    config.validate().unwrap();

    // Invalid YAML → Config error
    let result = Config::from_yaml("bits: [invalid");
    assert!(matches!(result.unwrap_err(), QuantizeError::Config { .. }));

    // Invalid TOML → Config error
    let result = Config::from_toml("bits = [invalid");
    assert!(matches!(result.unwrap_err(), QuantizeError::Config { .. }));

    // Invalid bits → Config error
    let cfg = Config::from_yaml("bits: 16").unwrap();
    assert!(matches!(cfg.validate().unwrap_err(), QuantizeError::Config { .. }));

    // Empty model input → Config error
    let yaml_bad = r#"
bits: 8
models:
  - input: ""
    output: "out.onnx"
"#;
    let cfg = Config::from_yaml(yaml_bad).unwrap();
    assert!(matches!(cfg.validate().unwrap_err(), QuantizeError::Config { .. }));
}
