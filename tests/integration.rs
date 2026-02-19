//! End-to-end integration tests for quantize-rs.
//!
//! These tests construct ONNX models in memory (no model files checked into
//! the repo) and exercise the full quantization pipeline.

use prost::Message;
use quantize_rs::*;
use quantize_rs::onnx_proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    ValueInfoProto, tensor_proto,
};
use quantize_rs::onnx_utils::graph_builder::QdqWeightInput;
use quantize_rs::quantization::{
    QuantConfig, QuantizedTensor, QuantizedTensorInt4, Quantizer,
};
use std::collections::HashMap;

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a minimal ONNX ModelProto with one Conv node and one weight tensor.
fn build_minimal_model(weight_data: &[f32], weight_shape: &[i64]) -> ModelProto {
    ModelProto {
        opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
            input:  vec![ValueInfoProto { name: "input".to_string(),  ..Default::default() }],
            output: vec![ValueInfoProto { name: "output".to_string(), ..Default::default() }],
            initializer: vec![TensorProto {
                name:       "weight".to_string(),
                data_type:  tensor_proto::DataType::Float as i32,
                dims:       weight_shape.to_vec(),
                float_data: weight_data.to_vec(),
                ..Default::default()
            }],
            node: vec![NodeProto {
                op_type: "Conv".to_string(),
                name:    "conv0".to_string(),
                input:   vec!["input".to_string(), "weight".to_string()],
                output:  vec!["output".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build a two-weight ONNX ModelProto with two Conv nodes chained.
fn build_two_weight_model(
    w1_data: &[f32], w1_shape: &[i64],
    w2_data: &[f32], w2_shape: &[i64],
) -> ModelProto {
    ModelProto {
        opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
        graph: Some(GraphProto {
            name: "test_two_weight".to_string(),
            input:  vec![ValueInfoProto { name: "input".to_string(),  ..Default::default() }],
            output: vec![ValueInfoProto { name: "output".to_string(), ..Default::default() }],
            initializer: vec![
                TensorProto {
                    name:       "w1".to_string(),
                    data_type:  tensor_proto::DataType::Float as i32,
                    dims:       w1_shape.to_vec(),
                    float_data: w1_data.to_vec(),
                    ..Default::default()
                },
                TensorProto {
                    name:       "w2".to_string(),
                    data_type:  tensor_proto::DataType::Float as i32,
                    dims:       w2_shape.to_vec(),
                    float_data: w2_data.to_vec(),
                    ..Default::default()
                },
            ],
            node: vec![
                NodeProto {
                    op_type: "Conv".to_string(),
                    name:    "conv1".to_string(),
                    input:   vec!["input".to_string(), "w1".to_string()],
                    output:  vec!["mid".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Conv".to_string(),
                    name:    "conv2".to_string(),
                    input:   vec!["mid".to_string(), "w2".to_string()],
                    output:  vec!["output".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Encode a ModelProto to a tempfile and return the path.
fn write_model_to_tempfile(
    model: &ModelProto,
    dir: &tempfile::TempDir,
    name: &str,
) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut buf = Vec::new();
    model.encode(&mut buf).unwrap();
    std::fs::write(&path, buf).unwrap();
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
    let quantizer = Quantizer::new(QuantConfig { bits: 8, per_channel: false, calibration_method: None, ..Default::default() });
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
    let quantizer = Quantizer::new(QuantConfig { bits: 4, per_channel: false, calibration_method: None, ..Default::default() });
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
    let quantizer = Quantizer::new(QuantConfig { bits: 8, per_channel: true, calibration_method: None, ..Default::default() });

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
    let quantizer = Quantizer::new(QuantConfig { bits: 3, per_channel: false, calibration_method: None, ..Default::default() });
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

/// Mixed-precision: w1 → INT4, w2 → INT8 in the same model.
#[test]
fn test_mixed_precision_quantization() {
    let w1_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let w2_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let model_proto = build_two_weight_model(&w1_data, &[4, 4], &w2_data, &[4, 4]);

    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();
    assert_eq!(weights.len(), 2);

    // layer_bits: w1 → INT4, w2 → INT8 (default)
    let mut layer_bits = HashMap::new();
    layer_bits.insert("w1".to_string(), 4u8);

    let config = QuantConfig {
        bits: 8,
        per_channel: false,
        calibration_method: None,
        layer_bits,
        ..Default::default()
    };

    let mut qdq_data = Vec::new();
    for w in &weights {
        let layer_config = QuantConfig {
            bits: config.bits_for_layer(&w.name),
            ..config.clone()
        };
        let quantized = Quantizer::new(layer_config)
            .quantize_tensor(&w.data, w.shape.clone())
            .unwrap();

        let (scales, zero_points) = quantized.get_all_scales_zero_points();
        let bits_used = quantized.bits();

        // w1 must be INT4, w2 must be INT8
        if w.name == "w1" {
            assert!(quantized.is_int4(), "w1 should be INT4");
            assert_eq!(bits_used, 4);
        } else {
            assert!(quantized.is_int8(), "w2 should be INT8");
            assert_eq!(bits_used, 8);
        }

        qdq_data.push(QdqWeightInput {
            original_name: w.name.clone(),
            quantized_values: quantized.data(),
            scales,
            zero_points,
            bits: bits_used,
            axis: None,
        });
    }

    // Save
    let output_path = dir.path().join("model_mixed.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    // Reload and validate connectivity
    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken: {:?}", report.broken_refs);

    // Per-weight bit-width must survive the round-trip through metadata
    let qinfo = reloaded.load_quantized_info();
    assert_eq!(qinfo.len(), 2);
    let w1_info = qinfo.iter().find(|q| q.name == "w1").expect("w1 not found");
    let w2_info = qinfo.iter().find(|q| q.name == "w2").expect("w2 not found");
    assert_eq!(w1_info.bits, 4, "w1 bits should be 4 in metadata");
    assert_eq!(w2_info.bits, 8, "w2 bits should be 8 in metadata");
    assert!(w1_info.scale > 0.0);
    assert!(w2_info.scale > 0.0);
}

/// Config file: layer_bits YAML/TOML round-trip and validation.
#[test]
fn test_config_layer_bits() {
    // YAML with layer_bits
    let yaml = r#"
bits: 8
models:
  - input: model.onnx
    output: model_mixed.onnx
    layer_bits:
      conv1.weight: 4
      head.weight: 8
"#;
    let config = Config::from_yaml(yaml).unwrap();
    config.validate().unwrap();
    let model_cfg = &config.models[0];
    let lb = config.get_layer_bits(model_cfg);
    assert_eq!(lb.get("conv1.weight"), Some(&4u8));
    assert_eq!(lb.get("head.weight"), Some(&8u8));

    // TOML with layer_bits
    let toml_str = r#"
bits = 8

[[models]]
input = "model.onnx"
output = "model_mixed.onnx"

[models.layer_bits]
"conv1.weight" = 4
"head.weight" = 8
"#;
    let config = Config::from_toml(toml_str).unwrap();
    config.validate().unwrap();
    let lb = config.get_layer_bits(&config.models[0]);
    assert_eq!(lb.get("conv1.weight"), Some(&4u8));

    // Invalid layer bit-width (3) → Config error
    let yaml_bad = r#"
bits: 8
models:
  - input: model.onnx
    output: model_mixed.onnx
    layer_bits:
      conv1.weight: 3
"#;
    let cfg = Config::from_yaml(yaml_bad).unwrap();
    assert!(matches!(cfg.validate().unwrap_err(), QuantizeError::Config { .. }));
}

// ===========================================================================
// Multi-layer realistic model helpers
// ===========================================================================

/// Build a synthetic multi-layer ONNX model that mirrors a small CNN:
///
/// ```text
/// conv1.weight [8, 3, 3, 3]  → 216 elements   (large, quantizable)
/// conv1.bias   [8]           →   8 elements   (small, skippable via min_elements)
/// conv2.weight [16, 8, 3, 3] → 1152 elements  (large, quantizable)
/// conv2.bias   [16]          →  16 elements   (small, skippable via min_elements)
/// fc.weight    [10, 144]     → 1440 elements  (large, quantizable)
/// fc.bias      [10]          →  10 elements   (small, skippable via min_elements)
/// ```
fn build_multilayer_model() -> ModelProto {
    let make_tensor = |name: &str, dims: &[i64], n: usize| TensorProto {
        name: name.to_string(),
        data_type: tensor_proto::DataType::Float as i32,
        dims: dims.to_vec(),
        float_data: (0..n).map(|i| (i as f32) / (n as f32) - 0.5).collect(),
        ..Default::default()
    };
    ModelProto {
        opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
        graph: Some(GraphProto {
            name: "multilayer".to_string(),
            input:  vec![ValueInfoProto { name: "input".to_string(), ..Default::default() }],
            output: vec![ValueInfoProto { name: "output".to_string(), ..Default::default() }],
            initializer: vec![
                make_tensor("conv1.weight", &[8, 3, 3, 3], 216),
                make_tensor("conv1.bias",   &[8],           8),
                make_tensor("conv2.weight", &[16, 8, 3, 3], 1152),
                make_tensor("conv2.bias",   &[16],          16),
                make_tensor("fc.weight",    &[10, 144],     1440),
                make_tensor("fc.bias",      &[10],          10),
            ],
            node: vec![
                NodeProto {
                    op_type: "Conv".to_string(),
                    name:    "conv1".to_string(),
                    input:   vec!["input".to_string(), "conv1.weight".to_string(), "conv1.bias".to_string()],
                    output:  vec!["conv1_out".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Conv".to_string(),
                    name:    "conv2".to_string(),
                    input:   vec!["conv1_out".to_string(), "conv2.weight".to_string(), "conv2.bias".to_string()],
                    output:  vec!["conv2_out".to_string()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Gemm".to_string(),
                    name:    "fc".to_string(),
                    input:   vec!["conv2_out".to_string(), "fc.weight".to_string(), "fc.bias".to_string()],
                    output:  vec!["output".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Quantize all weights that pass `config.should_quantize`, respecting per-layer bit overrides.
fn quantize_weights(
    config: &QuantConfig,
    weights: &[quantize_rs::WeightTensor],
) -> Vec<QdqWeightInput> {
    weights
        .iter()
        .filter(|w| config.should_quantize(&w.name, w.data.len()))
        .map(|w| {
            let layer_config = QuantConfig {
                bits: config.bits_for_layer(&w.name),
                ..config.clone()
            };
            let quantized = Quantizer::new(layer_config)
                .quantize_tensor(&w.data, w.shape.clone())
                .unwrap();
            let (scales, zero_points) = quantized.get_all_scales_zero_points();
            let bits = quantized.bits();
            let is_pc = quantized.is_per_channel();
            QdqWeightInput {
                original_name: w.name.clone(),
                quantized_values: quantized.data(),
                scales,
                zero_points,
                bits,
                axis: if is_pc { Some(0) } else { None },
            }
        })
        .collect()
}

// ===========================================================================
// Multi-layer integration tests
// ===========================================================================

/// Weights below `min_elements` are skipped; larger weights are quantized.
#[test]
fn test_multilayer_min_elements() {
    let model_proto = build_multilayer_model();
    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();
    assert_eq!(weights.len(), 6);

    // Only tensors with ≥ 100 elements pass; biases (8, 16, 10) are skipped.
    let config = QuantConfig { bits: 8, min_elements: 100, ..Default::default() };
    let qdq_data = quantize_weights(&config, &weights);

    assert_eq!(qdq_data.len(), 3, "expected 3 large weights quantized");
    let names: Vec<&str> = qdq_data.iter().map(|q| q.original_name.as_str()).collect();
    assert!(names.contains(&"conv1.weight"));
    assert!(names.contains(&"conv2.weight"));
    assert!(names.contains(&"fc.weight"));
    assert!(!names.contains(&"conv1.bias"));
    assert!(!names.contains(&"conv2.bias"));
    assert!(!names.contains(&"fc.bias"));

    let output_path = dir.path().join("model_min_elements.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    let reloaded = OnnxModel::load(&output_path).unwrap();
    assert!(reloaded.validate_connectivity().valid);
    assert_eq!(reloaded.load_quantized_info().len(), 3);
}

/// Excluded layers are skipped; all other weights are quantized.
#[test]
fn test_multilayer_excluded_layers() {
    let model_proto = build_multilayer_model();
    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();

    let config = QuantConfig {
        bits: 8,
        excluded_layers: vec!["conv1.weight".to_string(), "fc.bias".to_string()],
        ..Default::default()
    };
    let qdq_data = quantize_weights(&config, &weights);

    // 6 total − 2 excluded = 4 quantized
    assert_eq!(qdq_data.len(), 4, "expected 4 weights after exclusions");
    let names: Vec<&str> = qdq_data.iter().map(|q| q.original_name.as_str()).collect();
    assert!(!names.contains(&"conv1.weight"), "conv1.weight should be excluded");
    assert!(!names.contains(&"fc.bias"),      "fc.bias should be excluded");

    let output_path = dir.path().join("model_excluded.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    let reloaded = OnnxModel::load(&output_path).unwrap();
    assert!(reloaded.validate_connectivity().valid);
    assert_eq!(reloaded.load_quantized_info().len(), 4);
}

/// Full quantize → save → load → validate round-trip for all six layers.
#[test]
fn test_multilayer_full_round_trip() {
    let model_proto = build_multilayer_model();
    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let weights = model.extract_weights();
    assert_eq!(weights.len(), 6);

    let config = QuantConfig { bits: 8, per_channel: true, ..Default::default() };
    let qdq_data = quantize_weights(&config, &weights);
    assert_eq!(qdq_data.len(), 6, "all 6 weights should be quantized");

    let output_path = dir.path().join("model_full.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken: {:?}", report.broken_refs);

    let qinfo = reloaded.load_quantized_info();
    assert_eq!(qinfo.len(), 6, "all 6 weights should appear in metadata");
    for info in &qinfo {
        assert!(info.scale > 0.0, "scale must be positive for {}", info.name);
        assert_eq!(info.bits, 8);
    }
}

/// INT4 produces a smaller byte footprint than INT8 on the same model.
#[test]
fn test_multilayer_compression_ratio() {
    let model_proto = build_multilayer_model();
    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "model.onnx");

    let weights = {
        let m = OnnxModel::load(&model_path).unwrap();
        m.extract_weights()
    };

    let original_bytes: usize = weights.iter().map(|w| w.data.len() * 4).sum();

    let cfg8 = QuantConfig { bits: 8, ..Default::default() };
    let bytes_int8: usize = weights
        .iter()
        .map(|w| Quantizer::new(cfg8.clone()).quantize_tensor(&w.data, w.shape.clone()).unwrap().size_bytes())
        .sum();

    let cfg4 = QuantConfig { bits: 4, ..Default::default() };
    let bytes_int4: usize = weights
        .iter()
        .map(|w| Quantizer::new(cfg4.clone()).quantize_tensor(&w.data, w.shape.clone()).unwrap().size_bytes())
        .sum();

    assert!(bytes_int8 < original_bytes,
        "INT8 ({bytes_int8} B) should be smaller than FP32 ({original_bytes} B)");
    assert!(bytes_int4 < bytes_int8,
        "INT4 ({bytes_int4} B) should be smaller than INT8 ({bytes_int8} B)");

    // INT8 is ~25% of FP32, INT4 is ~12.5%
    let ratio8 = bytes_int8 as f64 / original_bytes as f64;
    let ratio4 = bytes_int4 as f64 / original_bytes as f64;
    assert!(ratio8 < 0.5, "INT8 ratio {ratio8:.2} should be < 0.5");
    assert!(ratio4 < 0.3, "INT4 ratio {ratio4:.2} should be < 0.3");
}

/// Model where weights appear in both graph.input AND graph.initializer
/// (the standard ONNX-1 convention). QDQ transform must remove duplicate
/// input entries so ONNX Runtime doesn't reject the model.
#[test]
fn test_dual_input_initializer_model() {
    let weight_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    // Weight appears as BOTH a graph input and an initializer
    let model_proto = ModelProto {
        opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
        graph: Some(GraphProto {
            name: "dual_input".to_string(),
            input: vec![
                ValueInfoProto { name: "input".to_string(),  ..Default::default() },
                ValueInfoProto { name: "weight".to_string(), ..Default::default() },
            ],
            output: vec![ValueInfoProto { name: "output".to_string(), ..Default::default() }],
            initializer: vec![TensorProto {
                name:       "weight".to_string(),
                data_type:  tensor_proto::DataType::Float as i32,
                dims:       vec![4, 4],
                float_data: weight_data.clone(),
                ..Default::default()
            }],
            node: vec![NodeProto {
                op_type: "Conv".to_string(),
                name:    "conv0".to_string(),
                input:   vec!["input".to_string(), "weight".to_string()],
                output:  vec!["output".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let dir = tempfile::tempdir().unwrap();
    let model_path = write_model_to_tempfile(&model_proto, &dir, "dual.onnx");

    let mut model = OnnxModel::load(&model_path).unwrap();
    let info = model.info();
    assert_eq!(info.inputs.len(), 2, "model should have 2 inputs (data + weight)");

    let weights = model.extract_weights();
    assert_eq!(weights.len(), 1);

    let quantizer = Quantizer::new(QuantConfig { bits: 8, ..Default::default() });
    let quantized = quantizer.quantize_tensor(&weights[0].data, weights[0].shape.clone()).unwrap();
    let (scales, zero_points) = quantized.get_all_scales_zero_points();

    let qdq_data = vec![QdqWeightInput {
        original_name: weights[0].name.clone(),
        quantized_values: quantized.data(),
        scales,
        zero_points,
        bits: 8,
        axis: None,
    }];

    let output_path = dir.path().join("dual_int8.onnx");
    model.save_quantized(&qdq_data, &output_path).unwrap();

    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken: {:?}", report.broken_refs);

    // The weight-as-input entry should have been removed by QDQ transform
    let reloaded_info = reloaded.info();
    assert_eq!(reloaded_info.inputs.len(), 1,
        "QDQ transform should remove weight from graph.input; got {:?}", reloaded_info.inputs);
    assert_eq!(reloaded_info.inputs[0], "input");
}

// ===========================================================================
// Real-model smoke tests (opt-in via environment variable)
//
// Run with:
//   QUANTIZE_RS_TEST_MODEL=/path/to/model.onnx cargo test -- --include-ignored
// ===========================================================================

/// Load a user-supplied ONNX model, quantize to INT8, and validate the result.
#[test]
#[ignore = "set QUANTIZE_RS_TEST_MODEL=/path/to/model.onnx to enable"]
fn test_real_model_int8() {
    let path = std::env::var("QUANTIZE_RS_TEST_MODEL")
        .expect("QUANTIZE_RS_TEST_MODEL must point to an ONNX file");

    let mut model = OnnxModel::load(&path).expect("failed to load model");
    let weights = model.extract_weights();
    assert!(!weights.is_empty(), "model has no extractable weights");

    let config = QuantConfig { bits: 8, per_channel: false, min_elements: 128, ..Default::default() };
    let qdq_data = quantize_weights(&config, &weights);
    assert!(!qdq_data.is_empty(), "no weights passed the min_elements=128 filter");

    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("model_int8.onnx");
    model.save_quantized(&qdq_data, &output_path).expect("save failed");

    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken after INT8 quantization: {:?}", report.broken_refs);

    let qinfo = reloaded.load_quantized_info();
    assert_eq!(qinfo.len(), qdq_data.len(), "metadata weight count mismatch");
    for info in &qinfo {
        assert_eq!(info.bits, 8);
        assert!(info.scale > 0.0);
    }
}

/// Load a user-supplied ONNX model, quantize to INT4, and validate the result.
#[test]
#[ignore = "set QUANTIZE_RS_TEST_MODEL=/path/to/model.onnx to enable"]
fn test_real_model_int4() {
    let path = std::env::var("QUANTIZE_RS_TEST_MODEL")
        .expect("QUANTIZE_RS_TEST_MODEL must point to an ONNX file");

    let mut model = OnnxModel::load(&path).expect("failed to load model");
    let weights = model.extract_weights();

    let config = QuantConfig { bits: 4, per_channel: false, min_elements: 128, ..Default::default() };
    let qdq_data = quantize_weights(&config, &weights);

    let dir = tempfile::tempdir().unwrap();
    let output_path = dir.path().join("model_int4.onnx");
    model.save_quantized(&qdq_data, &output_path).expect("save failed");

    let reloaded = OnnxModel::load(&output_path).unwrap();
    let report = reloaded.validate_connectivity();
    assert!(report.valid, "Connectivity broken after INT4 quantization: {:?}", report.broken_refs);

    let qinfo = reloaded.load_quantized_info();
    assert_eq!(qinfo.len(), qdq_data.len());
    for info in &qinfo {
        assert_eq!(info.bits, 4);
        assert!(info.scale > 0.0);
    }
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
