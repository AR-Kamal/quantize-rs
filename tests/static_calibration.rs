#![cfg(feature = "calibration")]
//! Real tract inference and graph serialization; no downloaded model needed.
use prost::Message;
use quantize_rs::onnx_proto::{self as pb, tensor_proto::DataType};
use quantize_rs::{quantize_static, CalibrationDataset, OnnxModel, QuantConfig, Quantizer};

fn value(name: &str, dims: &[i64]) -> pb::ValueInfoProto {
    pb::ValueInfoProto {
        name: name.into(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type: DataType::Float as i32,
                shape: Some(pb::TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|d| pb::tensor_shape_proto::Dimension {
                            value: Some(pb::tensor_shape_proto::dimension::Value::DimValue(*d)),
                            ..Default::default()
                        })
                        .collect(),
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn model() -> pb::ModelProto {
    pb::ModelProto {
        ir_version: 8,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(pb::GraphProto {
            name: "calibration_regression".into(),
            input: vec![value("X", &[1, 1, 2, 2])],
            output: vec![value("Y", &[1, 2, 2, 2])],
            initializer: vec![pb::TensorProto {
                name: "conv.weight".into(),
                dims: vec![2, 1, 1, 1],
                data_type: DataType::Float as i32,
                float_data: vec![-0.1, 0.2],
                ..Default::default()
            }],
            node: vec![pb::NodeProto {
                name: "unrelated_node_name".into(),
                op_type: "Conv".into(),
                input: vec!["X".into(), "conv.weight".into()],
                output: vec!["Y".into()],
                attribute: vec![pb::AttributeProto {
                    name: "kernel_shape".into(),
                    r#type: pb::attribute_proto::AttributeType::Ints as i32,
                    ints: vec![1, 1],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn data(magnitude: f32) -> CalibrationDataset {
    CalibrationDataset::from_samples(
        vec![vec![-magnitude, magnitude, 0.0, magnitude / 2.0]],
        vec![1, 2, 2],
    )
    .unwrap()
}

#[test]
fn test_real_calibration_changes_activations_not_weights() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("source.onnx");
    std::fs::write(&src, model().encode_to_vec()).unwrap();
    for per_channel in [false, true] {
        let cfg = QuantConfig::int8()
            .with_per_channel(per_channel)
            .with_symmetric(true);
        let weight_only = Quantizer::new(cfg.clone())
            .quantize_model(&OnnxModel::load(&src).unwrap())
            .unwrap();
        let mut scales = Vec::new();
        for magnitude in [0.01, 100.0] {
            let dst = dir.path().join("calibrated.onnx");
            quantize_static(
                src.to_str().unwrap(),
                dst.to_str().unwrap(),
                &data(magnitude),
                cfg.clone(),
            )
            .unwrap();
            let loaded = OnnxModel::load(&dst).unwrap();
            assert!(loaded.validate_connectivity().valid);
            assert_eq!(loaded.info().inputs, vec!["X"]);
            assert_eq!(loaded.info().outputs, vec!["Y"]);
            assert_eq!(
                loaded.load_quantized_info()[0].scales,
                weight_only[0].qdq.scales
            );
            let output = pb::ModelProto::decode(std::fs::read(dst).unwrap().as_slice()).unwrap();
            let graph = output.graph.unwrap();
            assert_eq!(
                graph
                    .node
                    .iter()
                    .filter(|n| n.op_type == "QuantizeLinear")
                    .count(),
                2
            );
            let weight = graph
                .initializer
                .iter()
                .find(|t| t.name == "conv.weight_quantized")
                .unwrap();
            assert_eq!(
                weight.raw_data,
                weight_only[0]
                    .qdq
                    .quantized_values
                    .iter()
                    .map(|v| *v as u8)
                    .collect::<Vec<_>>()
            );
            let q = graph
                .node
                .iter()
                .find(|n| n.op_type == "QuantizeLinear" && n.input[0] == "X")
                .unwrap();
            let scale = graph
                .initializer
                .iter()
                .find(|t| t.name == q.input[1])
                .unwrap();
            let zp = graph
                .initializer
                .iter()
                .find(|t| t.name == q.input[2])
                .unwrap();
            assert_eq!(zp.data_type, DataType::Int8 as i32);
            assert!(scale.dims.is_empty());
            scales.push(scale.float_data[0]);
        }
        assert!((scales[1] / scales[0] - 10_000.0).abs() < 1.0);
    }
}

#[test]
fn test_filters_shared_inputs_and_generated_name_collisions() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("source.onnx");
    let dst = dir.path().join("result.onnx");
    let mut m = model();
    let g = m.graph.as_mut().unwrap();
    // Both Conv nodes share the input but use different weights.
    let mut w = g.initializer[0].clone();
    w.name = "second.weight".into();
    g.initializer.push(w);
    let mut n = g.node[0].clone();
    n.name = "second".into();
    n.input[1] = "second.weight".into();
    n.output[0] = "Z".into();
    g.node.push(n);
    g.output.push(value("Z", &[1, 2, 2, 2]));
    g.initializer.push(pb::TensorProto {
        name: "_quantize_rs_activation_0_scale".into(),
        data_type: DataType::Float as i32,
        float_data: vec![42.0],
        ..Default::default()
    });
    std::fs::write(&src, m.encode_to_vec()).unwrap();
    for excluded in [false, true] {
        let mut cfg = QuantConfig::int8();
        if excluded {
            cfg.excluded_layers.push("second.weight".into());
        }
        quantize_static(
            src.to_str().unwrap(),
            dst.to_str().unwrap(),
            &data(1.0),
            cfg,
        )
        .unwrap();
        let g = pb::ModelProto::decode(std::fs::read(&dst).unwrap().as_slice())
            .unwrap()
            .graph
            .unwrap();
        assert_eq!(
            g.node
                .iter()
                .filter(|n| n.op_type == "QuantizeLinear" && n.input[0] == "X")
                .count(),
            1
        );
        assert_eq!(
            g.initializer
                .iter()
                .find(|t| t.name == "_quantize_rs_activation_0_scale")
                .unwrap()
                .float_data,
            vec![42.0]
        );
        assert!(OnnxModel::load(&dst).unwrap().validate_connectivity().valid);
        if excluded {
            let second = g.node.iter().find(|n| n.name == "second").unwrap();
            assert_eq!(second.input, vec!["X", "second.weight"]);
            assert_eq!(second.output, vec!["Z"]);
        }
    }
    let mut cfg = QuantConfig::int8();
    cfg.min_elements = 3;
    assert!(quantize_static(
        src.to_str().unwrap(),
        dst.to_str().unwrap(),
        &data(1.0),
        cfg
    )
    .is_err());
}

#[test]
fn test_unsupported_calibration_preserves_existing_output() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("source.onnx");
    let dst = dir.path().join("existing.onnx");
    std::fs::write(&dst, b"existing output").unwrap();
    for case in 0..9 {
        let mut m = model();
        let mut cfg = QuantConfig::int8();
        let mut dataset = data(1.0);
        match case {
            0 => cfg.bits = 4,
            1 => {
                cfg.layer_bits.insert("conv.weight".into(), 4);
            }
            2 => m
                .graph
                .as_mut()
                .unwrap()
                .input
                .push(value("extra", &[1, 1, 2, 2])),
            3 => m.graph.as_mut().unwrap().input[0] = value("X", &[-1, 1, 2, 2]),
            4 => dataset.samples[0][0] = f32::NAN,
            5 => m.opset_import[0].version = 11,
            6 => cfg.excluded_layers.push("conv.weight".into()),
            7 => m.graph.as_mut().unwrap().initializer.push(pb::TensorProto {
                name: "conv.weight_scale".into(),
                data_type: DataType::Float as i32,
                float_data: vec![1.0],
                ..Default::default()
            }),
            8 => m.graph.as_mut().unwrap().initializer[0].float_data = vec![-f32::MAX, f32::MAX],
            _ => unreachable!(),
        }
        std::fs::write(&src, m.encode_to_vec()).unwrap();
        assert!(
            quantize_static(src.to_str().unwrap(), dst.to_str().unwrap(), &dataset, cfg).is_err(),
            "case {case}"
        );
        assert_eq!(std::fs::read(&dst).unwrap(), b"existing output");
    }
}
