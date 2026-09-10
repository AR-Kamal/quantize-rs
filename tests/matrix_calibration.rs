#![cfg(feature = "calibration")]
//! Prototype matrix calibration through real tract inference and serialization.
use prost::Message;
use quantize_rs::onnx_proto::{self as pb, tensor_proto::DataType};
use quantize_rs::{quantize_static_matrix, CalibrationDataset, OnnxModel, QuantConfig, Quantizer};

fn value(name: &str, shape: &[i64]) -> pb::ValueInfoProto {
    pb::ValueInfoProto {
        name: name.into(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(pb::type_proto::Tensor {
                elem_type: DataType::Float as i32,
                shape: Some(pb::TensorShapeProto {
                    dim: shape
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

fn model(gemm: bool, trans_b: bool) -> pb::ModelProto {
    pb::ModelProto {
        ir_version: 8,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(pb::GraphProto {
            name: "matrix_calibration_test".into(),
            input: vec![value("X", &[1, 3])],
            output: vec![value("Y", &[1, 2])],
            initializer: vec![pb::TensorProto {
                name: "W".into(),
                dims: if trans_b { vec![2, 3] } else { vec![3, 2] },
                data_type: DataType::Float as i32,
                float_data: vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
                ..Default::default()
            }],
            node: vec![pb::NodeProto {
                name: "linear".into(),
                op_type: if gemm { "Gemm" } else { "MatMul" }.into(),
                input: vec!["X".into(), "W".into()],
                output: vec!["Y".into()],
                attribute: if gemm {
                    vec![pb::AttributeProto {
                        name: "transB".into(),
                        r#type: pb::attribute_proto::AttributeType::Int as i32,
                        i: i64::from(trans_b),
                        ..Default::default()
                    }]
                } else {
                    vec![]
                },
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn config() -> QuantConfig {
    QuantConfig::int8()
        .with_per_channel(true)
        .with_symmetric(true)
}
fn data(m: f32) -> CalibrationDataset {
    CalibrationDataset::from_samples(vec![vec![-m, 0., m], vec![m, -m, m / 2.]], vec![3]).unwrap()
}
fn saved(path: &std::path::Path) -> pb::ModelProto {
    pb::ModelProto::decode(std::fs::read(path).unwrap().as_slice()).unwrap()
}

#[test]
fn matrix_axes_and_activation_ranges_come_from_real_inference() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("source.onnx");
    let dst = dir.path().join("output.onnx");
    for (gemm, trans_b) in [(false, false), (true, false), (true, true)] {
        std::fs::write(&src, model(gemm, trans_b).encode_to_vec()).unwrap();
        let weight_only = Quantizer::new(config())
            .quantize_model(&OnnxModel::load(&src).unwrap())
            .unwrap();
        let mut input_scales = vec![];
        for magnitude in [0.01, 100.] {
            quantize_static_matrix(
                src.to_str().unwrap(),
                dst.to_str().unwrap(),
                &data(magnitude),
                config(),
            )
            .unwrap();
            let loaded = OnnxModel::load(&dst).unwrap();
            assert!(loaded.validate_connectivity().valid);
            assert_eq!(loaded.info().inputs, vec!["X"]);
            assert_eq!(loaded.info().outputs, vec!["Y"]);
            let weights = loaded.try_load_quantized_info().unwrap();
            assert_eq!(weights[0].scales, weight_only[0].qdq.scales);
            let graph = saved(&dst).graph.unwrap();
            let dq = graph
                .node
                .iter()
                .find(|n| n.op_type == "DequantizeLinear" && n.output[0] == "W")
                .unwrap();
            assert_eq!(
                dq.attribute.iter().find(|a| a.name == "axis").unwrap().i,
                if trans_b { 0 } else { 1 }
            );
            let q = graph
                .node
                .iter()
                .find(|n| n.op_type == "QuantizeLinear" && n.input[0] == "X")
                .unwrap();
            input_scales.push(
                graph
                    .initializer
                    .iter()
                    .find(|t| t.name == q.input[1])
                    .unwrap()
                    .float_data[0],
            );
            assert_eq!(
                graph
                    .node
                    .iter()
                    .filter(|n| n.op_type == "QuantizeLinear")
                    .count(),
                2
            );
        }
        assert!((input_scales[1] / input_scales[0] - 10000.).abs() < 1.);
    }
}

#[test]
fn compatible_shared_weights_and_inputs_reuse_qdq() {
    let mut source = model(false, false);
    let g = source.graph.as_mut().unwrap();
    let mut other = g.node[0].clone();
    other.name = "other".into();
    other.op_type = "Gemm".into();
    other.output = vec!["Z".into()];
    g.node.push(other);
    g.output.push(value("Z", &[1, 2]));
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("in.onnx");
    let dst = dir.path().join("out.onnx");
    std::fs::write(&src, source.encode_to_vec()).unwrap();
    quantize_static_matrix(
        src.to_str().unwrap(),
        dst.to_str().unwrap(),
        &data(1.),
        config(),
    )
    .unwrap();
    let g = saved(&dst).graph.unwrap();
    assert_eq!(
        g.node
            .iter()
            .filter(|n| n.op_type == "QuantizeLinear")
            .count(),
        3
    );
    assert_eq!(
        g.node
            .iter()
            .filter(|n| n.op_type == "DequantizeLinear" && n.output[0] == "W")
            .count(),
        1
    );
}

#[test]
fn prototype_rejections_preserve_existing_output() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("in.onnx");
    let dst = dir.path().join("out.onnx");
    for case in 0..17 {
        let mut source = model(true, false);
        let mut cfg = config();
        let mut samples = data(1.);
        let g = source.graph.as_mut().unwrap();
        match case {
            0 => g.input[0] = value("X", &[2, 3]),
            1 => g.input.push(value("extra", &[1, 3])),
            2 => g.node[0].attribute[0].i = 2,
            3 => g.node[0].attribute.push(pb::AttributeProto {
                name: "alpha".into(),
                r#type: pb::attribute_proto::AttributeType::Float as i32,
                f: 2.,
                ..Default::default()
            }),
            4 => g.node[0].domain = "custom".into(),
            5 => g.initializer[0].float_data[0] = f32::NAN,
            6 => g.initializer[0].dims = vec![1, 3, 2],
            7 => cfg.bits = 4,
            8 => cfg.symmetric = false,
            9 => cfg.per_channel = false,
            10 => cfg.excluded_layers.push("W".into()),
            11 => samples.samples[0][0] = f32::NAN,
            12 => source.opset_import[0].version = 12,
            13 => g.node[0].input.push("X".into()),
            14 => cfg.min_elements = 7,
            15 => {
                g.initializer.push(pb::TensorProto {
                    name: "bias".into(),
                    dims: vec![1, 2],
                    data_type: DataType::Float as i32,
                    float_data: vec![0., 0.],
                    ..Default::default()
                });
                g.node[0].input.push("bias".into());
            }
            16 => g.node.push(pb::NodeProto {
                op_type: "Add".into(),
                input: vec!["W".into(), "W".into()],
                output: vec!["other".into()],
                ..Default::default()
            }),
            _ => unreachable!(),
        }
        std::fs::write(&src, source.encode_to_vec()).unwrap();
        std::fs::write(&dst, b"keep me").unwrap();
        assert!(
            quantize_static_matrix(src.to_str().unwrap(), dst.to_str().unwrap(), &samples, cfg)
                .is_err(),
            "case {case}"
        );
        assert_eq!(std::fs::read(&dst).unwrap(), b"keep me");
    }
}

#[test]
fn filters_leave_weights_float_and_adjacent_layers_share_activation_qdq() {
    let mut source = model(false, false);
    let g = source.graph.as_mut().unwrap();
    g.node[0].output[0] = "H".into();
    g.initializer.push(pb::TensorProto {
        name: "V".into(),
        dims: vec![2, 2],
        data_type: DataType::Float as i32,
        float_data: vec![0.2, -0.3, 0.4, 0.5],
        ..Default::default()
    });
    g.node.push(pb::NodeProto {
        op_type: "MatMul".into(),
        input: vec!["H".into(), "V".into()],
        output: vec!["Y".into()],
        ..Default::default()
    });
    g.output.push(value("H", &[1, 2]));
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("in.onnx");
    let dst = dir.path().join("out.onnx");
    std::fs::write(&src, source.encode_to_vec()).unwrap();
    for excluded in [None, Some("W"), Some("V")] {
        let mut cfg = config();
        if let Some(name) = excluded {
            cfg.excluded_layers.push(name.into());
        }
        quantize_static_matrix(src.to_str().unwrap(), dst.to_str().unwrap(), &data(1.), cfg)
            .unwrap();
        let output = saved(&dst).graph.unwrap();
        assert_eq!(output.output, source.graph.as_ref().unwrap().output);
        assert_eq!(
            output
                .node
                .iter()
                .filter(|n| n.op_type == "QuantizeLinear")
                .count(),
            if excluded.is_none() { 3 } else { 2 }
        );
        if let Some(name) = excluded {
            let original = source
                .graph
                .as_ref()
                .unwrap()
                .initializer
                .iter()
                .find(|t| t.name == name)
                .unwrap();
            assert_eq!(
                output.initializer.iter().find(|t| t.name == name).unwrap(),
                original
            );
        }
    }
}
