//! Graph-aware weight selection and axis propagation, without runtime downloads.
use prost::Message;
use quantize_rs::onnx_proto::{self as pb, attribute_proto, tensor_proto::DataType};
use quantize_rs::{OnnxModel, QuantConfig, Quantizer};

fn weight(name: &str, shape: &[i64]) -> pb::TensorProto {
    pb::TensorProto {
        name: name.into(),
        dims: shape.to_vec(),
        data_type: DataType::Float as i32,
        float_data: (0..shape.iter().product::<i64>())
            .map(|i| i as f32 * 0.1 - 0.5)
            .collect(),
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], output: &str, trans_b: Option<i64>) -> pb::NodeProto {
    pb::NodeProto {
        op_type: op.into(),
        input: inputs.iter().map(|s| (*s).into()).collect(),
        output: vec![output.into()],
        attribute: trans_b
            .map(|i| {
                vec![pb::AttributeProto {
                    name: "transB".into(),
                    i,
                    r#type: attribute_proto::AttributeType::Int as i32,
                    ..Default::default()
                }]
            })
            .unwrap_or_default(),
        ..Default::default()
    }
}

fn proto(nodes: Vec<pb::NodeProto>, weights: Vec<pb::TensorProto>) -> pb::ModelProto {
    pb::ModelProto {
        ir_version: 8,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(pb::GraphProto {
            node: nodes,
            initializer: weights,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn load(proto: &pb::ModelProto) -> OnnxModel {
    OnnxModel::from_bytes(&proto.encode_to_vec()).unwrap()
}

#[test]
fn operator_axes_and_layer_overrides_reach_qdq() {
    let model = load(&proto(
        vec![
            node("Conv", &["X", "conv"], "a", None),
            node("MatMul", &["X", "mm"], "b", None),
            node("MatMul", &["X", "batched"], "c", None),
            node("Gemm", &["X", "gemm"], "d", None),
            node("Gemm", &["X", "gemmt"], "e", Some(1)),
        ],
        vec![
            weight("conv", &[2, 1, 1, 1]),
            weight("mm", &[3, 5]),
            weight("batched", &[2, 3, 5]),
            weight("gemm", &[3, 5]),
            weight("gemmt", &[5, 3]),
        ],
    ));
    let mut config = QuantConfig::int8()
        .with_per_channel(true)
        .with_symmetric(true);
    config.layer_bits.insert("mm".into(), 4);
    let selected = model.select_weights(&config).unwrap();
    assert_eq!(
        selected.iter().map(|w| w.axis()).collect::<Vec<_>>(),
        [0, 1, 2, 1, 0]
    );
    let outputs = Quantizer::new(config)
        .quantize_selected_weights(&selected)
        .unwrap();
    for (out, (axis, channels)) in outputs.iter().zip([(0, 2), (1, 5), (2, 5), (1, 5), (0, 5)]) {
        assert_eq!(out.qdq.axis, Some(axis));
        assert_eq!(out.qdq.scales.len(), channels);
        assert!(out.qdq.zero_points.iter().all(|&z| z == 0));
    }
    assert_eq!(outputs[1].qdq.bits, 4);
    assert_eq!(outputs[1].quantized_size_bytes, 8);
    let tensorwise = Quantizer::new(QuantConfig::int8())
        .quantize_model(&model)
        .unwrap();
    assert!(tensorwise
        .iter()
        .all(|o| o.qdq.axis.is_none() && o.qdq.scales.len() == 1));
}

#[test]
fn embeddings_biases_left_operands_and_indirect_weights_stay_float() {
    let model = load(&proto(
        vec![
            node("MatMul", &["left", "right"], "a", None),
            node("Gather", &["embedding", "indices"], "b", None),
            node("Add", &["a", "bias"], "c", None),
            node("Transpose", &["indirect"], "transposed", None),
            node("MatMul", &["X", "transposed"], "d", None),
        ],
        vec![
            weight("left", &[2, 3]),
            weight("right", &[3, 5]),
            weight("embedding", &[8, 3]),
            weight("bias", &[1, 5]),
            weight("indirect", &[5, 3]),
            weight("unused", &[2, 2]),
        ],
    ));
    let weights = model.select_weights(&QuantConfig::default()).unwrap();
    assert_eq!(weights.len(), 1);
    assert_eq!(weights[0].weight().name, "right");
    // Raw extraction and explicit tensor APIs remain available for advanced use.
    assert_eq!(model.extract_weights().len(), 6);
}

#[test]
fn shared_compatible_weights_are_quantized_once_and_conflicts_rejected() {
    let mut source = proto(
        vec![
            node("MatMul", &["X", "w"], "a", None),
            node("Gemm", &["X", "w"], "b", Some(0)),
        ],
        vec![weight("w", &[3, 5])],
    );
    let config = QuantConfig::int8().with_per_channel(true);
    assert_eq!(load(&source).select_weights(&config).unwrap().len(), 1);
    let graph = source.graph.as_mut().unwrap();
    graph.node[1].attribute[0].i = 1;
    for pc in [false, true] {
        let error = load(&source)
            .select_weights(&QuantConfig::int8().with_per_channel(pc))
            .unwrap_err();
        assert!(error.to_string().contains("conflicting"));
    }
    source.graph.as_mut().unwrap().node[1] = node("Gather", &["w", "indices"], "b", None);
    assert!(load(&source)
        .select_weights(&config)
        .unwrap_err()
        .to_string()
        .contains("unsupported use"));
    let mut excluded = config.clone();
    excluded.excluded_layers.push("w".into());
    assert!(load(&source).select_weights(&excluded).unwrap().is_empty());
    excluded.excluded_layers.clear();
    excluded.min_elements = 16;
    assert!(load(&source).select_weights(&excluded).unwrap().is_empty());
}

#[test]
fn malformed_attributes_payloads_and_ambiguous_graphs_fail_cleanly() {
    let base = proto(
        vec![node("Gemm", &["X", "w"], "Y", Some(0))],
        vec![weight("w", &[3, 5])],
    );
    for case in 0..9 {
        let mut source = base.clone();
        let g = source.graph.as_mut().unwrap();
        match case {
            0 => g.node[0].attribute[0].i = 2,
            1 => g.node[0].attribute[0].r#type = attribute_proto::AttributeType::Float as i32,
            2 => {
                let attr = g.node[0].attribute[0].clone();
                g.node[0].attribute.push(attr);
            }
            3 => g.initializer[0].dims = vec![i64::MAX, 3],
            4 => g.initializer[0].float_data.clear(),
            5 => g.initializer[0].data_location = pb::tensor_proto::DataLocation::External as i32,
            6 => g.initializer.push(weight("w_scale", &[1])),
            7 => g.output.push(pb::ValueInfoProto {
                name: "w".into(),
                ..Default::default()
            }),
            8 => g.node[0].attribute.push(pb::AttributeProto {
                name: "body".into(),
                g: Some(pb::GraphProto::default()),
                ..Default::default()
            }),
            _ => unreachable!(),
        }
        assert!(
            load(&source)
                .select_weights(&QuantConfig::default())
                .is_err(),
            "case {case}"
        );
    }
    let mut custom = base;
    custom.graph.as_mut().unwrap().node[0].domain = "custom".into();
    assert!(load(&custom)
        .select_weights(&QuantConfig::default())
        .unwrap()
        .is_empty());
}
