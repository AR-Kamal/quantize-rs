//! Restricted static INT8 Conv quantization. All mutations happen on a candidate
//! model so validation or I/O errors leave the caller's model unchanged.

use super::{graph_builder::QdqWeightInput, OnnxModel, WeightTensor};
use crate::calibration::{
    methods::CalibrationMethod,
    stats::{calculate_optimal_range_from_stats, ActivationStats},
};
use crate::errors::{QuantizeError, Result};
use crate::onnx_proto::{tensor_proto, tensor_shape_proto, type_proto, NodeProto, TensorProto};
use crate::quantization::{QuantConfig, QuantParams};
use std::collections::{BTreeSet, HashMap, HashSet};

fn unsupported(reason: impl Into<String>) -> QuantizeError {
    QuantizeError::UnsupportedConfig {
        reason: reason.into(),
    }
}

impl OnnxModel {
    /// Sample shape for static Conv calibration (excluding batch).
    ///
    /// Requires one FP32 input with fixed positive NCHW dimensions and batch 1.
    /// Dynamic shapes and multiple inputs are deliberately rejected.
    pub fn calibration_sample_shape(&self) -> Result<Vec<usize>> {
        let graph = self
            .proto
            .graph
            .as_ref()
            .ok_or_else(|| unsupported("model has no graph"))?;
        let inputs: Vec<_> = graph
            .input
            .iter()
            .filter(|i| !graph.initializer.iter().any(|w| w.name == i.name))
            .collect();
        if inputs.len() != 1 {
            return Err(unsupported(
                "static Conv calibration requires exactly one FP32 NCHW input",
            ));
        }
        let tensor = match inputs[0].r#type.as_ref().and_then(|t| t.value.as_ref()) {
            Some(type_proto::Value::TensorType(t))
                if t.elem_type == tensor_proto::DataType::Float as i32 =>
            {
                t
            }
            _ => {
                return Err(unsupported(
                    "static Conv calibration requires an FP32 input",
                ))
            }
        };
        let dims = tensor
            .shape
            .as_ref()
            .ok_or_else(|| unsupported("calibration requires a fixed NCHW input shape"))?;
        let shape: Option<Vec<usize>> = dims
            .dim
            .iter()
            .map(|d| match d.value {
                Some(tensor_shape_proto::dimension::Value::DimValue(n)) if n > 0 => {
                    usize::try_from(n).ok()
                }
                _ => None,
            })
            .collect();
        let shape = shape.ok_or_else(|| {
            unsupported(
                "calibration requires fixed positive NCHW dimensions; export a fixed-shape model",
            )
        })?;
        if shape.len() != 4 || shape[0] != 1 {
            return Err(unsupported(
                "static Conv calibration requires NCHW input with batch size 1",
            ));
        }
        shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| unsupported("input shape overflows usize"))?;
        Ok(shape[1..].to_vec())
    }

    pub(crate) fn calibration_plan(
        &self,
        config: &QuantConfig,
    ) -> Result<(Vec<WeightTensor>, Vec<String>)> {
        self.calibration_sample_shape()?;
        if config.bits != 8 || config.layer_bits.values().any(|b| *b != 8) {
            return Err(unsupported("static activation quantization supports INT8 only; use quantize for INT4 or mixed precision"));
        }
        if self.info().opset_version < 13 {
            return Err(unsupported(
                "static Conv calibration requires opset >= 13; convert the source model first",
            ));
        }
        if self.dropped.any() || self.count_external_data_initializers() > 0 {
            return Err(unsupported("static calibration requires inline tensors and preserved ONNX sections (no local functions, sparse initializers or training_info)"));
        }
        let graph = self
            .proto
            .graph
            .as_ref()
            .ok_or_else(|| unsupported("model has no graph"))?;
        if graph.node.iter().any(|n| {
            (!n.domain.is_empty() && n.domain != "ai.onnx")
                || n.attribute
                    .iter()
                    .any(|a| a.g.is_some() || !a.graphs.is_empty())
                || matches!(n.op_type.as_str(), "QuantizeLinear" | "DequantizeLinear")
        }) {
            return Err(unsupported("static calibration does not support custom domains, subgraphs or already quantized models"));
        }
        let report = self.validate_connectivity();
        if !report.valid {
            return Err(unsupported(report.summary()));
        }
        let conv_weights: HashSet<_> = graph
            .node
            .iter()
            .filter(|n| n.op_type == "Conv")
            .filter_map(|n| n.input.get(1))
            .collect();
        let weights: Vec<_> = self
            .extract_weights()
            .into_iter()
            .filter(|w| {
                conv_weights.contains(&w.name) && config.should_quantize(&w.name, w.num_elements())
            })
            .collect();
        if weights.is_empty() {
            return Err(unsupported("no eligible Conv weights remain after excluded_layers / min_elements; static calibration supports inline FP32 Conv weights only"));
        }
        for weight in &weights {
            if weight.shape.len() != 4
                || weight.shape.contains(&0)
                || weight.data.iter().any(|v| !v.is_finite())
            {
                return Err(unsupported(format!(
                    "Conv weight '{}' must be a finite FP32 rank-4 tensor",
                    weight.name
                )));
            }
        }
        let mut defined: HashSet<String> = HashSet::new();
        for tensor in &graph.initializer {
            if !defined.insert(tensor.name.clone()) {
                return Err(unsupported(format!(
                    "duplicate initializer '{}'",
                    tensor.name
                )));
            }
        }
        defined.extend(graph.input.iter().map(|v| v.name.clone()));
        for node in &graph.node {
            for output in node.output.iter().filter(|n| !n.is_empty()) {
                if !defined.insert(output.clone()) {
                    return Err(unsupported(format!(
                        "duplicate tensor definition '{output}'"
                    )));
                }
            }
        }
        if graph.output.iter().any(|o| !defined.contains(&o.name)) {
            return Err(unsupported("graph output has no tensor definition"));
        }
        // The weight writer uses canonical names, so reject occupied names
        // before calibration rather than generating duplicate ONNX tensors.
        for weight in &weights {
            let generated =
                super::quantization_nodes::DequantLinearNames::from_original(&weight.name);
            for name in [
                generated.quantized_name,
                generated.scale_name,
                generated.zp_name,
            ] {
                if !defined.insert(name.clone()) {
                    return Err(unsupported(format!(
                        "weight QDQ name '{name}' already exists; rename it before calibration"
                    )));
                }
            }
            if graph.node.iter().any(|n| n.name == generated.node_name) {
                return Err(unsupported(format!(
                    "weight QDQ node name '{}' already exists",
                    generated.node_name
                )));
            }
        }
        let names: HashSet<_> = weights.iter().map(|w| &w.name).collect();
        // Shared initializers cannot safely be selected for a non-Conv use too.
        for node in &graph.node {
            for (slot, name) in node.input.iter().enumerate() {
                if names.contains(name) && !(node.op_type == "Conv" && slot == 1) {
                    return Err(unsupported(format!(
                        "Conv weight '{name}' is shared with an unsupported use"
                    )));
                }
            }
        }
        let mut edges = BTreeSet::new();
        for node in &graph.node {
            if node.op_type == "Conv" && node.input.get(1).is_some_and(|w| names.contains(w)) {
                if node.output.len() != 1 || node.output[0].is_empty() || node.input[0].is_empty() {
                    return Err(unsupported("Conv must have a data input and one output"));
                }
                edges.insert(node.input[0].clone());
                edges.insert(node.output[0].clone());
            }
        }
        Ok((weights, edges.into_iter().collect()))
    }

    pub(crate) fn save_calibrated(
        &mut self,
        weights: &[QdqWeightInput],
        stats: &HashMap<String, ActivationStats>,
        method: CalibrationMethod,
        path: &str,
    ) -> Result<()> {
        self.save_calibrated_operators(weights, stats, method, path, &["Conv"])
    }

    pub(crate) fn save_calibrated_operators(
        &mut self,
        weights: &[QdqWeightInput],
        stats: &HashMap<String, ActivationStats>,
        method: CalibrationMethod,
        path: &str,
        operators: &[&str],
    ) -> Result<()> {
        let mut candidate = Self {
            proto: self.proto.clone(),
            dropped: self.dropped.clone(),
        };
        let graph = candidate
            .proto
            .graph
            .as_mut()
            .ok_or_else(|| unsupported("model has no graph"))?;
        let weight_names: HashSet<_> = weights.iter().map(|w| w.original_name.as_str()).collect();
        let selected = |n: &NodeProto| {
            operators.contains(&n.op_type.as_str())
                && n.input
                    .get(1)
                    .is_some_and(|w| weight_names.contains(w.as_str()))
        };
        let output_edges: HashSet<_> = graph
            .node
            .iter()
            .filter(|n| selected(n))
            .map(|n| n.output[0].clone())
            .collect();
        let mut used: HashSet<String> = graph
            .initializer
            .iter()
            .map(|t| t.name.clone())
            .chain(
                graph
                    .input
                    .iter()
                    .chain(&graph.output)
                    .chain(&graph.value_info)
                    .map(|v| v.name.clone()),
            )
            .collect();
        for node in &graph.node {
            used.insert(node.name.clone());
            used.extend(node.input.iter().chain(&node.output).cloned());
        }
        for weight in weights {
            let names =
                super::quantization_nodes::DequantLinearNames::from_original(&weight.original_name);
            used.extend([
                names.quantized_name,
                names.scale_name,
                names.zp_name,
                names.node_name,
            ]);
        }
        let mut counter = 0;
        let mut input_edges: HashMap<String, String> = HashMap::new();
        let mut nodes = Vec::new();
        for mut node in std::mem::take(&mut graph.node) {
            if selected(&node) {
                let edge = node.input[0].clone();
                if !output_edges.contains(&edge) {
                    let dq = if let Some(dq) = input_edges.get(&edge) {
                        dq.clone()
                    } else {
                        let names = fresh_names(&mut used, &mut counter);
                        let dq = names[2].clone();
                        append_qdq(
                            &mut nodes,
                            &mut graph.initializer,
                            stats,
                            method,
                            &edge,
                            &edge,
                            &dq,
                            &names,
                        )?;
                        input_edges.insert(edge, dq.clone());
                        dq
                    };
                    node.input[0] = dq;
                }
                let edge = node.output[0].clone();
                let names = fresh_names(&mut used, &mut counter);
                node.output[0] = names[2].clone();
                nodes.push(node);
                append_qdq(
                    &mut nodes,
                    &mut graph.initializer,
                    stats,
                    method,
                    &edge,
                    &names[2],
                    &edge,
                    &names,
                )?;
            } else {
                nodes.push(node);
            }
        }
        graph.node = nodes;
        let report = candidate.validate_connectivity();
        if !report.valid {
            return Err(unsupported(report.summary()));
        }
        // Reuses the existing atomic writer and weight QDQ transform.
        candidate.save_quantized(weights, path)?;
        *self = candidate;
        Ok(())
    }
}

fn fresh_names(used: &mut HashSet<String>, counter: &mut usize) -> [String; 5] {
    loop {
        let prefix = format!("_quantize_rs_activation_{}", *counter);
        *counter += 1;
        let names = ["scale", "zp", "float", "q", "dq"].map(|suffix| format!("{prefix}_{suffix}"));
        if names.iter().all(|n| !used.contains(n)) {
            used.extend(names.iter().cloned());
            return names;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_qdq(
    nodes: &mut Vec<NodeProto>,
    tensors: &mut Vec<TensorProto>,
    stats: &HashMap<String, ActivationStats>,
    method: CalibrationMethod,
    edge: &str,
    input: &str,
    output: &str,
    names: &[String; 5],
) -> Result<()> {
    let stat = stats
        .get(edge)
        .ok_or_else(|| unsupported(format!("missing activation statistics for tensor '{edge}'")))?;
    if stat.count() == 0 {
        return Err(unsupported(format!(
            "no finite activation samples for '{edge}'"
        )));
    }
    let (min, max) = calculate_optimal_range_from_stats(stat, method);
    if !min.is_finite() || !max.is_finite() || min > max {
        return Err(unsupported(format!(
            "invalid activation range for '{edge}'"
        )));
    }
    // Include zero, including for positive-only and constant activations.
    let params = QuantParams::from_range(min.min(0.0), max.max(0.0));
    if !params.scale().is_finite() || params.scale() <= 0.0 {
        return Err(unsupported(format!(
            "activation scale overflow for '{edge}'"
        )));
    }
    tensors.push(TensorProto {
        name: names[0].clone(),
        data_type: tensor_proto::DataType::Float as i32,
        float_data: vec![params.scale()],
        ..Default::default()
    });
    tensors.push(TensorProto {
        name: names[1].clone(),
        data_type: tensor_proto::DataType::Int8 as i32,
        raw_data: vec![params.zero_point() as u8],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: "QuantizeLinear".into(),
        name: names[3].clone(),
        input: vec![input.into(), names[0].clone(), names[1].clone()],
        output: vec![names[3].clone()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: "DequantizeLinear".into(),
        name: names[4].clone(),
        input: vec![names[3].clone(), names[0].clone(), names[1].clone()],
        output: vec![output.into()],
        ..Default::default()
    });
    Ok(())
}
