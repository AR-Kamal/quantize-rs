//! Direct initializer selection for model-aware weight quantization.
use super::{OnnxModel, WeightTensor};
use crate::errors::{QuantizeError, Result};
use crate::onnx_proto::{attribute_proto, tensor_proto, NodeProto};
use crate::quantization::QuantConfig;
use std::collections::{HashMap, HashSet};

/// A weight selected from graph usage, with an inferred output-channel axis.
///
/// Construct through [`OnnxModel::select_weights`]. Private fields preserve the
/// association between the validated tensor and its axis without changing the
/// existing `WeightTensor` public field set.
#[derive(Debug, Clone)]
pub struct SelectedWeight {
    weight: WeightTensor,
    axis: usize,
}

impl SelectedWeight {
    /// Selected FP32 weight and its original initializer name/shape.
    pub fn weight(&self) -> &WeightTensor {
        &self.weight
    }
    /// Inferred axis (also reported when per-tensor quantization is configured).
    pub fn axis(&self) -> usize {
        self.axis
    }
}

fn unsupported(name: &str, reason: &str) -> QuantizeError {
    QuantizeError::UnsupportedConfig {
        reason: format!("Weight '{name}': {reason}; exclude this initializer to keep it FP32"),
    }
}

fn supported_use(node: &NodeProto, slot: usize) -> bool {
    slot == 1
        && (node.domain.is_empty() || node.domain == "ai.onnx")
        && matches!(node.op_type.as_str(), "Conv" | "MatMul" | "Gemm")
}

fn weight_axis(node: &NodeProto, rank: usize, name: &str) -> Result<usize> {
    match node.op_type.as_str() {
        "Conv" => Ok(0),
        "MatMul" => Ok(rank - 1),
        "Gemm" => {
            if rank != 2 {
                return Err(unsupported(name, "Gemm requires a rank-2 weight"));
            }
            let attrs: Vec<_> = node
                .attribute
                .iter()
                .filter(|a| a.name == "transB")
                .collect();
            if attrs.len() > 1
                || attrs.iter().any(|a| {
                    a.r#type != attribute_proto::AttributeType::Int as i32 || !matches!(a.i, 0 | 1)
                })
            {
                return Err(unsupported(
                    name,
                    "Gemm transB must be a single INT attribute, 0 or 1",
                ));
            }
            Ok(if attrs.first().is_some_and(|a| a.i == 1) {
                0
            } else {
                1
            })
        }
        _ => unreachable!("only supported uses reach axis selection"),
    }
}

impl OnnxModel {
    /// Select direct inline FP32 weights at input 1 of standard Conv, MatMul
    /// and Gemm nodes. Exclusions/minimum size apply before shared-use checks.
    ///
    /// Conv uses axis 0; MatMul uses its right operand's last dimension (rank >= 2);
    /// rank-2 Gemm uses axis 1 or axis 0 when transB=1. Shared initializers require
    /// all consumers to use the same supported weight role and axis, even in
    /// per-tensor mode. Subgraphs and weights exposed as graph outputs are rejected.
    /// Indirect weights through Reshape/Transpose and unrelated tensors are skipped.
    ///
    /// Returns typed errors for conflicting uses, malformed eligible weights,
    /// external selected weights and occupied generated names. This is a selection
    /// policy, not complete ONNX schema validation. Empty selection returns `Ok([])`.
    pub fn select_weights(&self, config: &QuantConfig) -> Result<Vec<SelectedWeight>> {
        let Some(graph) = self.proto.graph.as_ref() else {
            return Ok(Vec::new());
        };
        let mut uses: HashMap<&str, Vec<(&NodeProto, usize)>> = HashMap::new();
        for node in &graph.node {
            for (slot, input) in node.input.iter().enumerate() {
                uses.entry(input).or_default().push((node, slot));
            }
        }
        let mut names: HashSet<&str> = HashSet::new();
        for init in &graph.initializer {
            if !names.insert(&init.name) {
                return Err(unsupported(&init.name, "duplicate initializer name"));
            }
        }
        let subgraphs = graph.node.iter().any(|n| {
            n.attribute
                .iter()
                .any(|a| a.g.is_some() || !a.graphs.is_empty())
        });
        let mut selected = Vec::new();
        for init in &graph.initializer {
            if init.data_type != tensor_proto::DataType::Float as i32
                || init.dims.len() < 2
                || config.excluded_layers.contains(&init.name)
                || init.name.starts_with("_quantize_rs_")
            {
                continue;
            }
            if let Some(base) = init.name.strip_suffix("_scale") {
                if names.contains(format!("{base}_quantized").as_str()) {
                    continue;
                }
            }
            let Some(consumers) = uses.get(init.name.as_str()) else {
                continue;
            };
            if !consumers.iter().any(|(n, slot)| supported_use(n, *slot)) {
                continue;
            }
            let invalid = || QuantizeError::InvalidTensor {
                reason: format!(
                    "Weight '{}' has invalid dimensions or FP32 payload",
                    init.name
                ),
            };
            let shape: Vec<usize> = init
                .dims
                .iter()
                .map(|&d| {
                    usize::try_from(d)
                        .ok()
                        .filter(|&d| d > 0)
                        .ok_or_else(invalid)
                })
                .collect::<Result<_>>()?;
            let elements = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(invalid)?;
            if !config.should_quantize(&init.name, elements) {
                continue;
            }
            if subgraphs {
                return Err(unsupported(
                    &init.name,
                    "subgraph capture analysis is not supported",
                ));
            }
            if graph.output.iter().any(|v| v.name == init.name) {
                return Err(unsupported(
                    &init.name,
                    "initializer is also a graph output",
                ));
            }
            let mut axis = None;
            for &(node, slot) in consumers {
                if !supported_use(node, slot) {
                    return Err(unsupported(
                        &init.name,
                        &format!("shared with unsupported use {} input {slot}", node.op_type),
                    ));
                }
                let this_axis = weight_axis(node, shape.len(), &init.name)?;
                if axis.is_some_and(|a| a != this_axis) {
                    return Err(unsupported(
                        &init.name,
                        "shared consumers require conflicting quantization axes",
                    ));
                }
                axis = Some(this_axis);
            }
            if init.data_location == tensor_proto::DataLocation::External as i32
                || !init.external_data.is_empty()
            {
                return Err(unsupported(
                    &init.name,
                    "only inline FP32 weight data is supported",
                ));
            }
            let data: Vec<f32> = if init.raw_data.is_empty() {
                if init.float_data.len() != elements {
                    return Err(invalid());
                }
                init.float_data.clone()
            } else {
                if elements.checked_mul(4) != Some(init.raw_data.len()) {
                    return Err(invalid());
                }
                init.raw_data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect()
            };
            if data.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            selected.push(SelectedWeight {
                weight: WeightTensor {
                    name: init.name.clone(),
                    data,
                    shape,
                },
                axis: axis.expect("selected candidate has a supported consumer"),
            });
        }
        let mut occupied: HashSet<String> = graph
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
            .chain(
                graph
                    .node
                    .iter()
                    .flat_map(|n| n.input.iter().chain(&n.output).cloned()),
            )
            .collect();
        let mut node_names: HashSet<String> = graph.node.iter().map(|n| n.name.clone()).collect();
        for selection in &selected {
            let name = &selection.weight.name;
            let generated = super::quantization_nodes::DequantLinearNames::from_original(name);
            for generated_name in [
                generated.quantized_name,
                generated.scale_name,
                generated.zp_name,
            ] {
                if !occupied.insert(generated_name.clone()) {
                    return Err(unsupported(
                        name,
                        &format!("generated tensor name '{generated_name}' already exists"),
                    ));
                }
            }
            if !node_names.insert(generated.node_name) {
                return Err(unsupported(
                    name,
                    "generated DequantizeLinear node name already exists",
                ));
            }
        }
        Ok(selected)
    }
}
