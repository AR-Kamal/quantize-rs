//! Deliberately narrow planning for the experimental matrix activation path.
use super::{OnnxModel, SelectedWeight};
use crate::errors::{QuantizeError, Result};
use crate::onnx_proto::{attribute_proto, tensor_proto, tensor_shape_proto, type_proto};
use crate::QuantConfig;
use std::collections::{BTreeSet, HashMap, HashSet};

fn unsupported(reason: impl Into<String>) -> QuantizeError {
    QuantizeError::UnsupportedConfig {
        reason: reason.into(),
    }
}

impl OnnxModel {
    /// Experimental matrix calibration sample shape, excluding the leading 1.
    /// Requires one fixed FP32 input `[1, K]`. Token IDs and dynamic shapes
    /// are not accepted. Other graph restrictions are checked by quantization.
    pub fn matrix_calibration_sample_shape(&self) -> Result<Vec<usize>> {
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
                "matrix calibration requires exactly one FP32 [1, K] input",
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
                    "matrix calibration requires FP32 input; token IDs are unsupported",
                ))
            }
        };
        let dims = tensor
            .shape
            .as_ref()
            .ok_or_else(|| unsupported("matrix calibration requires fixed [1, K] input"))?;
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
            unsupported("matrix calibration requires fixed positive [1, K] dimensions")
        })?;
        if shape.len() != 2
            || shape[0] != 1
            || shape[1]
                .checked_mul(4)
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(unsupported(
                "matrix calibration requires a representable FP32 [1, K] input",
            ));
        }
        Ok(vec![shape[1]])
    }

    pub(crate) fn matrix_calibration_plan(
        &self,
        config: &QuantConfig,
    ) -> Result<(Vec<SelectedWeight>, Vec<String>)> {
        self.matrix_calibration_sample_shape()?;
        if config.bits != 8
            || !config.symmetric
            || !config.per_channel
            || config.layer_bits.values().any(|b| *b != 8)
        {
            return Err(unsupported("matrix calibration requires symmetric per-channel INT8 weights; INT4/mixed precision are unsupported"));
        }
        if self.info().opset_version < 13
            || self.dropped.any()
            || self.count_external_data_initializers() > 0
        {
            return Err(unsupported("matrix calibration requires opset >= 13, inline tensors and preserved ONNX sections"));
        }
        let graph = self
            .proto
            .graph
            .as_ref()
            .ok_or_else(|| unsupported("model has no graph"))?;
        let mut defined = HashSet::new();
        let mut initializers = HashMap::new();
        for t in &graph.initializer {
            if !defined.insert(t.name.clone()) {
                return Err(unsupported(format!("duplicate initializer '{}'", t.name)));
            }
            if t.name.is_empty()
                || t.data_type != tensor_proto::DataType::Float as i32
                || t.dims.len() > 2
            {
                return Err(unsupported(
                    "matrix prototype requires inline FP32 scalar/vector/matrix initializers",
                ));
            }
            let count = t.dims.iter().try_fold(1usize, |n, &d| {
                usize::try_from(d)
                    .ok()
                    .filter(|&d| d > 0)
                    .and_then(|d| n.checked_mul(d))
            });
            let Some(count) = count else {
                return Err(unsupported("invalid matrix initializer dimensions"));
            };
            let valid = if t.raw_data.is_empty() {
                t.float_data.len() == count && t.float_data.iter().all(|x| x.is_finite())
            } else {
                count.checked_mul(4) == Some(t.raw_data.len())
                    && t.raw_data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|&b| f32::from_le_bytes(b).is_finite())
            };
            if !valid {
                return Err(unsupported(format!(
                    "invalid or nonfinite initializer '{}'",
                    t.name
                )));
            }
            initializers.insert(t.name.as_str(), t);
        }
        defined.extend(graph.input.iter().map(|i| i.name.clone()));
        let mut dynamic: HashSet<String> = graph
            .input
            .iter()
            .filter(|i| !initializers.contains_key(i.name.as_str()))
            .map(|i| i.name.clone())
            .collect();
        for node in &graph.node {
            let matrix = matches!(node.op_type.as_str(), "MatMul" | "Gemm");
            let arity_ok = match node.op_type.as_str() {
                "MatMul" | "Add" => node.input.len() == 2,
                "Gemm" => matches!(node.input.len(), 2 | 3),
                "Relu" | "Identity" => node.input.len() == 1,
                _ => false,
            };
            if (!node.domain.is_empty() && node.domain != "ai.onnx")
                || !arity_ok
                || node.output.len() != 1
                || node.output[0].is_empty()
            {
                return Err(unsupported("matrix prototype supports standard MatMul/Gemm, Add, Relu and Identity with normal arities only"));
            }
            let mut attrs = HashSet::new();
            for attr in &node.attribute {
                let valid = node.op_type == "Gemm"
                    && match attr.name.as_str() {
                        "transA" | "transB" => {
                            attr.r#type == attribute_proto::AttributeType::Int as i32
                                && matches!(attr.i, 0 | 1)
                        }
                        "alpha" | "beta" => {
                            attr.r#type == attribute_proto::AttributeType::Float as i32
                                && attr.f == 1.0
                        }
                        _ => false,
                    };
                if !valid || !attrs.insert(&attr.name) {
                    return Err(unsupported("matrix prototype supports Gemm transA/transB=0 or 1 and alpha=beta=1 only; other attributes/subgraphs are unsupported"));
                }
            }
            for (slot, input) in node.input.iter().enumerate() {
                if node.op_type == "Gemm" && slot == 2 && input.is_empty() {
                    continue;
                }
                if input.is_empty() || !defined.contains(input) {
                    return Err(unsupported(format!(
                        "undefined matrix graph input '{input}'"
                    )));
                }
            }
            if matrix {
                if initializers
                    .get(node.input[1].as_str())
                    .is_none_or(|w| w.dims.len() != 2)
                    || !dynamic.contains(&node.input[0])
                {
                    return Err(unsupported("matrix operators require a dynamic left operand and direct rank-2 FP32 initializer right operand"));
                }
                if node.op_type == "Gemm"
                    && node
                        .input
                        .get(2)
                        .is_some_and(|b| !b.is_empty() && !initializers.contains_key(b.as_str()))
                {
                    return Err(unsupported(
                        "Gemm bias must be an inline FP32 initializer or absent",
                    ));
                }
                if node.op_type == "Gemm" {
                    if let Some(bias) = node.input.get(2).filter(|b| !b.is_empty()) {
                        let weight = initializers[node.input[1].as_str()];
                        let trans_b = node
                            .attribute
                            .iter()
                            .any(|a| a.name == "transB" && a.i == 1);
                        let channels = weight.dims[usize::from(!trans_b)];
                        if initializers[bias.as_str()].dims != [channels] {
                            return Err(unsupported("matrix prototype requires Gemm bias shape [N]; scalar/matrix bias forms are unsupported"));
                        }
                    }
                }
            }
            if !defined.insert(node.output[0].clone()) {
                return Err(unsupported("duplicate matrix graph tensor definition"));
            }
            if node.input.iter().any(|name| dynamic.contains(name)) {
                dynamic.insert(node.output[0].clone());
            }
        }
        if graph.output.is_empty() || graph.output.iter().any(|o| !defined.contains(&o.name)) {
            return Err(unsupported("matrix graph output has no tensor definition"));
        }
        let weights = self.select_weights(config)?;
        if weights.is_empty() {
            return Err(unsupported(
                "no eligible MatMul/Gemm weights remain after selection filters",
            ));
        }
        let names: HashSet<_> = weights.iter().map(|w| w.weight().name.as_str()).collect();
        let mut edges = BTreeSet::new();
        for node in &graph.node {
            if matches!(node.op_type.as_str(), "MatMul" | "Gemm")
                && names.contains(node.input[1].as_str())
            {
                edges.insert(node.input[0].clone());
                edges.insert(node.output[0].clone());
            }
        }
        Ok((weights, edges.into_iter().collect()))
    }
}
