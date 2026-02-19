//! Graph-level operations for quantized ONNX models.
//!
//! Three responsibilities:
//!   1. **QDQ transform** — replace FP32 initializers with INT8 + DequantizeLinear
//!   2. **Connectivity validation** — walk the graph and verify every edge resolves
//!   3. **Opset management** — ensure the model declares opset ≥ 13

use crate::errors::{QuantizeError, Result};
use crate::onnx_proto::{GraphProto, ModelProto, OperatorSetIdProto};
use std::collections::{HashMap, HashSet};

use super::quantization_nodes::{
    DequantLinearNames,
    build_dequantize_linear_node,
    build_quantized_weight_tensor,
    build_scale_tensor,
    build_zero_point_tensor,
};

// ===========================================================================
// Public types
// ===========================================================================

/// One weight to convert: FP32 initializer → INT8 + DequantizeLinear block.
#[derive(Debug)]
pub struct QdqWeightInput {
    /// Original initializer name (e.g., `"conv1.weight"`)
    pub original_name: String,
    /// Quantized values as i8.  For INT4 these are in [-8, 7]; for INT8 in [-128, 127].
    /// Always unpacked — one value per element.
    pub quantized_values: Vec<i8>,
    /// Quantization scales (FP32).
    /// Length 1 for per-tensor; one per channel for per-channel.
    pub scales: Vec<f32>,
    /// Zero points (INT8).
    /// Same length as `scales`.
    pub zero_points: Vec<i8>,
    /// Original bit-width (4 or 8).  Informational only — ONNX storage is always INT8.
    /// Persisted in model metadata so `load_quantized_info` can recover it.
    pub bits: u8,
    /// Per-channel quantization axis, or `None` for per-tensor.
    pub axis: Option<usize>,
}

/// Result of a graph-connectivity check.
#[derive(Debug)]
#[must_use]
pub struct ConnectivityReport {
    /// `true` if every node input resolves to a known tensor.
    pub valid: bool,
    /// Human-readable description of every dangling reference.  Empty when valid.
    pub broken_refs: Vec<String>,
}

impl ConnectivityReport {
    /// Render the report as a printable string (useful for CLI `validate` output).
    pub fn summary(&self) -> String {
        if self.valid {
            "  Graph connectivity: OK\n".to_string()
        } else {
            let mut s = format!(
                "  Graph connectivity: BROKEN ({} dangling reference{})\n",
                self.broken_refs.len(),
                if self.broken_refs.len() == 1 { "" } else { "s" }
            );
            for (i, r) in self.broken_refs.iter().enumerate() {
                s.push_str(&format!("    {}. {}\n", i + 1, r));
            }
            s
        }
    }
}

// ===========================================================================
// Connectivity validation
// ===========================================================================

/// Walk the graph and verify every node input resolves to *something*.
///
/// A valid input is exactly one of:
///   • a declared graph input (`graph.input`)
///   • an initializer name (`graph.initializer`)
///   • the output of a node that appears **earlier** in `graph.node`
///
/// This is the check ONNX Runtime performs on load — and the check that
/// v0.2.0's `validate` command skipped, letting the rename bug through.
pub fn validate_graph_connectivity(graph: &GraphProto) -> ConnectivityReport {
    let mut known: HashSet<String> = HashSet::new();

    // Seed: graph inputs + initializers are always available
    for inp in &graph.input {
        known.insert(inp.name.clone());
    }
    for init in &graph.initializer {
        known.insert(init.name.clone());
    }

    let mut broken = Vec::new();

    // Walk nodes in serialized order; each node's outputs become known afterwards
    for node in &graph.node {
        for name in &node.input {
            if name.is_empty() {
                continue; // optional input slot — empty string is valid
            }
            if !known.contains(name.as_str()) {
                broken.push(format!(
                    "Node '{}' (op={}) → unknown input '{}'",
                    node.name, node.op_type, name
                ));
            }
        }
        // Register outputs so later nodes can consume them
        for name in &node.output {
            if !name.is_empty() {
                known.insert(name.clone());
            }
        }
    }

    ConnectivityReport {
        valid: broken.is_empty(),
        broken_refs: broken,
    }
}

// ===========================================================================
// Opset version management
// ===========================================================================

/// Ensure the default ONNX domain opset is at least `min_version`.
///
/// DequantizeLinear requires opset ≥ 10 (per-tensor) or ≥ 13 (per-channel axis).
/// We always request 13 to leave the door open for per-channel.
pub fn ensure_opset_version(model: &mut ModelProto, min_version: i64) {
    // The default ONNX domain is identified by an empty string
    for opset in model.opset_import.iter_mut() {
        if opset.domain.is_empty() {
            if opset.version < min_version {
                opset.version = min_version;
            }
            return; // found and updated (or already sufficient)
        }
    }

    // No default-domain entry at all — add one
    model.opset_import.push(OperatorSetIdProto {
        domain:  String::new(), // "" = standard ONNX domain
        version: min_version,
    });
}

// ===========================================================================
// QDQ transform
// ===========================================================================

/// Replace FP32 weight initializers with INT8 quantized equivalents +
/// DequantizeLinear nodes.
///
/// ### What happens per weight in `inputs`:
///
/// **Removed:**
///   - Initializer `"{name}"` (the original FP32 weight data)
///
/// **Added (initializers):**
///   - `"{name}_quantized"` — INT8, same shape as original
///   - `"{name}_scale"`     — FP32 scalar
///   - `"{name}_zp"`        — INT8 scalar
///
/// **Added (node, prepended before all existing nodes):**
///   - `DequantizeLinear` with output = `"{name}"`
///
/// Because the DequantizeLinear output carries the **original** name, every
/// downstream node (Conv, MatMul, BatchNorm, …) remains completely unchanged.
/// Graph connectivity is preserved by construction.
///
/// ---
/// ### INT4 storage note
///
/// ONNX `DequantizeLinear` requires INT8 input in opsets < 21.  INT4-quantized
/// values (range [-8, 7]) are widened to INT8 here.  The quantization *accuracy*
/// is INT4-level (scale and zero_point were computed for the 4-bit range), but
/// on-disk storage is 4× compression rather than the 8× that bit-packing would
/// give.  True INT4 packing is planned for a future version (opset 21 or custom op).
pub fn apply_qdq_transform(
    graph: &mut GraphProto,
    inputs: &[QdqWeightInput],
) -> Result<()> {
    // -----------------------------------------------------------------------
    // 0.  Snapshot shapes before modifying the initializer list
    // -----------------------------------------------------------------------
    let shape_map: HashMap<String, Vec<i64>> = graph
        .initializer
        .iter()
        .map(|init| (init.name.clone(), init.dims.clone()))
        .collect();

    let quant_set: HashSet<&str> = inputs.iter().map(|i| i.original_name.as_str()).collect();

    // -----------------------------------------------------------------------
    // 1.  Remove the original FP32 initializers for every weight we're replacing
    // -----------------------------------------------------------------------
    graph.initializer.retain(|init| !quant_set.contains(init.name.as_str()));

    // -----------------------------------------------------------------------
    // 1b. Also remove weights from graph.input (critical fix for "Duplicate definition")
    // -----------------------------------------------------------------------
    // Some ONNX models list weights as both initializers AND graph inputs.
    // This is valid ONNX, but when DequantizeLinear outputs reuse the original
    // weight names, ONNX Runtime sees two definitions of the same tensor.
    graph.input.retain(|inp| !quant_set.contains(inp.name.as_str()));

    // -----------------------------------------------------------------------
    // 2.  Add quantized initializer triples + build DequantizeLinear nodes
    // -----------------------------------------------------------------------
    let mut dq_nodes = Vec::new();

    for inp in inputs {
        let shape = shape_map
            .get(&inp.original_name)
            .ok_or_else(|| {
                QuantizeError::GraphTransform {
                    reason: format!(
                        "Weight '{}' not found in model initializers — \
                         verify the name matches exactly",
                        inp.original_name
                    ),
                }
            })?;

        let expected_len: i64 = shape.iter().product();
        if inp.quantized_values.len() as i64 != expected_len {
            return Err(QuantizeError::GraphTransform {
                reason: format!(
                    "Weight '{}': quantized_values has {} elements but shape {:?} expects {}",
                    inp.original_name, inp.quantized_values.len(), shape, expected_len
                ),
            });
        }

        let names = DequantLinearNames::from_original(&inp.original_name);

        graph.initializer.push(
            build_quantized_weight_tensor(&names, &inp.quantized_values, shape),
        );
        graph.initializer.push(
            build_scale_tensor(&names, &inp.scales),
        );
        graph.initializer.push(
            build_zero_point_tensor(&names, &inp.zero_points),
        );

        dq_nodes.push(build_dequantize_linear_node(&names, inp.axis));
    }

    // -----------------------------------------------------------------------
    // 3.  Prepend DequantizeLinear nodes before all existing computation nodes.
    //     They must appear first so their outputs are "known" when the validator
    //     (or ONNX Runtime) walks the node list in order.
    // -----------------------------------------------------------------------
    let existing_nodes = std::mem::take(&mut graph.node);
    graph.node = dq_nodes;
    graph.node.extend(existing_nodes);

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_proto::{
        GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
        TensorProto, ValueInfoProto, tensor_proto,
    };

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Minimal graph: one graph input "input", one FP32 initializer "w" (shape [2,2]),
    /// one Conv node consuming both, producing "out".
    fn make_simple_graph() -> GraphProto {
        GraphProto {
            input: vec![ValueInfoProto { name: "input".to_string(), ..Default::default() }],
            initializer: vec![TensorProto {
                name:       "w".to_string(),
                data_type:  tensor_proto::DataType::Float as i32,
                dims:       vec![2, 2],
                float_data: vec![1.0, 2.0, 3.0, 4.0],
                ..Default::default()
            }],
            node: vec![NodeProto {
                op_type: "Conv".to_string(),
                name:    "conv0".to_string(),
                input:   vec!["input".to_string(), "w".to_string()],
                output:  vec!["out".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Two-weight graph: "w1" and "w2", two Conv nodes chained.
    fn make_two_weight_graph() -> GraphProto {
        GraphProto {
            input: vec![ValueInfoProto { name: "input".to_string(), ..Default::default() }],
            initializer: vec![
                TensorProto {
                    name:       "w1".to_string(),
                    data_type:  tensor_proto::DataType::Float as i32,
                    dims:       vec![2, 2],
                    float_data: vec![1.0, 2.0, 3.0, 4.0],
                    ..Default::default()
                },
                TensorProto {
                    name:       "w2".to_string(),
                    data_type:  tensor_proto::DataType::Float as i32,
                    dims:       vec![2, 2],
                    float_data: vec![5.0, 6.0, 7.0, 8.0],
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
                    output:  vec!["out".to_string()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------------
    // Connectivity validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_connectivity_passes_on_valid_graph() {
        let graph  = make_simple_graph();
        let report = validate_graph_connectivity(&graph);
        assert!(
            report.valid,
            "original graph should be valid; broken: {:?}",
            report.broken_refs
        );
    }

    #[test]
    fn test_connectivity_detects_renamed_initializer() {
        // Simulate the exact v0.2.0 bug: rename "w" in the initializer list
        // without updating the Conv node that references it.
        let mut graph = make_simple_graph();

        for init in graph.initializer.iter_mut() {
            if init.name == "w" {
                init.name = "w__qINT8_s0.00392_z-3_len4".to_string();
            }
        }

        let report = validate_graph_connectivity(&graph);
        assert!(!report.valid, "should detect broken reference to 'w'");
        assert_eq!(report.broken_refs.len(), 1);
        assert!(
            report.broken_refs[0].contains("'w'"),
            "error should mention 'w': {}",
            report.broken_refs[0]
        );
    }

    #[test]
    fn test_connectivity_detects_multiple_broken_refs() {
        let mut graph = make_two_weight_graph();

        for init in graph.initializer.iter_mut() {
            if init.name == "w1" {
                init.name = "w1_broken".to_string();
            } else if init.name == "w2" {
                init.name = "w2_broken".to_string();
            }
        }

        let report = validate_graph_connectivity(&graph);
        assert!(!report.valid);
        assert_eq!(report.broken_refs.len(), 2);
    }

    #[test]
    fn test_connectivity_summary_formatting() {
        let valid = ConnectivityReport {
            valid: true,
            broken_refs: vec![],
        };
        assert!(valid.summary().contains("OK"));

        let broken = ConnectivityReport {
            valid: false,
            broken_refs: vec!["Node 'x' → unknown input 'y'".to_string()],
        };
        let s = broken.summary();
        assert!(s.contains("BROKEN"));
        assert!(s.contains("1 dangling reference"));
        assert!(s.contains("unknown input 'y'"));
    }

    // -----------------------------------------------------------------------
    // Opset version tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ensure_opset_bumps_low_version() {
        let mut model = ModelProto {
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 10 }],
            ..Default::default()
        };

        ensure_opset_version(&mut model, 13);

        assert_eq!(model.opset_import[0].version, 13);
    }

    #[test]
    fn test_ensure_opset_leaves_sufficient_version() {
        let mut model = ModelProto {
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 17 }],
            ..Default::default()
        };

        ensure_opset_version(&mut model, 13);

        assert_eq!(model.opset_import[0].version, 17, "should not downgrade");
    }

    #[test]
    fn test_ensure_opset_adds_missing_default_domain() {
        let mut model = ModelProto::default();
        // No opset_import at all
        ensure_opset_version(&mut model, 13);

        assert_eq!(model.opset_import.len(), 1);
        assert!(model.opset_import[0].domain.is_empty());
        assert_eq!(model.opset_import[0].version, 13);
    }

    // -----------------------------------------------------------------------
    // QDQ transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_qdq_single_weight_produces_valid_graph() {
        let mut graph = make_simple_graph();

        let inputs = vec![QdqWeightInput {
            original_name:    "w".to_string(),
            quantized_values: vec![25, 51, 76, 102],
            scales:           vec![0.039_215_686], // ≈ 1/25.5
            zero_points:      vec![0],
            bits:             8,
            axis:             None,
        }];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        let report = validate_graph_connectivity(&graph);
        assert!(
            report.valid,
            "graph after QDQ must be valid; broken: {:?}",
            report.broken_refs
        );
    }

    #[test]
    fn test_qdq_adds_correct_initializers() {
        let mut graph = make_simple_graph();

        let inputs = vec![QdqWeightInput {
            original_name:    "w".to_string(),
            quantized_values: vec![10, 20, 30, 40],
            scales:           vec![0.1],
            zero_points:      vec![-5],
            bits:             8,
            axis:             None,
        }];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        let init_names: Vec<&str> = graph.initializer.iter().map(|i| i.name.as_str()).collect();

        assert!(init_names.contains(&"w_quantized"), "missing w_quantized");
        assert!(init_names.contains(&"w_scale"),     "missing w_scale");
        assert!(init_names.contains(&"w_zp"),        "missing w_zp");
        assert!(
            !init_names.contains(&"w"),
            "original FP32 'w' should be removed"
        );
    }

    #[test]
    fn test_qdq_node_order_dequant_first() {
        let mut graph = make_simple_graph();

        let inputs = vec![QdqWeightInput {
            original_name:    "w".to_string(),
            quantized_values: vec![10, 20, 30, 40],
            scales:           vec![0.1],
            zero_points:      vec![0],
            bits:             8,
            axis:             None,
        }];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        let ops: Vec<&str> = graph.node.iter().map(|n| n.op_type.as_str()).collect();

        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0], "DequantizeLinear");
        assert_eq!(ops[1], "Conv");
    }

    #[test]
    fn test_qdq_dequant_output_is_original_name() {
        let mut graph = make_simple_graph();

        let inputs = vec![QdqWeightInput {
            original_name:    "w".to_string(),
            quantized_values: vec![1, 2, 3, 4],
            scales:           vec![1.0],
            zero_points:      vec![0],
            bits:             8,
            axis:             None,
        }];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        let dq = &graph.node[0]; // first node = DequantizeLinear
        assert_eq!(dq.output[0], "w", "DequantizeLinear output must be original name");
    }

    #[test]
    fn test_qdq_two_weights_both_transformed() {
        let mut graph = make_two_weight_graph();

        let inputs = vec![
            QdqWeightInput {
                original_name:    "w1".to_string(),
                quantized_values: vec![10, 20, 30, 40],
                scales:           vec![0.1],
                zero_points:      vec![0],
                bits:             8,
                axis:             None,
            },
            QdqWeightInput {
                original_name:    "w2".to_string(),
                quantized_values: vec![50, 60, 70, 80],
                scales:           vec![0.2],
                zero_points:      vec![-1],
                bits:             8,
                axis:             None,
            },
        ];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        // Connectivity must still be valid
        let report = validate_graph_connectivity(&graph);
        assert!(report.valid, "two-weight graph broken: {:?}", report.broken_refs);

        // Should have 2 DequantizeLinear + 2 Conv = 4 nodes
        assert_eq!(graph.node.len(), 4);

        // First two nodes are DequantizeLinear
        assert_eq!(graph.node[0].op_type, "DequantizeLinear");
        assert_eq!(graph.node[1].op_type, "DequantizeLinear");

        // Their outputs are the original weight names
        let dq_outputs: Vec<&str> = graph.node.iter().take(2)
            .map(|n| n.output[0].as_str())
            .collect();
        assert!(dq_outputs.contains(&"w1"));
        assert!(dq_outputs.contains(&"w2"));
    }

    #[test]
    fn test_qdq_int4_values_stored_as_int8() {
        let mut graph = make_simple_graph();

        // INT4 range [-8, 7] — these arrive as i8 from ensure_unpacked()
        let inputs = vec![QdqWeightInput {
            original_name:    "w".to_string(),
            quantized_values: vec![-8, -1, 0, 7],
            scales:           vec![0.5],
            zero_points:      vec![0],
            bits:             4, // flag says INT4; storage must still be INT8
            axis:             None,
        }];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        let quant_init = graph
            .initializer
            .iter()
            .find(|i| i.name == "w_quantized")
            .expect("w_quantized not found");

        // Data type must be INT8 (ONNX DequantizeLinear requirement)
        assert_eq!(quant_init.data_type, tensor_proto::DataType::Int8 as i32);

        // Byte-level round-trip must be exact
        let recovered: Vec<i8> = quant_init.raw_data.iter().map(|&b| b as i8).collect();
        assert_eq!(recovered, vec![-8, -1, 0, 7]);
    }

    #[test]
    fn test_qdq_unknown_weight_returns_error() {
        let mut graph = make_simple_graph();

        let inputs = vec![QdqWeightInput {
            original_name:    "does_not_exist".to_string(),
            quantized_values: vec![1, 2, 3],
            scales:           vec![1.0],
            zero_points:      vec![0],
            bits:             8,
            axis:             None,
        }];

        let result = apply_qdq_transform(&mut graph, &inputs);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("does_not_exist"),
            "error should name the missing weight"
        );
    }

    #[test]
    fn test_qdq_non_quantized_initializers_preserved() {
        // Add an extra initializer "bias" that is NOT being quantized.
        // It must survive the transform untouched.
        let mut graph = make_simple_graph();

        graph.initializer.push(TensorProto {
            name:       "bias".to_string(),
            data_type:  tensor_proto::DataType::Float as i32,
            dims:       vec![2],
            float_data: vec![0.1, 0.2],
            ..Default::default()
        });

        // Also add "bias" as a Conv input so connectivity stays valid
        graph.node[0].input.push("bias".to_string());

        let inputs = vec![QdqWeightInput {
            original_name:    "w".to_string(),
            quantized_values: vec![10, 20, 30, 40],
            scales:           vec![0.1],
            zero_points:      vec![0],
            bits:             8,
            axis:             None,
        }];

        apply_qdq_transform(&mut graph, &inputs).expect("QDQ transform failed");

        // "bias" must still be present and untouched
        let bias_init = graph.initializer.iter().find(|i| i.name == "bias");

        assert!(bias_init.is_some(), "non-quantized 'bias' initializer must be preserved");
        assert!((bias_init.unwrap().float_data[0] - 0.1).abs() < 1e-6);

        // Full connectivity check
        let report = validate_graph_connectivity(&graph);
        assert!(report.valid, "broken: {:?}", report.broken_refs);
    }
}
