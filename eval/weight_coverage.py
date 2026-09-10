"""Read-only coverage audit for fresh FP32 graphs; verify against actual Rust QDQ output.

This explains direct uses, not full ONNX validation or constant-expression evaluation.
The CLI remains authoritative: an audit candidate becomes selected only after export.
"""
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def digest(tensor):
    return hashlib.sha256(tensor.SerializeToString()).hexdigest()


def standard(node):
    return node.domain in ("", "ai.onnx")


def supported(node, slot):
    return standard(node) and slot == 1 and node.op_type in ("Conv", "MatMul", "Gemm")


def axis_for(node, rank):
    if node.op_type == "Conv":
        return 0
    if node.op_type == "MatMul":
        return rank - 1
    attrs = [a for a in node.attribute if a.name == "transB"]
    if rank != 2 or len(attrs) > 1 or any(a.type != onnx.AttributeProto.INT or a.i not in (0, 1) for a in attrs):
        raise ValueError("invalid Gemm rank/transB")
    return 0 if attrs and attrs[0].i else 1


def audit(model, min_elements=0, excluded=()):
    """Include every initializer and every matrix operator's right operand."""
    graph = model.graph
    uses, producers = defaultdict(list), {}
    initializers = {t.name: t for t in graph.initializer}
    if len(initializers) != len(graph.initializer):
        raise ValueError("Duplicate initializer names")
    for index, node in enumerate(graph.node):
        for slot, name in enumerate(node.input):
            uses[name].append((index, node, slot))
        for name in node.output:
            producers[name] = node
    if any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in graph.node):
        raise ValueError("Coverage evaluation requires a fresh FP32 source without existing QDQ")
    subgraphs = any(a.HasField("g") or a.graphs for n in graph.node for a in n.attribute)
    output_names = {v.name for v in graph.output}
    occupied = set(initializers) | {v.name for v in list(graph.input) + list(graph.output) + list(graph.value_info)}
    occupied |= {s for n in graph.node for s in list(n.input) + list(n.output)}
    node_names = {n.name for n in graph.node}
    rows = []
    for tensor in graph.initializer:
        name, shape = tensor.name, list(tensor.dims)
        elements = math.prod(shape)
        consumers = uses[name]
        row = dict(name=name, shape=shape, elements=elements, dtype=TensorProto.DataType.Name(tensor.data_type),
                   fp32_bytes=elements * 4 if tensor.data_type == TensorProto.FLOAT else 0,
                   source_tensor_sha256=digest(tensor), status="skipped", axis=None,
                   consumers=[dict(node=n.name or f"node_{i}", operator=n.op_type, domain=n.domain or "ai.onnx", input_index=s)
                              for i, n, s in consumers])
        reason = None
        if tensor.data_type != TensorProto.FLOAT:
            reason = "not_fp32"
        elif len(shape) < 2:
            reason = "rank_below_2"
        elif name in excluded:
            reason = "excluded"
        elif name.startswith("_quantize_rs_") or (name.endswith("_scale") and name[:-6] + "_quantized" in initializers):
            reason = "quantization_scaffolding"
        elif not consumers:
            reason = "unused"
        elif not any(supported(n, s) for _, n, s in consumers):
            if any(standard(n) and n.op_type in ("Transpose", "Reshape", "Identity") and s == 0 for _, n, s in consumers):
                reason = "indirect_or_transformed_use"
            elif all(n.op_type == "Gather" and s == 0 and standard(n) for _, n, s in consumers):
                reason = "embedding_or_lookup"
            else:
                reason = "no_supported_direct_weight_use"
        elif any(d <= 0 for d in shape):
            reason, row["status"] = "invalid_shape", "blocked"
        elif elements < min_elements:
            reason = "below_min_elements"
        else:
            row["status"] = "candidate"
            axes = []
            for _, node, slot in consumers:
                if not supported(node, slot):
                    reason = "unsupported_shared_use"
                    continue
                try:
                    axes.append(axis_for(node, len(shape)))
                except ValueError:
                    reason = "invalid_gemm_attributes"
            row["consumer_axes"] = axes
            if len(set(axes)) > 1:
                reason = "conflicting_axes"
            row["axis"] = axes[0] if axes and len(set(axes)) == 1 else None
            if subgraphs:
                reason = "subgraph_capture_unsupported"
            elif name in output_names:
                reason = "weight_is_graph_output"
            elif tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
                reason = "external_weight"
            elif any(name + suffix in occupied for suffix in ("_quantized", "_scale", "_zp")) or "DequantizeLinear_" + name in node_names:
                reason = "generated_name_collision"
            if reason:
                row["status"] = "blocked"
        row["reason"] = reason or "supported_direct_weight"
        rows.append(row)

    def origin(name):
        visited, chain = set(), []
        while name not in initializers and name not in visited:
            visited.add(name)
            node = producers.get(name)
            if node is None or not standard(node) or node.op_type not in ("Transpose", "Reshape", "Identity") or not node.input:
                return dict(kind="dynamic_or_unresolved", initializer=None, transforms=chain)
            chain.append(node.op_type)
            name = node.input[0]
        return dict(kind="indirect_initializer" if chain else "direct_initializer", initializer=name, transforms=chain) if name in initializers else dict(kind="unresolved", initializer=None, transforms=chain)

    operators = [dict(node=n.name or f"node_{i}", operator=n.op_type, rhs=n.input[1], **origin(n.input[1]))
                 for i, n in enumerate(graph.node) if standard(n) and n.op_type in ("Conv", "MatMul", "Gemm") and len(n.input) > 1]
    report = dict(scope="direct-use diagnostic; actual selection verified against CLI export", initializers=rows, matrix_operators=operators)
    summarize(report)
    return report


def summarize(report):
    rows = report["initializers"]
    matrices = [r for r in rows if r["dtype"] == "FLOAT" and len(r["shape"]) >= 2]
    report["summary"] = dict(
        initializer_count=len(rows), status_counts=dict(Counter(r["status"] for r in rows)),
        skipped_reasons=dict(Counter(r["reason"] for r in rows if r["status"] == "skipped")),
        fp32_matrix_elements=sum(r["elements"] for r in matrices),
        fp32_matrix_bytes=sum(r["fp32_bytes"] for r in matrices),
        selected_elements=sum(r["elements"] for r in rows if r["status"] == "selected"),
        selected_fp32_bytes=sum(r["fp32_bytes"] for r in rows if r["status"] == "selected"),
        selected_axis_counts=dict(Counter(str(r["axis"]) for r in rows if r["status"] == "selected")),
        matrix_rhs_counts=dict(Counter(r["kind"] for r in report["matrix_operators"])),
    )


def verify_export(report, source, output):
    """Require exact agreement between the diagnostic and real symmetric INT8 QDQ."""
    onnx.checker.check_model(output)
    if [v.SerializeToString() for v in source.graph.output] != [v.SerializeToString() for v in output.graph.output]:
        raise ValueError("Graph output interface changed")
    expected = {r["name"] for r in report["initializers"] if r["status"] == "candidate"}
    dqs = [n for n in output.graph.node if n.op_type == "DequantizeLinear" and standard(n)]
    if len(dqs) != len(expected) or {n.output[0] for n in dqs} != expected:
        raise ValueError("Coverage audit disagrees with actual CLI-selected weights")
    tensors = {t.name: t for t in output.graph.initializer}
    metadata = {p.key: p.value for p in output.metadata_props}
    by_name = {n.output[0]: n for n in dqs}
    for row in report["initializers"]:
        name = row["name"]
        if name not in expected:
            if name not in tensors or digest(tensors[name]) != row["source_tensor_sha256"]:
                raise ValueError(f"Unselected initializer changed: {name}")
            continue
        node = by_name[name]
        if metadata.get("quantize_rs.bits." + name, "8") != "8":
            raise ValueError(f"Expected INT8; stored quantizer bit width differs: {name}")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("axis") != row["axis"] or name in tensors:
            raise ValueError(f"Wrong exported axis/replacement: {name}")
        values, scale, zero = [tensors[s] for s in node.input]
        scales, zeros = numpy_helper.to_array(scale), numpy_helper.to_array(zero)
        if (values.data_type != TensorProto.INT8 or zero.data_type != TensorProto.INT8
                or list(values.dims) != row["shape"] or scales.shape != (row["shape"][row["axis"]],)
                or zeros.shape != scales.shape or not np.all(zeros == 0)
                or not np.all(np.isfinite(scales) & (scales > 0))):
            raise ValueError(f"Invalid symmetric per-channel INT8 payload: {name}")
        row["status"] = "selected"
    if not expected:
        raise ValueError("No weights selected; refusing a vacuous quality comparison")
    report["verified_against_cli_output"] = True
    summarize(report)
