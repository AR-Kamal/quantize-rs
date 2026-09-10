"""Small, offline regressions for coverage and the evaluation math."""
import copy
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from gpt2_evaluation import asset, negative_log_likelihood, quality_gate, windows
from weight_coverage import audit, verify_export


def model(nodes, tensors):
    graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])],
                              [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
                              [numpy_helper.from_array(data, name) for name, data in tensors])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


class EvaluationMathTests(unittest.TestCase):
    def test_every_target_counted_once_with_overlap_and_partial_final_windows(self):
        for length in (2, 7, 8, 9, 15, 16, 17, 33):
            for context, stride in ((2, 1), (8, 3), (8, 7)):
                ids = np.arange(length)
                seen, total = [], 0.0
                for chunk, first in windows(ids, context, stride):
                    seen.extend(chunk[first:])
                    # Uniform logits with a large common offset exercise stable logsumexp.
                    logits = np.full((1, len(chunk), length), 10000.0)
                    nll, count = negative_log_likelihood(logits, chunk, first)
                    self.assertEqual(count, len(chunk) - first)
                    total += nll
                self.assertEqual(seen, list(range(1, length)))
                self.assertAlmostEqual(math.exp(total / (length - 1)), length, places=8)

    def test_shifted_labels_and_context_only_targets(self):
        ids = np.array([0, 1, 0, 1])
        logits = np.array([[[0., math.log(3)], [math.log(3), 0.], [0., math.log(3)], [99., -99.]]])
        nll, count = negative_log_likelihood(logits, ids, 2)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(nll, -2 * math.log(.75))
        with self.assertRaises(ValueError):
            negative_log_likelihood(logits * np.nan, ids, 1)
        with self.assertRaises(ValueError):
            list(windows(ids, 3, 3))

    def test_gate_rejects_degradation_invalid_baseline_and_nonfinite_values(self):
        self.assertTrue(quality_gate({"perplexity": 40}, {"perplexity": 41})["passed"])
        self.assertFalse(quality_gate({"perplexity": 40}, {"perplexity": 42})["passed"])
        self.assertFalse(quality_gate({"perplexity": 101}, {"perplexity": 101})["passed"])
        with self.assertRaises(ValueError):
            quality_gate({"perplexity": float("nan")}, {"perplexity": 40})

    def test_cached_asset_checksum_is_enforced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "vocab.json").write_text("wrong contents")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                asset(path, "vocab.json")

    def test_missing_dataset_fails_without_substituting_texts(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with patch.dict("os.environ", {"HF_HUB_CACHE": str(path / "empty-cache")}):
                with self.assertRaisesRegex(ValueError, "No fallback corpus"):
                    asset(path / "assets", "test.parquet")


class CoverageTests(unittest.TestCase):
    def test_roles_transpose_conflicts_exclusions_and_indirect_rhs(self):
        source = model([
            helper.make_node("MatMul", ["X", "W"], ["Y"]),
            helper.make_node("Gemm", ["X", "W"], ["other"], transB=1),
            helper.make_node("Gather", ["E", "idx"], ["embedding"]),
            helper.make_node("Transpose", ["T"], ["Tt"]),
            helper.make_node("MatMul", ["X", "Tt"], ["indirect"]),
        ], [(name, np.ones(shape, np.float32)) for name, shape in (("W", (3, 2)), ("E", (4, 3)), ("T", (2, 3)), ("bias", (2,)))])
        report = audit(source)
        rows = {r["name"]: r for r in report["initializers"]}
        self.assertEqual(rows["W"]["reason"], "conflicting_axes")
        self.assertEqual(rows["W"]["status"], "blocked")
        self.assertEqual(rows["E"]["reason"], "embedding_or_lookup")
        self.assertEqual(rows["T"]["reason"], "indirect_or_transformed_use")
        self.assertEqual(report["matrix_operators"][-1]["initializer"], "T")
        self.assertEqual(audit(source, excluded=["W"])["initializers"][0]["reason"], "excluded")
        self.assertEqual(audit(source, min_elements=7)["initializers"][0]["reason"], "below_min_elements")
        source.graph.node[1].attribute[0].i = 0
        self.assertEqual(audit(source)["initializers"][0]["axis"], 1)

    def test_export_verification_catches_axis_drift_and_unselected_mutation(self):
        source = model([helper.make_node("MatMul", ["X", "W"], ["Y"])],
                       [("W", np.ones((3, 2), np.float32)), ("embedding", np.ones((2, 2), np.float32))])
        output = copy.deepcopy(source)
        del output.graph.initializer[0]
        output.graph.initializer.extend([numpy_helper.from_array(np.ones((3, 2), np.int8), "W_quantized"),
                                         numpy_helper.from_array(np.ones(2, np.float32), "W_scale"),
                                         numpy_helper.from_array(np.zeros(2, np.int8), "W_zp")])
        output.graph.node.insert(0, helper.make_node("DequantizeLinear", ["W_quantized", "W_scale", "W_zp"], ["W"], axis=1))
        report = audit(source)
        verify_export(report, source, output)
        self.assertEqual(report["summary"]["selected_elements"], 6)
        output.graph.node[0].attribute[0].i = 0
        with self.assertRaises(ValueError):
            verify_export(audit(source), source, output)
        output.graph.node[0].attribute[0].i = 1
        prop = output.metadata_props.add()
        prop.key, prop.value = "quantize_rs.bits.W", "4"
        with self.assertRaisesRegex(ValueError, "Expected INT8"):
            verify_export(audit(source), source, output)
        prop.value = "8"
        output.graph.initializer[0].raw_data = np.zeros((2, 2), np.float32).tobytes()
        with self.assertRaisesRegex(ValueError, "Unselected"):
            verify_export(audit(source), source, output)


if __name__ == "__main__":
    unittest.main()
