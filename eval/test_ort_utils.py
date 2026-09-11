"""Numerical regression for optimized INT8 Conv on CPUs without VNNI."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from ort_utils import cpu_session_options


class CpuQuantizationPrecisionTests(unittest.TestCase):
    def test_large_dot_product_matches_qdq_reference(self):
        # Two unsigned activations times signed weights exceed 32767 in the
        # AVX2 U8S8 kernel's intermediate sum, despite a small final FP32 value.
        graph = helper.make_graph(
            [helper.make_node("QuantizeLinear", ["X", "xs", "xz"], ["Xq"]),
             helper.make_node("DequantizeLinear", ["Xq", "xs", "xz"], ["Xdq"]),
             helper.make_node("DequantizeLinear", ["W", "ws", "wz"], ["Wdq"], axis=0),
             helper.make_node("Conv", ["Xdq", "Wdq"], ["H"], kernel_shape=[1, 1]),
             helper.make_node("QuantizeLinear", ["H", "ys", "yz"], ["Yq"]),
             helper.make_node("DequantizeLinear", ["Yq", "ys", "yz"], ["Y"])],
            "avx2_saturation",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 1, 1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 1])],
            [numpy_helper.from_array(np.array(value, dtype=dtype), name)
             for name, value, dtype in [
                 ("xs", .01, np.float32), ("xz", -1, np.int8),
                 ("W", [[[[127]], [[127]]]], np.int8),
                 ("ws", [.01], np.float32), ("wz", [0], np.int8),
                 ("ys", .05, np.float32), ("yz", -1, np.int8)]]
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        with tempfile.TemporaryDirectory() as folder:
            optimized_path = Path(folder) / "optimized.onnx"
            options = cpu_session_options()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.optimized_model_filepath = str(optimized_path)
            optimized = ort.InferenceSession(model.SerializeToString(), options,
                                             providers=["CPUExecutionProvider"])
            self.assertIn("QLinearConv", [n.op_type for n in onnx.load(optimized_path).graph.node])
            options.optimized_model_filepath = ""
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            plain = ort.InferenceSession(model.SerializeToString(), options,
                                         providers=["CPUExecutionProvider"])
            for value in (1.0, -1.0, 1.2):
                with self.subTest(input=value):
                    sample = np.full((1, 2, 1, 1), value, np.float32)
                    # Independent QDQ calculation, including clipping and ties-to-even.
                    quantized = np.clip(np.rint(sample / np.float32(.01)) - 1, -128, 127)
                    dequantized = (quantized + 1) * np.float32(.01)
                    convolution = dequantized.sum(axis=1, keepdims=True) * np.float32(1.27)
                    output_q = np.clip(np.rint(convolution / np.float32(.05)) - 1, -128, 127)
                    expected = (output_q + 1) * np.float32(.05)
                    for session in (plain, optimized):
                        np.testing.assert_allclose(session.run(None, {"X": sample})[0], expected,
                                                   atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
