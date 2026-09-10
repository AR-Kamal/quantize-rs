"""Regression coverage for strict export parity and default-runtime quality gates."""
import unittest

import numpy as np

from operator_axes_smoke_test import check_outputs


class OutputChecksTest(unittest.TestCase):
    def setUp(self):
        self.reference = np.ones((2, 5), dtype=np.float32)

    def check(self, *, plain=None, strict=None, optimized=None, bits=8):
        check_outputs(
            self.reference,
            self.reference if plain is None else plain,
            self.reference if strict is None else strict,
            self.reference if optimized is None else optimized,
            bits, "regression fixture",
        )

    def test_default_runtime_can_add_small_quantization_error(self):
        self.check(optimized=self.reference * 1.01)

    def test_strict_parity_still_rejects_small_differences(self):
        with self.assertRaises(AssertionError):
            self.check(strict=self.reference * 1.01)

    def test_quality_gate_applies_to_every_execution_mode(self):
        for bits, factor in [(8, 1.04), (4, 1.36)]:
            for mode in ["plain", "strict", "optimized"]:
                with self.subTest(bits=bits, mode=mode):
                    with self.assertRaisesRegex(AssertionError, "relative RMSE"):
                        self.check(**{mode: self.reference * factor}, bits=bits)

    def test_nonfinite_outputs_are_rejected_in_every_mode(self):
        for mode in ["plain", "strict", "optimized"]:
            for value in [np.nan, np.inf]:
                with self.subTest(mode=mode, value=value):
                    with self.assertRaisesRegex(AssertionError, "nonfinite"):
                        self.check(**{mode: np.full_like(self.reference, value)})

    def test_broadcastable_wrong_output_shape_is_rejected(self):
        for mode in ["plain", "strict", "optimized"]:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(AssertionError, "output shape"):
                    self.check(**{mode: self.reference[:1]})


if __name__ == "__main__":
    unittest.main()
