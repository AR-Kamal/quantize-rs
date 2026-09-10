"""Type stubs for the ``quantize_rs`` native extension module.

These describe the functions implemented in Rust (``src/python.rs``) so that
editors and type checkers (mypy / pyright) can offer completion and checking.

``quantize_with_calibration`` is present when the wheel is built with the
default ``calibration`` feature — which the published wheels are.
"""

from typing import Optional

__version__: str

class ModelInfo:
    """Structural metadata about an ONNX model (returned by :func:`model_info`)."""

    name: str
    version: int
    opset_version: int
    num_nodes: int
    inputs: list[str]
    outputs: list[str]

def quantize(
    input_path: str,
    output_path: str,
    bits: int = ...,
    per_channel: bool = ...,
    excluded_layers: Optional[list[str]] = ...,
    min_elements: int = ...,
    layer_bits: Optional[dict[str, int]] = ...,
    native_int4: bool = ...,
    symmetric: bool = ...,
) -> None:
    """Weight-based INT8/INT4 quantization of an ONNX model.

    Args:
        input_path: Path to the input ONNX model.
        output_path: Path to write the quantized model.
        bits: Bit width, ``8`` or ``4``.
        per_channel: Conv axis 0, MatMul last weight axis, Gemm axis 1 or 0
            according to transB. Conflicting shared uses are rejected.
        excluded_layers: Initializer names to leave in FP32.
        min_elements: Skip tensors with fewer than this many elements.
        layer_bits: Per-layer bit-width overrides, e.g. ``{"conv1.weight": 4}``.
        native_int4: Store INT4 as native ONNX ``DataType.Int4`` (opset 21).
        symmetric: Force ``zero_point == 0`` (symmetric quantization).
    """
    ...

def quantize_with_calibration(
    input_path: str,
    output_path: str,
    calibration_data: Optional[str] = ...,
    bits: int = ...,
    per_channel: bool = ...,
    method: str = ...,
    num_samples: int = ...,
    sample_shape: Optional[list[int]] = ...,
    native_int4: bool = ...,
    symmetric: bool = ...,
    excluded_layers: Optional[list[str]] = ...,
    min_elements: int = ...,
    layer_bits: Optional[dict[str, int]] = ...,
) -> None:
    """Static INT8 Conv activation QDQ with representative .npy data.

    Requires one fixed FP32 NCHW input, batch 1 and opset >= 13.
    Weights use their own ranges. Only selected Conv boundaries receive QDQ.
    INT4/native_int4 and missing calibration_data are rejected. num_samples is
    retained but unused; sample_shape, if supplied, must match the dataset.
    layer_bits accepts only 8; exclusions match Conv initializer names.
    """
    ...


def model_info(input_path: str) -> ModelInfo:
    """Return structural metadata (name, version, opset, nodes, inputs, outputs)."""
    ...
