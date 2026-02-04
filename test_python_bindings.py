# test_python_bindings.py
"""
Test suite for quantize-rs Python bindings.

Run with: pytest test_python_bindings.py -v
"""

import pytest
import quantize_rs
import numpy as np
import os
import tempfile

# Assume mnist.onnx exists in project root
MODEL_PATH = "mnist.onnx"

@pytest.fixture
def temp_output():
    """Create a temporary file for output"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.remove(f.name)

def test_model_info():
    """Test model_info() returns expected metadata"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    info = quantize_rs.model_info(MODEL_PATH)
    
    assert isinstance(info.name, str)
    assert info.num_nodes > 0
    assert len(info.inputs) > 0
    assert len(info.outputs) > 0
    
    print(f"Model: {info.name}, Nodes: {info.num_nodes}")

def test_basic_quantize(temp_output):
    """Test basic quantization without calibration"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    quantize_rs.quantize(
        input_path=MODEL_PATH,
        output_path=temp_output,
        bits=8,
        per_channel=False
    )
    
    assert os.path.exists(temp_output)
    assert os.path.getsize(temp_output) > 0
    
    # Verify it's smaller than original
    original_size = os.path.getsize(MODEL_PATH)
    quantized_size = os.path.getsize(temp_output)
    
    assert quantized_size < original_size
    print(f"Compression: {original_size / quantized_size:.2f}×")

def test_quantize_with_calibration_random(temp_output):
    """Test quantization with random calibration data"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    quantize_rs.quantize_with_calibration(
        input_path=MODEL_PATH,
        output_path=temp_output,
        calibration_data=None,  # Use random data
        bits=8,
        method="minmax",
        num_samples=10,  # Small for fast test
        sample_shape=[1, 28, 28]  # MNIST shape
    )
    
    assert os.path.exists(temp_output)
    assert os.path.getsize(temp_output) > 0

def test_quantize_with_calibration_npy(temp_output):
    """Test quantization with .npy calibration data"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    # Create dummy calibration data
    calib_data = np.random.randn(10, 1, 28, 28).astype(np.float32)
    calib_path = "test_calibration.npy"
    np.save(calib_path, calib_data)
    
    try:
        quantize_rs.quantize_with_calibration(
            input_path=MODEL_PATH,
            output_path=temp_output,
            calibration_data=calib_path,
            bits=8,
            method="percentile"
        )
        
        assert os.path.exists(temp_output)
        assert os.path.getsize(temp_output) > 0
    finally:
        # Cleanup
        if os.path.exists(calib_path):
            os.remove(calib_path)

def test_int4_quantization(temp_output):
    """Test INT4 quantization"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    quantize_rs.quantize(
        input_path=MODEL_PATH,
        output_path=temp_output,
        bits=4,
        per_channel=True
    )
    
    assert os.path.exists(temp_output)

def test_invalid_model_path():
    """Test error handling for non-existent model"""
    with pytest.raises(Exception) as exc_info:
        quantize_rs.quantize(
            input_path="nonexistent.onnx",
            output_path="output.onnx"
        )
    
    assert "Failed to load model" in str(exc_info.value)

def test_invalid_calibration_method(temp_output):
    """Test error handling for invalid calibration method"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    with pytest.raises(Exception) as exc_info:
        quantize_rs.quantize_with_calibration(
            input_path=MODEL_PATH,
            output_path=temp_output,
            method="invalid_method"
        )
    
    assert "Unknown method" in str(exc_info.value)

def test_onnxruntime_compatibility(temp_output):
    """Test that quantized model loads in ONNX Runtime"""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"{MODEL_PATH} not found")
    
    # Quantize
    quantize_rs.quantize(
        input_path=MODEL_PATH,
        output_path=temp_output,
        bits=8
    )
    
    # Try to load with ONNX Runtime
    session = ort.InferenceSession(temp_output)
    
    # Verify input/output metadata accessible
    assert len(session.get_inputs()) > 0
    assert len(session.get_outputs()) > 0
    
    # Try inference
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Create dummy input matching the shape
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    output = session.run(None, {input_name: dummy_input})
    
    assert len(output) > 0
    print(f"✓ Inference works! Output shape: {output[0].shape}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])