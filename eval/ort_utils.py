"""ONNX Runtime CPU settings for static quantization evaluation."""
import onnxruntime as ort

CPU_SESSION_CONFIG = {"session.x64quantprecision": "1"}


def cpu_session_options():
    """Keep optimized INT8 math accurate on x64 CPUs without VNNI.

    ORT may lower signed QDQ to U8S8 kernels whose intermediate 16-bit sums
    saturate on AVX2/AVX512 CPUs without VNNI. Its precision mode selects
    overflow-safe U8U8 execution on affected CPUs; graph fusion stays enabled.
    https://github.com/microsoft/onnxruntime/blob/v1.29.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
    """
    options = ort.SessionOptions()
    for key, value in CPU_SESSION_CONFIG.items():
        options.add_session_config_entry(key, value)
    return options
