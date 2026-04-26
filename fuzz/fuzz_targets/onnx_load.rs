#![no_main]
//! Fuzz target for `OnnxModel::from_bytes`.
//!
//! Feeds arbitrary byte sequences to the ONNX protobuf decoder to catch
//! panics, unchecked unwraps, or unbounded allocations in the parsing path.
//! `OnnxModel::from_bytes` delegates to prost's generated decoder for
//! `ModelProto`; any panic surfaced here is a library bug.

use libfuzzer_sys::fuzz_target;
use quantize_rs::onnx_utils::OnnxModel;

fuzz_target!(|data: &[u8]| {
    // A Result is fine — we only care that parsing does not panic.
    let _ = OnnxModel::from_bytes(data);
});
