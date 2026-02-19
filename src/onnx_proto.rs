//! Prost-generated ONNX protobuf types.
//!
//! These types are generated at build time from `proto/onnx.proto3` using
//! [`prost-build`](https://docs.rs/prost-build) and
//! [`protox`](https://docs.rs/protox) (a pure-Rust protobuf compiler,
//! eliminating the `protoc` system dependency).
//!
//! # Usage
//!
//! ```rust,ignore
//! use quantize_rs::onnx_proto::{ModelProto, tensor_proto};
//! use prost::Message;
//!
//! let proto = ModelProto::decode(bytes.as_ref()).unwrap();
//! let graph = proto.graph.as_ref().unwrap();
//! ```

// Include prost-generated code.  The generated file is named after the proto
// package (`onnx`), so cargo places it at `$OUT_DIR/onnx.rs`.
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
