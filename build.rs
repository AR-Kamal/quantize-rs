fn main() {
    // Compile the vendored ONNX proto3 file using protox (pure-Rust protoc
    // replacement — no system `protoc` binary required) and prost-build.
    let file_descriptor_set = protox::compile(["proto/onnx.proto3"], ["proto/"])
        .expect("failed to compile ONNX proto3 file");

    prost_build::Config::new()
        .compile_fds(file_descriptor_set)
        .expect("failed to generate prost bindings from ONNX proto3");

    // Re-run if the proto file changes.
    println!("cargo:rerun-if-changed=proto/onnx.proto3");
}
