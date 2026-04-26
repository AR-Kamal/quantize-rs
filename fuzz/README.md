# Fuzzing quantize-rs

Uses [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html) to
stress-test the ONNX protobuf decoder. `OnnxModel::load` accepts
user-supplied `.onnx` files, so catching panics in the parse path is
worth the minimal setup here.

## One-time setup

```
cargo install cargo-fuzz
```

Fuzzing requires a nightly toolchain:

```
rustup toolchain install nightly
```

## Run the target

From the repo root:

```
cargo +nightly fuzz run onnx_load
```

This runs indefinitely; press Ctrl-C to stop. Crash artifacts land in
`fuzz/artifacts/onnx_load/`, and the accumulated corpus in
`fuzz/corpus/onnx_load/`.

## Adding targets

Each `fuzz_targets/<name>.rs` becomes a binary target named `<name>`.
Register it in `fuzz/Cargo.toml` under `[[bin]]` and run
`cargo +nightly fuzz run <name>`.
