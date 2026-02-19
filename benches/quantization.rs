//! Criterion benchmarks for quantize-rs.
//!
//! Run with: `cargo bench`
//!
//! # Benchmark groups
//!
//! 1. **quantize_throughput** — INT8 and INT4 per-tensor quantization at
//!    1 K, 100 K, and 1 M elements; reported as elements/second.
//!
//! 2. **per_channel_vs_per_tensor** — four variants on a [64, 27] (1 728-element)
//!    tensor: INT8 per-tensor, INT8 per-channel, INT4 per-tensor, INT4 per-channel.
//!
//! 3. **pack_int4** — raw `pack_int4` and `unpack_int4` throughput at
//!    10 K, 100 K, and 1 M elements.
//!
//! 4. **quantize_model** — full `Quantizer` loop over 8 synthetic weight
//!    tensors (no I/O) for both INT8 and INT4.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quantize_rs::quantization::{QuantizedTensor, QuantizedTensorInt4};
use quantize_rs::{pack_int4, unpack_int4, QuantConfig, Quantizer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic `Vec<f32>` of `n` elements in the range [-1, 1].
fn synthetic_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as f32) / (n as f32);
            (t * std::f32::consts::TAU).sin()
        })
        .collect()
}

/// Generate a `Vec<i8>` of `n` INT4-valid values (`[-8, 7]`).
fn synthetic_i8_int4(n: usize) -> Vec<i8> {
    (0..n).map(|i| ((i % 16) as i8) - 8).collect()
}

// ---------------------------------------------------------------------------
// Group 1 — Quantize throughput
// ---------------------------------------------------------------------------

fn bench_quantize_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_throughput");

    for &n in &[1_000_usize, 100_000, 1_000_000] {
        let data = synthetic_f32(n);
        let shape = vec![n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("int8", n), &data, |b, d| {
            b.iter(|| {
                QuantizedTensor::from_f32(black_box(d), black_box(shape.clone())).unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("int4", n), &data, |b, d| {
            b.iter(|| {
                QuantizedTensorInt4::from_f32(black_box(d), black_box(shape.clone())).unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2 — Per-channel vs per-tensor
// ---------------------------------------------------------------------------

fn bench_per_channel_vs_per_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_channel_vs_per_tensor");

    // Typical weight tensor shape: 64 output channels × 27 values each.
    let channels = 64_usize;
    let per_channel = 27_usize;
    let n = channels * per_channel;
    let data = synthetic_f32(n);
    let shape = vec![channels, per_channel];

    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("int8_per_tensor", |b| {
        b.iter(|| {
            QuantizedTensor::from_f32(black_box(&data), black_box(shape.clone())).unwrap()
        });
    });

    group.bench_function("int8_per_channel", |b| {
        b.iter(|| {
            QuantizedTensor::from_f32_per_channel(black_box(&data), black_box(shape.clone()))
                .unwrap()
        });
    });

    group.bench_function("int4_per_tensor", |b| {
        b.iter(|| {
            QuantizedTensorInt4::from_f32(black_box(&data), black_box(shape.clone())).unwrap()
        });
    });

    group.bench_function("int4_per_channel", |b| {
        b.iter(|| {
            QuantizedTensorInt4::from_f32_per_channel(black_box(&data), black_box(shape.clone()))
                .unwrap()
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3 — pack_int4 / unpack_int4
// ---------------------------------------------------------------------------

fn bench_pack_int4(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_int4");

    for &n in &[10_000_usize, 100_000, 1_000_000] {
        let values = synthetic_i8_int4(n);
        let packed = pack_int4(&values);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("pack", n), &values, |b, v| {
            b.iter(|| pack_int4(black_box(v)));
        });

        group.bench_with_input(BenchmarkId::new("unpack", n), &packed, |b, p| {
            b.iter(|| unpack_int4(black_box(p), black_box(n)));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4 — Full Quantizer loop (no I/O)
// ---------------------------------------------------------------------------

fn bench_quantize_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_model");

    // 8 synthetic weight tensors with shapes mimicking a small conv model.
    let shapes: &[&[usize]] = &[
        &[64, 3, 3, 3],    // conv1 weight  (1 728)
        &[64],             // conv1 bias     (64)
        &[128, 64, 3, 3],  // conv2 weight  (73 728)
        &[128],            // conv2 bias    (128)
        &[256, 128, 3, 3], // conv3 weight  (294 912)
        &[256],            // conv3 bias    (256)
        &[1024, 4096],     // fc1 weight    (4 194 304)
        &[1024],           // fc1 bias      (1 024)
    ];

    // Pre-generate data for each tensor.
    let tensors: Vec<(Vec<f32>, Vec<usize>)> = shapes
        .iter()
        .map(|&s| {
            let n: usize = s.iter().product();
            (synthetic_f32(n), s.to_vec())
        })
        .collect();

    let total_elements: u64 = tensors.iter().map(|(d, _)| d.len() as u64).sum();
    group.throughput(Throughput::Elements(total_elements));

    group.bench_function("int8", |b| {
        let cfg = QuantConfig {
            bits: 8,
            per_channel: false,
            ..Default::default()
        };
        let quantizer = Quantizer::new(cfg);
        b.iter(|| {
            for (data, shape) in black_box(&tensors) {
                let _ = quantizer
                    .quantize_tensor(black_box(data), black_box(shape.clone()))
                    .unwrap();
            }
        });
    });

    group.bench_function("int4", |b| {
        let cfg = QuantConfig {
            bits: 4,
            per_channel: false,
            ..Default::default()
        };
        let quantizer = Quantizer::new(cfg);
        b.iter(|| {
            for (data, shape) in black_box(&tensors) {
                let _ = quantizer
                    .quantize_tensor(black_box(data), black_box(shape.clone()))
                    .unwrap();
            }
        });
    });

    group.bench_function("int8_per_channel", |b| {
        let cfg = QuantConfig {
            bits: 8,
            per_channel: true,
            ..Default::default()
        };
        let quantizer = Quantizer::new(cfg);
        b.iter(|| {
            for (data, shape) in black_box(&tensors) {
                let _ = quantizer
                    .quantize_tensor(black_box(data), black_box(shape.clone()))
                    .unwrap();
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_quantize_throughput,
    bench_per_channel_vs_per_tensor,
    bench_pack_int4,
    bench_quantize_model,
);
criterion_main!(benches);
