//! Property-based tests for quantization correctness.
//!
//! Four property families are verified:
//!
//! 1. **Clamp safety** — `quantize(v)` is always within `[QMIN, QMAX]` for
//!    any finite input, for both INT8 and INT4.
//!
//! 2. **Round-trip accuracy** — for any value `v` inside the quantization
//!    range, `|dequantize(quantize(v)) - v| <= scale + ε`.  The bound is
//!    `scale/2` from rounding plus a small floating-point epsilon.
//!
//! 3. **Pack / unpack identity** — `unpack_int4(pack_int4(data), len) == data`
//!    for any non-empty slice of valid INT4 values (`[-8, 7]`), and the packed
//!    size is always `ceil(len / 2)`.
//!
//! 4. **No panic** — `from_f32`, `from_f32_with_range`, and
//!    `from_f32_per_channel` never panic for any finite input; they either
//!    succeed or return a typed `QuantizeError`.

use proptest::prelude::*;
use quantize_rs::quantization::{
    QuantParams, QuantParamsInt4, QuantizedTensor, QuantizedTensorInt4,
};
use quantize_rs::{pack_int4, unpack_int4};

// ---------------------------------------------------------------------------
// Shared strategies
// ---------------------------------------------------------------------------

/// A (min, max) pair that always spans zero and has a non-trivial width.
///
/// `from_range` forces `min = min.min(0)` and `max = max.max(0)`, so using a
/// range that already satisfies `min <= 0 <= max` gives predictable behaviour
/// without any implicit widening in the tests.
fn spanning_range() -> impl Strategy<Value = (f32, f32)> {
    (
        (-1e6_f32..=-1e-3_f32), // min: strictly negative
        (1e-3_f32..=1e6_f32),   // max: strictly positive
    )
}

/// An `i8` value in the valid INT4 domain `[-8, 7]`.
fn int4_value() -> impl Strategy<Value = i8> {
    -8_i8..=7_i8
}

/// A non-empty `Vec<i8>` whose every element is in `[-8, 7]`.
fn int4_vec() -> impl Strategy<Value = Vec<i8>> {
    prop::collection::vec(int4_value(), 1..=512)
}

/// A non-empty `Vec<f32>` with finite values in a moderate range.
fn finite_f32_vec() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1e6_f32..=1e6_f32, 1..=512)
}

// ---------------------------------------------------------------------------
// Property 1 — Clamp safety
//
// No matter what finite value is fed to `quantize`, the result is always
// within the representable integer range for the target type.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_int8_quantize_always_in_range(
        (min, max) in spanning_range(),
        v in -1e6_f32..=1e6_f32,
    ) {
        let params = QuantParams::from_range(min, max);
        let q = params.quantize(v);
        // i8 is already bounded to [-128, 127] by the type, so we only need
        // to assert the lower bound is respected (upper is guaranteed by i8).
        prop_assert!(q >= -128, "INT8 result {} < -128 (min={}, max={}, v={})", q, min, max, v);
        let _ = q; // q <= 127 is always true for i8
    }

    #[test]
    fn prop_int4_quantize_always_in_range(
        (min, max) in spanning_range(),
        v in -1e6_f32..=1e6_f32,
    ) {
        let params = QuantParamsInt4::from_range(min, max);
        let q = params.quantize(v);
        prop_assert!(q >= -8, "INT4 result {} < -8 (min={}, max={}, v={})", q, min, max, v);
        prop_assert!(q <= 7,  "INT4 result {} > 7  (min={}, max={}, v={})", q, min, max, v);
    }
}

// ---------------------------------------------------------------------------
// Property 2 — Round-trip accuracy
//
// For a value v inside [min, max], the reconstruction error after
// quantize → dequantize is at most one quantization step.
//
// Derivation: dequantize(quantize(v)) = round(v/scale) * scale, so
//   |v - round(v/scale)*scale| = |v/scale - round(v/scale)| * scale <= 0.5 * scale.
// An extra `scale * 0.5` margin covers floating-point arithmetic noise.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_int8_round_trip_error_bounded(
        (min, max) in spanning_range(),
        t in 0.0_f32..=1.0_f32,
    ) {
        let v = min + t * (max - min);
        let params = QuantParams::from_range(min, max);
        let q  = params.quantize(v);
        let dq = params.dequantize(q);
        let error = (v - dq).abs();
        let bound = params.scale() + 1e-4;
        prop_assert!(
            error <= bound,
            "INT8 round-trip error {:.9} > bound {:.9}  \
             (min={}, max={}, v={}, q={}, dq={}, scale={})",
            error, bound, min, max, v, q, dq, params.scale()
        );
    }

    #[test]
    fn prop_int4_round_trip_error_bounded(
        (min, max) in spanning_range(),
        t in 0.0_f32..=1.0_f32,
    ) {
        let v = min + t * (max - min);
        let params = QuantParamsInt4::from_range(min, max);
        let q  = params.quantize(v);
        let dq = params.dequantize(q);
        let error = (v - dq).abs();
        let bound = params.scale() + 1e-4;
        prop_assert!(
            error <= bound,
            "INT4 round-trip error {:.9} > bound {:.9}  \
             (min={}, max={}, v={}, q={}, dq={}, scale={})",
            error, bound, min, max, v, q, dq, params.scale()
        );
    }

    /// INT4 error must always exceed INT8 error for the same range, since INT4
    /// has ~16x fewer quantization levels.
    #[test]
    fn prop_int4_scale_larger_than_int8(
        (min, max) in spanning_range(),
    ) {
        let p8 = QuantParams::from_range(min, max);
        let p4 = QuantParamsInt4::from_range(min, max);
        prop_assert!(
            p4.scale() >= p8.scale(),
            "INT4 scale {} < INT8 scale {} for range [{}, {}]",
            p4.scale(), p8.scale(), min, max
        );
    }
}

// ---------------------------------------------------------------------------
// Property 3 — Pack / unpack identity
//
// Bit-packing is lossless: unpack(pack(data)) == data for any valid INT4 slice.
// The packed length is always ceil(len / 2).
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_pack_unpack_identity(data in int4_vec()) {
        let packed   = pack_int4(&data);
        let unpacked = unpack_int4(&packed, data.len());
        prop_assert_eq!(
            &data, &unpacked,
            "pack/unpack mismatch for {} values", data.len()
        );
    }

    #[test]
    fn prop_packed_size_is_ceil_half(data in int4_vec()) {
        let packed = pack_int4(&data);
        let expected = data.len().div_ceil(2);
        prop_assert_eq!(
            packed.len(), expected,
            "packed size {} != ceil({}/2) = {}",
            packed.len(), data.len(), expected
        );
    }

    /// Packing a single value must produce exactly 1 byte.
    #[test]
    fn prop_single_value_packs_to_one_byte(v in int4_value()) {
        let packed = pack_int4(&[v]);
        prop_assert_eq!(packed.len(), 1);
        let unpacked = unpack_int4(&packed, 1);
        prop_assert_eq!(unpacked, vec![v]);
    }
}

// ---------------------------------------------------------------------------
// Property 4 — No panic
//
// All constructors (`from_f32`, `from_f32_with_range`, `from_f32_per_channel`)
// must return Ok or a typed QuantizeError — never panic — for any finite input.
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn prop_int8_from_f32_no_panic(data in finite_f32_vec()) {
        let len = data.len();
        let result = QuantizedTensor::from_f32(&data, vec![len]);
        // A non-empty, correctly-shaped tensor must always succeed.
        prop_assert!(result.is_ok(), "from_f32 unexpectedly failed: {:?}", result.err());
    }

    #[test]
    fn prop_int4_from_f32_no_panic(data in finite_f32_vec()) {
        let len = data.len();
        let result = QuantizedTensorInt4::from_f32(&data, vec![len]);
        prop_assert!(result.is_ok(), "INT4 from_f32 unexpectedly failed: {:?}", result.err());
    }

    #[test]
    fn prop_int8_from_f32_with_range_no_panic(
        data in finite_f32_vec(),
        (min, max) in spanning_range(),
    ) {
        let len = data.len();
        let result = QuantizedTensor::from_f32_with_range(&data, vec![len], min, max);
        prop_assert!(result.is_ok(), "from_f32_with_range failed: {:?}", result.err());
    }

    #[test]
    fn prop_int4_from_f32_with_range_no_panic(
        data in finite_f32_vec(),
        (min, max) in spanning_range(),
    ) {
        let len = data.len();
        let result = QuantizedTensorInt4::from_f32_with_range(&data, vec![len], min, max);
        prop_assert!(result.is_ok(), "INT4 from_f32_with_range failed: {:?}", result.err());
    }

    #[test]
    fn prop_int8_per_channel_no_panic(
        channels     in 1_usize..=8,
        per_channel  in 1_usize..=64,
    ) {
        let data: Vec<f32> = (0..channels * per_channel)
            .map(|i| (i as f32) / ((channels * per_channel) as f32))
            .collect();
        let result = QuantizedTensor::from_f32_per_channel(&data, vec![channels, per_channel]);
        prop_assert!(result.is_ok(), "INT8 per_channel failed: {:?}", result.err());
    }

    #[test]
    fn prop_int4_per_channel_no_panic(
        channels     in 1_usize..=8,
        per_channel  in 1_usize..=64,
    ) {
        let data: Vec<f32> = (0..channels * per_channel)
            .map(|i| (i as f32) / ((channels * per_channel) as f32))
            .collect();
        let result = QuantizedTensorInt4::from_f32_per_channel(&data, vec![channels, per_channel]);
        prop_assert!(result.is_ok(), "INT4 per_channel failed: {:?}", result.err());
    }

    /// Per-channel INT8 round-trip: each dequantized value is within one
    /// quantization step of the original.  For INT8 with 255 levels the
    /// per-channel scale is `range / 255`, so the max error per element is
    /// bounded by `range / 255 + eps`.
    #[test]
    fn prop_int8_per_channel_round_trip(
        channels     in 1_usize..=8,
        per_channel  in 1_usize..=64,
    ) {
        let data: Vec<f32> = (0..channels * per_channel)
            .map(|i| ((i as f32) - (channels * per_channel / 2) as f32) * 0.01)
            .collect();
        let shape = vec![channels, per_channel];
        if let Ok(tensor) = QuantizedTensor::from_f32_per_channel(&data, shape) {
            let dq = tensor.to_f32();
            for ch in 0..channels {
                // Compute per-channel range and derive error bound
                let start = ch * per_channel;
                let end = start + per_channel;
                let ch_min = data[start..end].iter().copied().fold(f32::INFINITY, f32::min);
                let ch_max = data[start..end].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                // Quantization range always includes zero
                let adj_min = ch_min.min(0.0);
                let adj_max = ch_max.max(0.0);
                let range = (adj_max - adj_min).max(1e-10);
                // INT8: 255 levels, error <= scale = range/255
                let bound = range / 255.0 + 1e-4;
                for j in 0..per_channel {
                    let idx = start + j;
                    let error = (data[idx] - dq[idx]).abs();
                    prop_assert!(
                        error <= bound,
                        "per-channel INT8 error {:.9} > bound {:.9} (ch={}, idx={})",
                        error, bound, ch, idx
                    );
                }
            }
        }
    }

    /// Per-channel INT4 round-trip: same as above but INT4 has only 15 levels.
    #[test]
    fn prop_int4_per_channel_round_trip(
        channels     in 1_usize..=8,
        per_channel  in 1_usize..=64,
    ) {
        let data: Vec<f32> = (0..channels * per_channel)
            .map(|i| ((i as f32) - (channels * per_channel / 2) as f32) * 0.01)
            .collect();
        let shape = vec![channels, per_channel];
        if let Ok(tensor) = QuantizedTensorInt4::from_f32_per_channel(&data, shape) {
            let dq = tensor.to_f32();
            for ch in 0..channels {
                let start = ch * per_channel;
                let end = start + per_channel;
                let ch_min = data[start..end].iter().copied().fold(f32::INFINITY, f32::min);
                let ch_max = data[start..end].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let adj_min = ch_min.min(0.0);
                let adj_max = ch_max.max(0.0);
                let range = (adj_max - adj_min).max(1e-10);
                // INT4: 15 levels, error <= scale = range/15
                let bound = range / 15.0 + 1e-4;
                for j in 0..per_channel {
                    let idx = start + j;
                    let error = (data[idx] - dq[idx]).abs();
                    prop_assert!(
                        error <= bound,
                        "per-channel INT4 error {:.9} > bound {:.9} (ch={}, idx={})",
                        error, bound, ch, idx
                    );
                }
            }
        }
    }

    /// `to_f32` must not panic on any successfully-constructed tensor.
    #[test]
    fn prop_to_f32_no_panic(data in finite_f32_vec()) {
        let len = data.len();
        if let Ok(tensor) = QuantizedTensor::from_f32(&data, vec![len]) {
            let dq = tensor.to_f32();
            prop_assert_eq!(dq.len(), len, "dequantized length mismatch");
        }
        if let Ok(tensor) = QuantizedTensorInt4::from_f32(&data, vec![len]) {
            let dq = tensor.to_f32();
            prop_assert_eq!(dq.len(), len, "INT4 dequantized length mismatch");
        }
    }
}
