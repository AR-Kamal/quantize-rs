//! Calibration datasets and activation-based range estimation.
//!
//! - [`CalibrationDataset`] — load or generate calibration samples
//! - [`methods::CalibrationMethod`] — range optimization strategies
//! - [`stats::ActivationStats`] — incremental min/max/histogram tracker
//! - [`inference::ActivationEstimator`] — run inference to collect activation stats

use crate::errors::{QuantizeError, Result};
#[cfg(any(feature = "calibration", feature = "safetensors-input"))]
use std::path::Path;

#[cfg(feature = "calibration")]
pub mod inference;
#[cfg(feature = "calibration")]
mod static_quantization;
#[cfg(feature = "calibration")]
pub use static_quantization::{quantize_static, quantize_static_matrix};
pub mod methods;
pub mod stats;

#[cfg(feature = "calibration")]
pub use inference::ActivationEstimator;

/// A collection of FP32 calibration samples used for range estimation.
#[derive(Clone)]
pub struct CalibrationDataset {
    /// Individual samples, each flattened to match `shape`.
    pub samples: Vec<Vec<f32>>,

    /// Shape of a single sample (excluding batch dimension).
    pub shape: Vec<usize>,
}

// Check dimensions and byte capacities before deriving offsets or allocating.
// Rust allocations cannot exceed isize::MAX, even when usize multiplication fits.
fn sample_elements(shape: &[usize], num_samples: usize) -> Result<usize> {
    let invalid = || QuantizeError::Calibration {
        reason: format!(
            "Invalid dataset shape {shape:?} with {num_samples} samples: dimensions must be \
             positive and element/byte counts must fit allocation limits"
        ),
    };
    if shape.is_empty() || shape.contains(&0) || num_samples == 0 {
        return Err(invalid());
    }
    let elements = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(invalid)?;
    elements
        .checked_mul(num_samples)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(invalid)?;
    num_samples
        .checked_mul(std::mem::size_of::<Vec<f32>>())
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(invalid)?;
    Ok(elements)
}

impl std::fmt::Debug for CalibrationDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalibrationDataset")
            .field("num_samples", &self.samples.len())
            .field("shape", &self.shape)
            .finish()
    }
}

impl CalibrationDataset {
    /// Load calibration samples from a NumPy `.npy` file.
    ///
    /// The array must be at least 2-dimensional `[batch, ...]`.
    ///
    /// Requires the `calibration` feature (enabled by default).
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Calibration`] if the file is missing, not `.npy`,
    /// or has an invalid shape.
    #[cfg(feature = "calibration")]
    pub fn from_numpy(path: impl AsRef<Path>) -> Result<Self> {
        use ndarray::{Array, IxDyn};

        let path = path.as_ref();

        if !path.exists() {
            return Err(QuantizeError::Calibration {
                reason: format!("File not found: {}", path.display()),
            });
        }

        let array: Array<f32, IxDyn> = if path.extension().and_then(|s| s.to_str()) == Some("npy") {
            ndarray_npy::read_npy(path).map_err(|e| QuantizeError::Calibration {
                reason: format!("Failed to read NPY file '{}': {e}", path.display()),
            })?
        } else {
            return Err(QuantizeError::Calibration {
                reason: "Only .npy files supported currently".into(),
            });
        };

        let shape: Vec<usize> = array.shape().to_vec();

        if shape.is_empty() {
            return Err(QuantizeError::Calibration {
                reason: "Invalid array shape".into(),
            });
        }

        if shape.len() < 2 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Calibration data must be at least 2-dimensional (batch, ...). Got shape {:?}",
                    shape
                ),
            });
        }

        let num_samples = shape[0];
        let sample_size = sample_elements(&shape[1..], num_samples)?;

        // `into_raw_vec` returns data in memory order, so the array must be
        // C-contiguous for the per-sample slicing below to be correct.  Move the
        // buffer out directly in the common (already-standard) case; only a
        // Fortran-ordered `.npy` needs a re-layout copy.
        let data = if array.is_standard_layout() {
            array.into_raw_vec()
        } else {
            array.as_standard_layout().into_owned().into_raw_vec()
        };
        let samples = data
            .chunks_exact(sample_size)
            .map(<[f32]>::to_vec)
            .collect();

        Ok(Self {
            samples,
            shape: shape[1..].to_vec(),
        })
    }

    /// Load calibration samples from a HuggingFace `.safetensors` file that
    /// contains exactly one tensor.
    ///
    /// The tensor must be f32 and at least 2-dimensional `[batch, ...]`.
    /// Requires the `safetensors-input` feature.
    ///
    /// For files with multiple named tensors, use
    /// [`from_safetensors_named`](Self::from_safetensors_named) instead.
    #[cfg(feature = "safetensors-input")]
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let buffer = std::fs::read(path).map_err(|e| QuantizeError::Calibration {
            reason: format!("Failed to read safetensors file '{}': {e}", path.display()),
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&buffer).map_err(|e| {
            QuantizeError::Calibration {
                reason: format!("Failed to parse safetensors file: {e}"),
            }
        })?;
        let names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();
        if names.is_empty() {
            return Err(QuantizeError::Calibration {
                reason: "safetensors file contains no tensors".into(),
            });
        }
        if names.len() > 1 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "safetensors file contains {} tensors; pass one explicitly via \
                     from_safetensors_named().  Available tensors: {}",
                    names.len(),
                    names.join(", ")
                ),
            });
        }
        Self::from_safetensors_view(&tensors, &names[0])
    }

    /// Load calibration samples from a specific named tensor inside a
    /// `.safetensors` file.
    ///
    /// Requires the `safetensors-input` feature.
    #[cfg(feature = "safetensors-input")]
    pub fn from_safetensors_named(path: impl AsRef<Path>, tensor_name: &str) -> Result<Self> {
        let path = path.as_ref();
        let buffer = std::fs::read(path).map_err(|e| QuantizeError::Calibration {
            reason: format!("Failed to read safetensors file '{}': {e}", path.display()),
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&buffer).map_err(|e| {
            QuantizeError::Calibration {
                reason: format!("Failed to parse safetensors file: {e}"),
            }
        })?;
        Self::from_safetensors_view(&tensors, tensor_name)
    }

    #[cfg(feature = "safetensors-input")]
    fn from_safetensors_view(
        tensors: &safetensors::SafeTensors<'_>,
        tensor_name: &str,
    ) -> Result<Self> {
        use safetensors::Dtype;

        let view = tensors
            .tensor(tensor_name)
            .map_err(|e| QuantizeError::Calibration {
                reason: format!(
                    "Tensor '{}' not found in safetensors file: {e}",
                    tensor_name
                ),
            })?;

        if view.dtype() != Dtype::F32 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Tensor '{}' has dtype {:?}; only F32 is supported for calibration input",
                    tensor_name,
                    view.dtype()
                ),
            });
        }

        let shape: Vec<usize> = view.shape().to_vec();
        if shape.len() < 2 {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Calibration tensor must be at least 2-dimensional (batch, ...). \
                     Got shape {:?}",
                    shape
                ),
            });
        }
        let sample_size = sample_elements(&shape[1..], shape[0])?;
        // The helper has checked the complete product and byte capacity.
        let expected_bytes = sample_size * shape[0] * std::mem::size_of::<f32>();
        let raw = view.data();
        if raw.len() != expected_bytes {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Tensor '{}' data size {} bytes does not match shape {:?} \
                     × 4 = {} bytes",
                    tensor_name,
                    raw.len(),
                    shape,
                    expected_bytes
                ),
            });
        }

        // safetensors stores data little-endian, which matches every target
        // quantize-rs builds on today.  Decode per-f32 explicitly to stay
        // endian-safe rather than relying on an unchecked cast.
        let data: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let samples = data
            .chunks_exact(sample_size)
            .map(<[f32]>::to_vec)
            .collect();

        Ok(Self {
            samples,
            shape: shape[1..].to_vec(),
        })
    }

    /// Generate random calibration samples uniformly distributed in `range`.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Calibration`] for empty/zero/overflowing shapes,
    /// zero sample counts, allocation failure, or nonfinite/invalid ranges.
    pub fn random(shape: Vec<usize>, num_samples: usize, range: (f32, f32)) -> Result<Self> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(QuantizeError::Calibration {
                reason: format!("Invalid shape: {:?} - all dimensions must be > 0", shape),
            });
        }
        if num_samples == 0 {
            return Err(QuantizeError::Calibration {
                reason: "num_samples must be > 0".into(),
            });
        }
        if !range.0.is_finite()
            || !range.1.is_finite()
            || range.0 >= range.1
            || !(range.1 - range.0).is_finite()
        {
            return Err(QuantizeError::Calibration {
                reason: format!(
                    "Invalid range: ({}, {}) - bounds and width must be finite, with min < max",
                    range.0, range.1
                ),
            });
        }
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let sample_size = sample_elements(&shape, num_samples)?;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(num_samples)
            .map_err(|e| QuantizeError::Calibration {
                reason: format!("Cannot allocate sample list: {e}"),
            })?;

        for _ in 0..num_samples {
            let mut sample = Vec::new();
            sample
                .try_reserve_exact(sample_size)
                .map_err(|e| QuantizeError::Calibration {
                    reason: format!("Cannot allocate calibration sample: {e}"),
                })?;
            sample.extend((0..sample_size).map(|_| rng.gen_range(range.0..range.1)));
            samples.push(sample);
        }

        Ok(Self { samples, shape })
    }

    /// Create a dataset from pre-existing sample vectors.
    ///
    /// # Errors
    ///
    /// Returns [`QuantizeError::Calibration`] if `samples` is empty, shape has
    /// empty/zero/overflowing dimensions, or any sample has the wrong length.
    pub fn from_samples(samples: Vec<Vec<f32>>, shape: Vec<usize>) -> Result<Self> {
        let num_samples = samples.len();

        if num_samples == 0 {
            return Err(QuantizeError::Calibration {
                reason: "No samples provided".into(),
            });
        }

        let expected_size = sample_elements(&shape, num_samples)?;

        for (i, sample) in samples.iter().enumerate() {
            if sample.len() != expected_size {
                return Err(QuantizeError::Calibration {
                    reason: format!(
                        "Sample {} has size {} but expected {} (shape: {:?})",
                        i,
                        sample.len(),
                        expected_size,
                        shape
                    ),
                });
            }
        }

        Ok(Self { samples, shape })
    }

    /// Shape of a single sample (excluding batch dimension).
    pub fn sample_shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the dataset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_dimensions_are_rejected_before_allocation() {
        for shape in [
            vec![],
            vec![0],
            vec![usize::MAX, 2],
            vec![usize::MAX / 4 + 1],
        ] {
            assert!(CalibrationDataset::from_samples(vec![vec![]], shape.clone()).is_err());
            assert!(CalibrationDataset::random(shape, 1, (0.0, 1.0)).is_err());
        }
        for count in [
            usize::MAX,
            isize::MAX as usize / std::mem::size_of::<Vec<f32>>() + 1,
        ] {
            assert!(CalibrationDataset::random(vec![1], count, (0.0, 1.0)).is_err());
        }
        for range in [(f32::NAN, 1.0), (0.0, f32::INFINITY), (-f32::MAX, f32::MAX)] {
            assert!(CalibrationDataset::random(vec![1], 1, range).is_err());
        }
    }

    #[cfg(feature = "calibration")]
    #[test]
    fn numpy_malformed_shape_headers_return_errors() {
        // No payload allocation: an overflowing shape and zero-sized sample
        // dimensions must be rejected even when the header itself is valid NPY.
        for shape in [
            format!("({}, 2)", usize::MAX),
            "(1, 0)".into(),
            "(0, 2)".into(),
        ] {
            let mut header =
                format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
            let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
            header.push_str(&" ".repeat(padding));
            header.push('\n');
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            let file = tempfile::NamedTempFile::with_suffix(".npy").unwrap();
            std::fs::write(file.path(), bytes).unwrap();
            assert!(
                CalibrationDataset::from_numpy(file.path()).is_err(),
                "shape {shape}"
            );
        }
    }

    #[cfg(feature = "safetensors-input")]
    #[test]
    fn safetensors_malformed_shape_headers_return_errors() {
        for shape in [
            format!("[{}, 2]", usize::MAX),
            "[1, 0]".into(),
            "[0, 2]".into(),
        ] {
            let mut header = format!(
                "{{\"input\":{{\"dtype\":\"F32\",\"shape\":{shape},\"data_offsets\":[0,0]}}}}"
            );
            header.push_str(&" ".repeat((8 - header.len() % 8) % 8));
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            let file = tempfile::NamedTempFile::with_suffix(".safetensors").unwrap();
            std::fs::write(file.path(), bytes).unwrap();
            assert!(
                CalibrationDataset::from_safetensors(file.path()).is_err(),
                "shape {shape}"
            );
        }
    }

    #[test]
    fn test_random_dataset() {
        let dataset = CalibrationDataset::random(vec![3, 224, 224], 10, (-1.0, 1.0)).unwrap();

        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.sample_shape(), &[3, 224, 224]);
        assert_eq!(dataset.samples[0].len(), 3 * 224 * 224);
    }

    #[test]
    fn test_from_samples() {
        let samples = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let dataset = CalibrationDataset::from_samples(samples, vec![3]).unwrap();
        assert_eq!(dataset.len(), 2);
    }

    #[cfg(feature = "calibration")]
    #[test]
    fn test_from_numpy_fortran_order_slices_by_logical_samples() {
        use ndarray::{Array2, ShapeBuilder};

        // Logical content: 3 samples × 4 features.
        //   sample 0 = [10, 11, 12, 13]
        //   sample 1 = [20, 21, 22, 23]
        //   sample 2 = [30, 31, 32, 33]
        // Built in column-major (Fortran) memory so the on-disk `.npy` is
        // fortran_order — the layout that used to mis-slice into transposed
        // samples before from_numpy forced a standard layout.
        let f_memory: Vec<f32> = vec![
            10., 20., 30., // column 0
            11., 21., 31., // column 1
            12., 22., 32., // column 2
            13., 23., 33., // column 3
        ];
        let arr = Array2::from_shape_vec((3, 4).f(), f_memory).unwrap();
        assert!(
            !arr.is_standard_layout(),
            "test setup: array should be Fortran-ordered"
        );

        let tmp = tempfile::NamedTempFile::with_suffix(".npy").unwrap();
        ndarray_npy::write_npy(tmp.path(), &arr).unwrap();

        let dataset = CalibrationDataset::from_numpy(tmp.path()).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.sample_shape(), &[4]);
        assert_eq!(dataset.samples[0], vec![10., 11., 12., 13.]);
        assert_eq!(dataset.samples[1], vec![20., 21., 22., 23.]);
        assert_eq!(dataset.samples[2], vec![30., 31., 32., 33.]);
    }

    #[cfg(feature = "safetensors-input")]
    #[test]
    fn test_from_safetensors_round_trip() {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        use std::collections::HashMap;

        // Build 3 samples of shape [2, 4] = 24 floats.
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let raw: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![3, 2, 4], &raw).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("input".to_string(), view);
        let bytes = serialize(&tensors, &None).unwrap();

        let tmp = tempfile::NamedTempFile::with_suffix(".safetensors").unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let dataset = CalibrationDataset::from_safetensors(tmp.path()).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.sample_shape(), &[2, 4]);
        // Each sample holds 8 floats.
        assert_eq!(dataset.samples[0].len(), 8);
        // First float of sample 0 is 0.0, first of sample 1 is 0.8 (index 8 * 0.1).
        assert!((dataset.samples[0][0] - 0.0).abs() < 1e-6);
        assert!((dataset.samples[1][0] - 0.8).abs() < 1e-6);
    }

    #[cfg(feature = "safetensors-input")]
    #[test]
    fn test_from_safetensors_multi_tensor_errors_without_name() {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        use std::collections::HashMap;

        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let raw: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
        let v1 = TensorView::new(Dtype::F32, vec![2, 4], &raw).unwrap();
        let v2 = TensorView::new(Dtype::F32, vec![2, 4], &raw).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("a".to_string(), v1);
        tensors.insert("b".to_string(), v2);
        let bytes = serialize(&tensors, &None).unwrap();

        let tmp = tempfile::NamedTempFile::with_suffix(".safetensors").unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();

        let err = CalibrationDataset::from_safetensors(tmp.path()).unwrap_err();
        assert!(err.to_string().contains("contains 2 tensors"));

        // But named access works.
        let dataset = CalibrationDataset::from_safetensors_named(tmp.path(), "a").unwrap();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.sample_shape(), &[4]);
    }
}
