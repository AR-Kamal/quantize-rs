use anyhow::{Context, Result, bail};
use ndarray::{Array, IxDyn};
use std::path::Path;

pub mod stats;
pub mod methods;
pub mod inference;

pub use inference::ActivationEstimator;

#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    pub samples: Vec<Vec<f32>>,
    
    pub shape: Vec<usize>,
    
    pub num_samples: usize,
}

impl CalibrationDataset {
    pub fn from_numpy(path: &str) -> Result<Self> {
        let path = Path::new(path);
        
        if !path.exists() {
            bail!("File not found: {}", path.display());
        }
        
        let array: Array<f32, IxDyn> = if path.extension().and_then(|s| s.to_str()) == Some("npy") {
            ndarray_npy::read_npy(path)
                .with_context(|| format!("Failed to read NPY file: {}", path.display()))?
        } else {
            bail!("Only .npy files supported currently");
        };
        
        let shape: Vec<usize> = array.shape().to_vec();
        
        if shape.is_empty() {
            bail!("Invalid array shape");
        }
        
        let num_samples = shape[0];
        let sample_size: usize = shape[1..].iter().product();
        
        let data = array.into_raw_vec();
        let mut samples = Vec::with_capacity(num_samples);
        
        for i in 0..num_samples {
            let start = i * sample_size;
            let end = start + sample_size;
            samples.push(data[start..end].to_vec());
        }
        
        Ok(Self {
            samples,
            shape: shape[1..].to_vec(),
            num_samples,
        })
    }
    
    pub fn random(shape: Vec<usize>, num_samples: usize, range: (f32, f32)) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let sample_size: usize = shape.iter().product();
        let mut samples = Vec::with_capacity(num_samples);
        
        for _ in 0..num_samples {
            let sample: Vec<f32> = (0..sample_size)
                .map(|_| rng.gen_range(range.0..range.1))
                .collect();
            samples.push(sample);
        }
        
        Self {
            samples,
            shape,
            num_samples,
        }
    }
    
    pub fn from_samples(samples: Vec<Vec<f32>>, shape: Vec<usize>) -> Result<Self> {
        let num_samples = samples.len();
        
        if num_samples == 0 {
            bail!("No samples provided");
        }
        
        let expected_size: usize = shape.iter().product();
        
        for (i, sample) in samples.iter().enumerate() {
            if sample.len() != expected_size {
                bail!(
                    "Sample {} has size {} but expected {} (shape: {:?})",
                    i, sample.len(), expected_size, shape
                );
            }
        }
        
        Ok(Self {
            samples,
            shape,
            num_samples,
        })
    }
    
    pub fn get_batch(&self, start: usize, size: usize) -> &[Vec<f32>] {
        let end = (start + size).min(self.num_samples);
        &self.samples[start..end]
    }
    
    pub fn sample_shape(&self) -> &[usize] {
        &self.shape
    }
    
    pub fn len(&self) -> usize {
        self.num_samples
    }
    
    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_random_dataset() {
        let dataset = CalibrationDataset::random(vec![3, 224, 224], 10, (-1.0, 1.0));
        
        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.sample_shape(), &[3, 224, 224]);
        assert_eq!(dataset.samples[0].len(), 3 * 224 * 224);
    }
    
    #[test]
    fn test_from_samples() {
        let samples = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let dataset = CalibrationDataset::from_samples(samples, vec![3]).unwrap();
        assert_eq!(dataset.len(), 2);
    }
}