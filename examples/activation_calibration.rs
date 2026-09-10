//! Static INT8 Conv calibration using representative FP32 samples.
use quantize_rs::{quantize_static, CalibrationDataset, QuantConfig};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model = value("--model").unwrap_or_else(|| "model.onnx".into());
    let data = value("--calibration-data").ok_or_else(|| {
        anyhow::anyhow!("provide --calibration-data samples.npy (representative NCHW samples)")
    })?;
    let output = value("--output").unwrap_or_else(|| "model_calibrated.onnx".into());
    if value("--bits").is_some_and(|b| b != "8") {
        anyhow::bail!("static calibration supports INT8 only");
    }
    let dataset = CalibrationDataset::from_numpy(data)?;
    let config = QuantConfig::int8()
        .with_per_channel(args.iter().any(|a| a == "--per-channel"))
        .with_symmetric(true);
    quantize_static(&model, &output, &dataset, config)?;
    println!("Saved {output}. Compare output accuracy and latency on your deployment runtime.");
    Ok(())
}
