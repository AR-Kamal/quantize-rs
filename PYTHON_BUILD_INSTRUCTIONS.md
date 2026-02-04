# Python Bindings Setup Instructions

## Files to Add/Modify

### 1. Add python.rs
Copy `python.rs` to `src/python.rs`

### 2. Update Cargo.toml
Replace entire file with the new `Cargo.toml`

Key changes:
- Added `crate-type = ["rlib", "cdylib"]` to `[lib]`
- Added `pyo3` dependency with `optional = true`
- Added `python` feature flag
- Added `[package.metadata.maturin]` section

### 3. Add pyproject.toml
Copy `pyproject.toml` to project root

### 4. Create Python package structure
```bash
mkdir -p python/quantize_rs
```

Copy files:
- `__init__.py` → `python/quantize_rs/__init__.py`
- `__init__.pyi` → `python/quantize_rs/__init__.pyi`

### 5. Add README
Copy `README_PYTHON.md` to project root

### 6. Update lib.rs
Add this line at the end of `src/lib.rs`:

```rust
// Python bindings (compiled only when "python" feature is enabled)
#[cfg(feature = "python")]
mod python;
```

---

## Build & Test

### Install Maturin
```bash
pip install maturin
```

### Build Python Extension (Development)
```bash
# Build and install in editable mode
maturin develop --release --features python

# Or without release optimizations (faster build)
maturin develop --features python
```

### Test the Bindings
```bash
# Copy test file to project root
cp test_python_bindings.py .

# Install pytest
pip install pytest onnxruntime numpy

# Run tests
pytest test_python_bindings.py -v
```

### Quick Manual Test
```python
# In Python REPL
import quantize_rs

# Test model info
info = quantize_rs.model_info("mnist.onnx")
print(f"{info.name}: {info.num_nodes} nodes")

# Test basic quantization
quantize_rs.quantize("mnist.onnx", "mnist_int8.onnx", bits=8)

# Test with calibration
quantize_rs.quantize_with_calibration(
    "mnist.onnx",
    "mnist_calibrated.onnx",
    num_samples=10,
    sample_shape=[1, 28, 28],
    method="minmax"
)
```

---

## Build Wheels for Distribution

### Build for Current Platform
```bash
maturin build --release --features python
```

Wheel will be in `target/wheels/`

### Build for Multiple Python Versions
```bash
# Install multiple Python versions (pyenv recommended)
maturin build --release --features python -i python3.8 -i python3.9 -i python3.10 -i python3.11 -i python3.12
```

### Test Wheel
```bash
pip install target/wheels/quantize_rs-0.3.0-*.whl
python -c "import quantize_rs; print(quantize_rs.__version__)"
```

---

## Publish to PyPI

### Test PyPI (Recommended First)
```bash
# Build wheels
maturin build --release --features python

# Upload to Test PyPI
maturin upload --repository testpypi target/wheels/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ quantize-rs
```

### Production PyPI
```bash
# Set API token
export MATURIN_PYPI_TOKEN=your_pypi_token

# Build and upload
maturin publish --features python
```

Or manually:
```bash
maturin build --release --features python
twine upload target/wheels/*
```

---

## GitHub Actions (CI/CD)

Create `.github/workflows/python-wheels.yml`:

```yaml
name: Build Python Wheels

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build-wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Build wheels
        uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release --out dist --features python
          sccache: 'true'
          manylinux: auto
      
      - name: Upload wheels
        uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [build-wheels]
    if: startsWith(github.ref, 'refs/tags/')
    
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheels-*
          path: dist
          merge-multiple: true
      
      - uses: PyO3/maturin-action@v1
        with:
          command: upload
          args: --non-interactive --skip-existing dist/*
        env:
          MATURIN_PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
```

---

## Troubleshooting

### Error: "no such feature: python"
Make sure you're building with `--features python`

### Error: "pyo3 not found"
The `pyo3` dependency is optional. Build with `--features python` to enable it.

### Import error: "module not found"
After `maturin develop`, the module should be installed. If not:
```bash
pip uninstall quantize-rs
maturin develop --release --features python
```

### Slow builds
Use `maturin develop` without `--release` for faster development builds.

### Type checking doesn't work
Make sure `__init__.pyi` is in `python/quantize_rs/__init__.pyi`

---

## What's Next After Python Bindings Work

1. **Write blog post** demonstrating:
   - CLI usage
   - Python API usage
   - Accuracy benchmarks
   - Compression results

2. **Create YouTube demo** showing:
   - Quantizing ResNet-18
   - Comparing FP32 vs INT8 accuracy
   - Speed benchmarks

3. **Tag v0.3.0 release**:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

4. **Publish**:
   - `cargo publish` (Rust crate)
   - `maturin publish --features python` (Python package)