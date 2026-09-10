"""Inspect release archives without extracting or importing their contents.

Maintainer check (Python 3.11+): python scripts/check_release_artifacts.py ARCHIVE...
"""
import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile


def check(path, version):
    forbidden = {'.pypirc', '.env', 'test.py', 'test_python_bindings.py'}
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata_paths = [n for n in names if n.endswith('.dist-info/METADATA')]
            assert len(metadata_paths) == 1, 'Expected one wheel metadata file'
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            assert metadata['Name'] == 'quantization-rs', 'Unexpected distribution name'
            assert metadata['Version'] == version, 'Wheel version mismatch'
            assert (metadata['License-Expression'] or metadata['License']) == 'MIT', 'License mismatch'
            assert any(n.endswith(('.pyd', '.so')) for n in names), 'Missing native module'
            assert any(n.endswith('quantize_rs.pyi') or n.endswith('quantize_rs/__init__.pyi') for n in names), 'Missing type stubs'
            assert any(PurePosixPath(n).name == 'py.typed' for n in names), 'Missing py.typed'
            assert any(PurePosixPath(n).name == 'LICENSE' for n in names), 'Missing license file'
    else:
        with tarfile.open(path, 'r:gz') as archive:
            members = archive.getmembers()
            assert members and all(m.isfile() or m.isdir() for m in members), 'Unexpected archive member type'
            raw_names = {m.name for m in members}
            roots = {PurePosixPath(n).parts[0] for n in raw_names}
            assert len(roots) == 1, 'Expected one archive root'
            names = {'/'.join(PurePosixPath(n).parts[1:]) for n in raw_names}
            required = {'Cargo.toml', 'Cargo.lock', 'build.rs', 'proto/onnx.proto3', 'LICENSE',
                        'src/lib.rs', 'src/calibration/static_quantization.rs',
                        'src/onnx_utils/selection.rs', 'src/onnx_utils/calibrated.rs',
                        'src/onnx_utils/matrix_calibrated.rs'}
            if path.name.endswith('.tar.gz'):
                required |= {'pyproject.toml', 'quantize_rs.pyi', 'py.typed', 'README_PYTHON.md'}
            assert required <= names, f'Missing archive files: {sorted(required - names)}'
            root = next(iter(roots))
            manifest = tomllib.loads(archive.extractfile(root + '/Cargo.toml').read().decode())
            assert manifest['package']['version'] == version, 'Cargo version mismatch'
            assert manifest['package']['license'] == 'MIT', 'Cargo license mismatch'
            if path.name.endswith('.tar.gz'):
                project = tomllib.loads(archive.extractfile(root + '/pyproject.toml').read().decode())['project']
                assert project['version'] == version, 'Python version mismatch'
                assert project['license'] == {'text': 'MIT'}, 'Python license mismatch'
    assert not any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names), 'Unsafe archive path'
    assert not any(PurePosixPath(n).name in forbidden or n.endswith('.onnx') for n in names), 'Unexpected local/model file'
    print(f'PASS {path.name}: {len(names)} entries, version {version}, required files and metadata present')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / 'Cargo.toml').read_text())['package']['version']
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    assert project['version'] == version, 'Checkout versions disagree'
    for path in args.archives:
        check(path, version)


if __name__ == '__main__':
    main()
