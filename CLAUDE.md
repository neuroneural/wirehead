# PyPI Build Instructions

## Building and publishing to PyPI

### 1. Install build tools
```bash
pip install build twine
```

### 2. Generate distribution packages
```bash
python -m build
```

### 3. Upload to PyPI
```bash
python -m twine upload dist/*
```

### 4. Upload to TestPyPI (for testing before real release)
```bash
python -m twine upload --repository testpypi dist/*
```