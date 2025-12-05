# HEIR Comparisons with other FHE compilers

This repository stores reproducibility artifacts for the HEIR compiler paper.

## Installation

Requires Python3.11

```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```
