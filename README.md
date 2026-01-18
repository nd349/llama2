# Llama2 implementation from first-principles

PyTorch-llama2 implementation from first-principles.


## Contents

- `inference.py` — example inference script for loading model weights and running generation
- `model.py`, `tokenizer` (or similar) — helper modules for model loading and tokenization

## Quick start (tcsh)

Install dependencies (adjust versions or use `requirements.txt` in this folder if provided):

```tcsh
pip install --upgrade pip
pip install torch torchvision transformers accelerate sentencepiece safetensors tqdm
```

Place model weights

1. Create a folder for weights in this directory (example):

```tcsh
mkdir -p ./weights/llama
# copy or download your model files into ./weights/llama
```

2. Ensure you follow the license and usage terms of the model provider before downloading or redistributing weights.

## Running inference

The `inference.py` script demonstrates loading the model and running generation. An example invocation (adjust flags to your script):

```tcsh
python inference.py
```

## Typical workflow

1. Download or convert weights into `weights/llama`.
2. Install dependencies in a virtual environment.
3. Run `inference.py` or open any example notebook to test generation.

## Requirements
- Python 3.8+
- PyTorch (match CUDA version if using GPU)
- transformers, accelerate, sentencepiece, safetensors (optional)
- other utilities used by scripts in this folder (check `requirements.txt`)

## Citation

pytorch-llama (implementation reference): https://github.com/hkproj/pytorch-llama


# llama2
