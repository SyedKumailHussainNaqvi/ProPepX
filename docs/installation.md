# Installation

## Conda installation

```bash
git clone https://github.com/SyedKumailHussainNaqvi/ProPepX.git
cd ProPepX

conda env create -f propepx.yml
conda activate propepx
```

## Update an existing environment

```bash
conda activate propepx
pip install -U pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft captum sentencepiece huggingface-hub safetensors tokenizers
pip install numpy scipy scikit-learn pandas matplotlib plotly biopython h5py
```

## Hardware recommendation

| Mode | Recommended device |
|---|---|
| CPU quick test | Works for small examples, slower |
| Single GPU | NVIDIA GPU with CUDA support |
| ESM-3 inference | GPU strongly recommended |
| Batch inference | GPU recommended |

## Quick verification

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
