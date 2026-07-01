# ProPepX

<p align="center">
  <img src="ProPepX.png" alt="ProPepX conceptual workflow" width="850">
</p>

<p align="center">
  <b>A unified framework for interpretable bidirectional interaction-aware transfer learning in joint residue-level protein–peptide binding-site prediction</b>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#web-server">Web server</a> •
  <a href="#models-and-datasets">Models</a> •
  <a href="#training-and-reproducibility">Reproducibility</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.4-green" alt="CUDA">
  <img src="https://img.shields.io/badge/HuggingFace-Models-yellow" alt="HuggingFace">
  <img src="https://img.shields.io/badge/WebServer-FastAPI%2FHTML-purple" alt="WebServer">
  <img src="https://img.shields.io/badge/Status-Manuscript%20Submission-lightgrey" alt="Status">
</p>

---

## Overview

**ProPepX** is a deep learning framework for predicting binding residues in
protein–peptide interactions. Unlike sequence-only predictors that treat the
protein and peptide independently, ProPepX is designed around **interaction-aware
transfer learning**, allowing protein and peptide representations to be evaluated
in a shared binding context.

The repository provides:

- command-line inference for protein-side, peptide-side, joint, and zero-shot prediction;
- pretrained and fine-tuned checkpoints hosted on Hugging Face;
- reproducible pretraining and fine-tuning scripts;
- an interactive HTML/web-server interface;
- publication-ready prediction reports, residue-level scores, and visual outputs.

> This repository accompanies the manuscript submitted to **Nature Machine Intelligence**.

---

## Highlights

| Capability | Description |
|---|---|
| Protein-side prediction | Predicts peptide-binding residues on the protein chain |
| Peptide-side prediction | Predicts protein-binding residues on the peptide chain |
| Joint prediction | Predicts binding residues for both partners in a unified setting |
| Zero-shot prediction | Runs transfer-learning inference without task-specific fine-tuning |
| Embedding backbones | Supports ProtTransT5 and ESM-3 embeddings |
| Interpretability | Produces residue-level confidence, attention-style maps, and HTML reports |
| Web server | Provides a browser-based interface for non-programming users |
| Reproducibility | Includes training, fine-tuning, inference, and validation commands |

---

## Installation

### Option 1 — Conda installation

```bash
git clone https://github.com/<YOUR-USERNAME>/ProPepX.git
cd ProPepX

conda env create -f propepx.yml
conda activate propepx
```

### Option 2 — Update an existing environment

```bash
conda activate propepx
pip install -U pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft captum sentencepiece huggingface-hub safetensors tokenizers
pip install numpy scipy scikit-learn pandas matplotlib plotly biopython h5py
```

### Hardware recommendation

| Mode | Recommended device |
|---|---|
| CPU quick test | Works for small examples, slower |
| Single GPU | NVIDIA GPU with CUDA support |
| ESM-3 inference | GPU strongly recommended |
| Batch inference | GPU recommended |

---

## Quick start

After installation, ProPepX can be tested using the included example runner.

```bash
python run_propepx_example.py
```

For direct command-line inference:

```bash
mkdir -p results

python propepx_predict.py \
  --protein "MEMPQLSKWNQDSRNDAMENTLLVSHVLPNISVAQIHNALDGISFVQHFSLSTINLIKNDERSLWVHFKAGTNMDGAKEAVDGIQLDSNFTIESENPKIPTHTHPIPIFEIASSEQTCKNLLEKLIRFIDRASTKYSLPNDAAQRIEDRLKTHASMKDDDDKPTNFHDIRLSDLYAEYLRQVATFDFWTSKEYESLIALLQDSPAGYSRKKFNPSKEVGQEENIWLSDLENNFACLLEPENVDIKAKGALPVEDFINNELDSVIMKEDEQKYRCHVGTCAKLFLGPEFVRKHINKKHKDWLDHIKKVAICLYGYVLDPCRAMDPKVVSSAWSHPQFEK" \
  --peptide "KNEEDESNDSDKEDGEISEDD" \
  --embedding prottrans \
  --mode mode-GLOBAL \
  --dataset leads_ts251 \
  --gpu_id 0 \
  --save_html "results/"
```

Expected outputs:

```text
results/
├── prediction_report.html
├── protein_binding_scores.csv
├── peptide_binding_scores.csv
├── residue_probability_map.png
├── interaction_heatmap.png
└── summary.json
```

> A typical single-pair prediction can complete in less than 100 seconds after
> packages, model checkpoints, and embeddings are already available locally.
> Runtime depends on GPU, sequence length, embedding backend, and checkpoint loading.

---

## Input format

ProPepX accepts raw amino-acid sequences or FASTA-like sequences.

### Protein

```text
>Molecule_1
MEMPQLSKWNQDSRNDAMENTLLVSHVLPNISVAQIHNALDGISFVQHFSLSTINLI...
```

### Peptide

```text
>Peptide_1
KNEEDESNDSDKEDGEISEDD
```

Only standard amino-acid characters are used for inference.

---

## Prediction modes

| Mode | Argument | Description |
|---|---|---|
| Protein-side | `--mode prot` | Predicts binding residues on the protein |
| Peptide-side | `--mode pep` | Predicts binding residues on the peptide |
| Joint | `--mode mode-GLOBAL` | Predicts both protein and peptide binding residues |
| Zero-shot | `--mode zero-shot` | Runs zero-shot transfer prediction |

---

## Embedding backbones

| Embedding | Argument | Typical dimension |
|---|---:|---:|
| ProtTransT5 | `--embedding prottrans` | 1024 |
| ESM-3 600M | `--embedding esm` | 1152 |

---

## Example commands

### 1. Joint prediction with ProtTransT5

```bash
python propepx_predict.py \
  --protein "MEMPQLSKWNQDSRNDAMENTLLVSHVLPNISVAQIHNALDGISFVQHFSLSTINLIKNDERSLWVHFKAGTNMDGAKEAVDGIQLDSNFTIESENPKIPTHTHPIPIFEIASSEQTCKNLLEKLIRFIDRASTKYSLPNDAAQRIEDRLKTHASMKDDDDKPTNFHDIRLSDLYAEYLRQVATFDFWTSKEYESLIALLQDSPAGYSRKKFNPSKEVGQEENIWLSDLENNFACLLEPENVDIKAKGALPVEDFINNELDSVIMKEDEQKYRCHVGTCAKLFLGPEFVRKHINKKHKDWLDHIKKVAICLYGYVLDPCRAMDPKVVSSAWSHPQFEK" \
  --peptide "KNEEDESNDSDKEDGEISEDD" \
  --embedding prottrans \
  --mode mode-GLOBAL \
  --dataset leads_ts251 \
  --gpu_id 0 \
  --save_html "results/"
```

### 2. Joint prediction with ESM-3

```bash
python propepx_predict.py \
  --protein "MEMPQLSKWNQDSRNDAMENTLLVSHVLPNISVAQIHNALDGISFVQHFSLSTINLIKNDERSLWVHFKAGTNMDGAKEAVDGIQLDSNFTIESENPKIPTHTHPIPIFEIASSEQTCKNLLEKLIRFIDRASTKYSLPNDAAQRIEDRLKTHASMKDDDDKPTNFHDIRLSDLYAEYLRQVATFDFWTSKEYESLIALLQDSPAGYSRKKFNPSKEVGQEENIWLSDLENNFACLLEPENVDIKAKGALPVEDFINNELDSVIMKEDEQKYRCHVGTCAKLFLGPEFVRKHINKKHKDWLDHIKKVAICLYGYVLDPCRAMDPKVVSSAWSHPQFEK" \
  --peptide "KNEEDESNDSDKEDGEISEDD" \
  --embedding esm \
  --mode mode-GLOBAL \
  --dataset test167 \
  --gpu_id 0 \
  --save_html "results/"
```

### 3. Protein-side prediction

```bash
python propepx_predict.py \
  --protein "MEMPQLSKWNQDSRNDAMENTLLVSHVLPNISVAQIHNALDGISFVQHFSLSTINLIKNDERSLWVHFKAGTNMDGAKEAVDGIQLDSNFTIESENPKIPTHTHPIPIFEIASSEQTCKNLLEKLIRFIDRASTKYSLPNDAAQRIEDRLKTHASMKDDDDKPTNFHDIRLSDLYAEYLRQVATFDFWTSKEYESLIALLQDSPAGYSRKKFNPSKEVGQEENIWLSDLENNFACLLEPENVDIKAKGALPVEDFINNELDSVIMKEDEQKYRCHVGTCAKLFLGPEFVRKHINKKHKDWLDHIKKVAICLYGYVLDPCRAMDPKVVSSAWSHPQFEK" \
  --peptide "KNEEDESNDSDKEDGEISEDD" \
  --embedding prottrans \
  --mode prot \
  --dataset ts092 \
  --gpu_id 0 \
  --save_html "results/"
```

### 4. Peptide-side prediction

```bash
python propepx_predict.py \
  --protein "GGSEFSVGQGPAKTMEEASKRSYQFWDTQPVPKLGEVVNTHGPVEPDKDNIRQEPYTLPQGFTWDALDLGDRGVLKELYTLLNENYVEDDDNMFRFDYSPEFLLWALRPPGWLPQWHCGVRVVSSRKLVGFISAIPANIHIYDTEKKMVEINFLCVHKKLRSKRVAPVLIREITRRVHLEGIFQAVYTAGVVLPKPVGTCRYWHRSLNPRKLIEVKFSHLSRNMTMQRTMKLYRLPETPKTAGLRPMETKDIPVVHQLLTRYLKQFHLTPVMSQEEVEHWFYPQENIIDTFVVENANGEVTDFLSFYTLPSTIMNHPTHKSLKAAYSFYNVHTQTPLLDLMSDALVLAKMKGFDVFNALDLMENKTFLEKLKFGIGDGNLQYYLYNWKCPSMGAEKVGLVLQ" \
  --peptide "ANCFSKPR" \
  --embedding prottrans \
  --mode pep \
  --dataset camp_test231 \
  --gpu_id 0 \
  --save_html "results/"
```

### 5. Zero-shot prediction

```bash
python propepx_predict.py \
  --protein "QIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNL" \
  --peptide "ETFSDLWKLLPEN" \
  --embedding prottrans \
  --mode zero-shot \
  --dataset test167_zs \
  --gpu_id 0 \
  --save_html "results/"
```

---

## Web server

ProPepX includes a browser interface for interactive prediction.

### Static demo page

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

### Application server

If your repository includes the `app/` server directory:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

### Web-server workflow

```text
Protein sequence
      ↓
Peptide sequence
      ↓
Embedding backend
      ↓
Prediction mode
      ↓
Model checkpoint
      ↓
Interactive HTML report
```

---

## Docker

If Docker files are included in your repository:

```bash
docker build -t propepx:latest .
docker run --gpus all -p 8000:8000 propepx:latest
```

Open:

```text
http://localhost:8000
```

For CPU-only testing:

```bash
docker run -p 8000:8000 propepx:latest
```

---

## Models and datasets

Pretrained/fine-tuned checkpoints and benchmark embeddings are hosted on Hugging Face:

```text
https://huggingface.co/syedkumailhussain/ProPepX
```

### Available checkpoints

| Task | Backbone | Dataset/setting |
|---|---|---|
| Joint protein–peptide prediction | ESM-3 600M | LEADS TS251 |
| Joint protein–peptide prediction | ProtTransT5 | LEADS TS251 |
| Joint protein–peptide prediction | ESM-3 600M | Test167 |
| Joint protein–peptide prediction | ProtTransT5 | Test167 |
| Protein-side prediction | ESM-3 600M | TS092, TS125, TS251, TS639 |
| Protein-side prediction | ProtTransT5 | TS092, TS125, TS251, TS639 |
| Peptide-side prediction | ESM-3 600M | CAMP TS231 |
| Peptide-side prediction | ProtTransT5 | CAMP TS231 |
| Zero-shot prediction | ESM-3 600M | TS167 zero-shot |
| Zero-shot prediction | ProtTransT5 | TS167 zero-shot |

### Suggested local organization

```text
checkpoints/
├── joint/
├── protein_side/
├── peptide_side/
└── zero_shot/

data/
├── embeddings/
├── benchmark/
└── examples/
```

---

## Training and reproducibility

The main training scripts are:

```text
pretrain_propepx.py
finetune_propepx.py
```

### Pretraining: protein-side model

```bash
python pretrain_propepx.py \
  --mode prot \
  --emb_dim 1024 \
  --train_h5 "data/embeddings/Transfer_Learning_Training_embeddings.h5" \
  --val_h5 "data/embeddings/Transfer_Learning_Validation_embeddings.h5" \
  --ckpt_dir "checkpoints/pretrain/"
```

### Pretraining: peptide-side model

```bash
python pretrain_propepx.py \
  --mode pep \
  --emb_dim 1024 \
  --train_h5 "data/embeddings/TL_train_Prottran_embeddings.h5" \
  --val_h5 "data/embeddings/TL_validation_Prottran_embeddings.h5" \
  --ckpt_dir "checkpoints/pretrain/"
```

### Pretraining: joint model

```bash
python pretrain_propepx.py \
  --mode mode-GLOBAL \
  --emb_dim 1024 \
  --train_h5 "data/embeddings/Transfer_Learning_Training_embeddings.h5" \
  --val_h5 "data/embeddings/TL_validation_Prottran_embeddings.h5" \
  --ckpt_dir "checkpoints/pretrain/" \
  --gpu_id 0
```

### Fine-tuning: protein-side model

```bash
python finetune_propepx.py \
  --mode prot \
  --train_h5 "data/embeddings/ESM_train_TS092_Dataset.h5" \
  --val_h5 "data/embeddings/ESM_val_TS092_Dataset.h5" \
  --test_h5 "data/embeddings/ESM_test_TS092_Dataset.h5" \
  --pretrained_ckpt "checkpoints/pretrain/best_pretrain_prot.pt" \
  --ckpt_dir "checkpoints/finetune/prot_ts092/" \
  --emb_dim 1152 \
  --epochs 3 \
  --gpu_id 0
```

### Fine-tuning: peptide-side model

```bash
python finetune_propepx.py \
  --mode pep \
  --train_h5 "data/embeddings/CAMP_Train_Prottran_embeddings.h5" \
  --val_h5 "data/embeddings/CAMP_Test_Prottran_embeddings.h5" \
  --test_h5 "data/embeddings/CAMP_Test_Prottran_embeddings.h5" \
  --pretrained_ckpt "checkpoints/pretrain/best_pretrain_pep.pt" \
  --ckpt_dir "checkpoints/finetune/pep_camp231/" \
  --emb_dim 1024 \
  --epochs 3 \
  --gpu_id 0
```

### Fine-tuning: joint model

```bash
python finetune_propepx.py \
  --mode mode-GLOBAL \
  --train_h5 "data/embeddings/TrainingDataset-KGIPA_prot1418_embedding.h5" \
  --test_h5 "data/embeddings/Test167_embeddings.h5" \
  --pretrained_ckpt "checkpoints/pretrain/best_pretrain_global.pt" \
  --ckpt_dir "checkpoints/finetune/joint_test167/" \
  --emb_dim 1024 \
  --epochs 3 \
  --n_splits 5 \
  --gpu_id 0
```

---

## Output interpretation

| Output | Meaning |
|---|---|
| Residue probability | Probability that each residue participates in binding |
| Binary label | Binding/non-binding residue prediction after thresholding |
| HTML report | Interactive summary for proteins, peptides, and figures |
| Heatmap | Residue-level interaction or interpretability map |
| CSV file | Machine-readable prediction table |
| JSON file | Reproducible run metadata |

Example residue table:

| Chain | Residue index | Residue | Score | Prediction |
|---|---:|---:|---:|---|
| Protein | 1 | M | 0.12 | Non-binding |
| Protein | 2 | E | 0.79 | Binding |
| Peptide | 1 | K | 0.66 | Binding |

---

## Reproducibility checklist

Before submitting or reviewing results, verify:

- [ ] The Conda environment was created from `propepx.yml`.
- [ ] The correct checkpoint was selected for the prediction mode.
- [ ] The embedding backend matches the checkpoint family.
- [ ] Protein and peptide sequences contain valid amino-acid characters.
- [ ] Output directory exists and is writable.
- [ ] GPU ID is valid when using CUDA.
- [ ] Hugging Face model paths are synchronized with the local checkpoint paths.

---

## Citation

If you use ProPepX, please cite:

```bibtex
@article{propepx2026,
  title   = {ProPepX: interaction-aware transfer learning for protein-peptide binding residue prediction},
  author  = {Hussain, Syed Kumail and Chandra, Sourav and collaborators},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

Please see the `LICENSE` file for usage terms.

If this repository includes pretrained weights, datasets, or third-party models,
their use may also be governed by the licenses of the original providers
including PyTorch, Hugging Face, ProtTrans, ESM, and associated benchmark datasets.

---

## Contact

For questions, issues, and collaboration:

<table>
<tr>
<td>

**Syed Kumail Hussain Naqvi**  
Department of Physical-AI Convergence Engineering 
Jeonbuk National University, Republic of Korea

 <a href="mailto:syedkumailhussainnaqvi@jbnu.ac.kr">syedkumailhussainnaqvi@jbnu.ac.kr</a>  

 <a href="https://github.com/SyedKumailHussainNaqvi/ProPepX">GitHub Repository</a>  

<a href="https://huggingface.co/syedkumailhussain/ProPepX/tree/main">Hugging Face Model Hub</a>  

 <i>Manuscript under review.</i>

</td>
</tr>
</table>

Please open a **GitHub Issue** for reproducibility questions, installation problems, bug reports,
or requests for additional examples.
