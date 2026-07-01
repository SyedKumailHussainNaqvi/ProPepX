# ProPepX Model Weight and Test Dataset Features

Welcome to the **ProPepX Model Hub**.

This directory provides documentation for accessing the pretrained models, test dataset features, and inference resources used by **ProPepX**.

> **Note**
>
> The actual model weights and feature files are hosted on **Hugging Face** to ensure fast downloads, version control, and reproducible inference.

---

# Overview

The ProPepX Model Hub contains:

- Pretrained ProPepX model weights
- Fine-tuned ProPepX model weights
- Test dataset feature embeddings
- Protein-side models
- Peptide-side models
- Joint prediction models
- Zero-shot models
- ProtTransT5 models
- ESM-3 (600M) models

---

# Hugging Face Repository

All downloadable resources are available at:

## Repository

https://huggingface.co/syedkumailhussain/ProPepX

or

https://huggingface.co/syedkumailhussain/ProPepX/tree/main

---

# Repository Structure

```text
ProPepX
│
├── Joint-ProPep/
├── Protein-side/
├── Peptide-side/
└── Zero-shot/
```

Each directory contains the corresponding pretrained and fine-tuned weights and test feature embeddings required for reproducible inference.

---

# Included Resources

## Joint Prediction

- Protein + Peptide joint models
- LEADS TS251
- Test167
- ProtTransT5
- ESM-3 (600M)

---

## Protein-side Prediction

- TS092
- TS125
- TS251
- TS639

Available for

- ProtTransT5
- ESM-3 (600M)

---

## Peptide-side Prediction

- CAMP TS231

Available for

- ProtTransT5
- ESM-3 (600M)

---

## Zero-shot Prediction

Contains

- Zero-shot pretrained model
- Test167 feature embeddings

Supported embeddings

- ProtTransT5
- ESM-3 (600M)

---

# Using the Models

After downloading the required checkpoint and test feature files from Hugging Face, inference can be performed using:

```bash
python propepx_predict.py \
    --protein "<Protein Sequence>" \
    --peptide "<Peptide Sequence>" \
    --embedding esm \
    --mode mode-GLOBAL \
    --dataset leads_ts251
```

Please refer to the main repository README for complete installation instructions and additional prediction examples.

---



# Support

For installation issues or questions, please open an Issue in the GitHub repository.

GitHub

https://github.com/SyedKumailHussainNaqvi/ProPepX

Email

syedkumailhussainnaqvi@jbnu.ac.kr
