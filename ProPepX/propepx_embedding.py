"""
propepx_embedding.py
=====================
ProPepX Embedding Generator
-----------------------------
Generates per-residue embeddings for protein and peptide sequences using:
    - ProtTrans T5-XL-UniRef50   (default, 1024-dim)
    - ESM-2 600M                 (esm2_t33_650M_UR50D, 1280-dim)
    - Both                       (when no model is specified)

Typical usage (called by propepx_predict.py):
    from propepx_embedding import embed_sequences

    prot_emb_prottrans, pep_emb_prottrans = embed_sequences(
        protein_seq="MKTL...",
        peptide_seq="ACDE...",
        embedding_model="prottrans",   # or "esm" or "both"
        device=device,
    )

Docker / reproducibility note
------------------------------
All model weights are downloaded to ~/.cache by HuggingFace / ESM at first
run and cached automatically.  Set env var HF_HOME to override the cache dir.

Authors: Syed Kumail Hussain Naqvi et al.
"""

import sys
import os
import re

import numpy as np
import torch


sys.path.append("/home/kumail/Transformer Model/ProPepX_complete_code/")


# ============================================================
# Constants
# ============================================================

MAX_PROT_LEN = 1418
MAX_PEP_LEN = 50

PROTTRANS_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
ESM_MODEL_NAME = "esmc_600m"

PROTTRANS_EMB_DIM = 1024
ESM_EMB_DIM = 1152


# ============================================================
# Sequence validation
# ============================================================

def _validate_sequence(seq: str, label: str, max_len: int) -> str:
    """
    Clean and validate an amino-acid sequence.

    Parameters
    ----------
    seq : str
        Input amino-acid sequence.
    label : str
        Name used in error messages, e.g. Protein or Peptide.
    max_len : int
        Maximum allowed sequence length.

    Returns
    -------
    str
        Cleaned uppercase sequence.
    """

    seq = seq.strip().upper().replace(" ", "").replace("\n", "")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    bad = set(seq) - valid_aa

    if bad:
        raise ValueError(
            f"{label} contains non-standard amino-acid characters: {bad}. "
            "Only the 20 canonical amino acids are supported."
        )

    if len(seq) > max_len:
        raise ValueError(
            f"{label} length {len(seq)} exceeds the maximum allowed length "
            f"of {max_len}."
        )

    if len(seq) == 0:
        raise ValueError(f"{label} sequence is empty.")

    return seq


# ============================================================
# ProtTrans T5-XL-UniRef50
# ============================================================

def _load_prottrans(device: torch.device):
    """
    Lazy-load ProtTrans T5-XL-UniRef50 tokenizer and encoder.
    """

    try:
        from transformers import T5Tokenizer, T5EncoderModel
    except ImportError:
        raise ImportError(
            "transformers package not found. Install with:\n"
            "pip install transformers sentencepiece"
        )

    print("  ↓  Loading ProtTrans T5-XL-UniRef50 …")

    tokenizer = T5Tokenizer.from_pretrained(
        PROTTRANS_MODEL_NAME,
        do_lower_case=False,
    )

    model = T5EncoderModel.from_pretrained(PROTTRANS_MODEL_NAME)
    model = model.eval().to(device)

    print("  ✔  ProtTrans loaded.\n")

    return tokenizer, model


@torch.no_grad()
def _embed_prottrans(
    seq: str,
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """
    Generate per-residue ProtTrans embeddings.

    Returns
    -------
    np.ndarray
        Shape: (L, 1024)
    """

    seq_spaced = " ".join(list(re.sub(r"[UZOB]", "X", seq)))

    ids = tokenizer(
        seq_spaced,
        add_special_tokens=True,
        padding=False,
        return_tensors="pt",
    )

    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    emb = output.last_hidden_state[0, :len(seq), :]

    return emb.detach().cpu().numpy().astype(np.float32)


# ============================================================
# ESM-C 600M / ESM-3 family
# ============================================================

def _load_esm(device: torch.device):
    """
    Lazy-load ESM-C 600M.

    This replaces the old Fair-ESM2 loading API:
        esm.pretrained.esm2_t33_650M_UR50D()

    ESM-C uses:
        ESMC.from_pretrained("esmc_600m")
    """

    try:
        from esm.models.esmc import ESMC
    except ImportError:
        raise ImportError(
            "ESM-C package not found. Install with:\n"
            "pip install esm"
        )

    print("  ↓  Loading ESM-C 600M …")

    model = ESMC.from_pretrained(ESM_MODEL_NAME).to(device)
    model.eval()

    print("  ✔  ESM-C 600M loaded.\n")

    return model


@torch.no_grad()
def _embed_esm(
    seq: str,
    esm_model,
    device: torch.device,
) -> np.ndarray:
    """
    Generate per-residue embeddings using ESM-C 600M.

    Returns
    -------
    np.ndarray
        Shape: (L, 1152)
    """

    try:
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError(
            "ESM SDK API not found. Install or update with:\n"
            "pip install -U esm"
        )

    protein = ESMProtein(sequence=seq)

    token_tensor = esm_model.encode(protein).to(device)

    output = esm_model.logits(
        token_tensor,
        LogitsConfig(
            sequence=True,
            return_embeddings=True,
        ),
    )

    emb = output.embeddings.squeeze(0).to(torch.float32)

    # Remove start and end special tokens.
    emb = emb[1:-1]

    if emb.shape[0] != len(seq):
        raise RuntimeError(
            f"ESM-C embedding length mismatch: got {emb.shape[0]}, "
            f"expected {len(seq)}."
        )

    return emb.detach().cpu().numpy().astype(np.float32)


# ============================================================
# Public API
# ============================================================

def embed_sequences(
    protein_seq: str,
    peptide_seq: str,
    embedding_model: str = "both",
    device: torch.device = None,
) -> dict:
    """
    Generate per-residue embeddings for one protein-peptide pair.

    Parameters
    ----------
    protein_seq : str
        Protein amino-acid sequence, maximum 1418 residues.
    peptide_seq : str
        Peptide amino-acid sequence, maximum 50 residues.
    embedding_model : str
        One of:
            - "prottrans"
            - "esm"
            - "both"
    device : torch.device, optional
        Compute device. If None, CUDA is used if available.

    Returns
    -------
    dict
        {
            "prottrans": (protein_embedding, peptide_embedding),
            "esm":       (protein_embedding, peptide_embedding),
        }

        Each embedding is a numpy array:
            ProtTrans : (L, 1024)
            ESM-C     : (L, 1152)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = embedding_model.lower().strip()

    if embedding_model not in ("prottrans", "esm", "both"):
        raise ValueError(
            "embedding_model must be one of: 'prottrans', 'esm', or 'both'."
        )

    protein_seq = _validate_sequence(
        protein_seq,
        label="Protein",
        max_len=MAX_PROT_LEN,
    )

    peptide_seq = _validate_sequence(
        peptide_seq,
        label="Peptide",
        max_len=MAX_PEP_LEN,
    )

    results = {}

    if embedding_model in ("prottrans", "both"):
        tokenizer, prottrans_model = _load_prottrans(device)

        prot_emb = _embed_prottrans(
            seq=protein_seq,
            tokenizer=tokenizer,
            model=prottrans_model,
            device=device,
        )

        pep_emb = _embed_prottrans(
            seq=peptide_seq,
            tokenizer=tokenizer,
            model=prottrans_model,
            device=device,
        )

        results["prottrans"] = (prot_emb, pep_emb)

        del prottrans_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if embedding_model in ("esm", "both"):
        esm_model = _load_esm(device)

        prot_emb = _embed_esm(
            seq=protein_seq,
            esm_model=esm_model,
            device=device,
        )

        pep_emb = _embed_esm(
            seq=peptide_seq,
            esm_model=esm_model,
            device=device,
        )

        results["esm"] = (prot_emb, pep_emb)

        del esm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
