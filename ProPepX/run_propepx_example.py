"""
run_propepx_example.py
========================
ProPepX  –  End-to-End Example Script
---------------------------------------
This script demonstrates the full ProPepX pipeline for the PROTEIN mode
(prot) on a single A100 GPU using ProtTrans T5-XL embeddings:

    Stage 1 ── Transfer Pre-Training
                 Trains on pepPI Transfer-Learning embeddings (200 epochs)
                 Saves:  best_pretrain_prot.pt
                         last_pretrain_prot.pt
                         pretrain_history_prot.csv

    Stage 2 ── Fine-Tuning
                 Loads pre-trained checkpoint, fine-tunes on pepPI
                 binding-site labels (40 epochs, early stopping, patience=8)
                 Saves:  best_finetune_prot.pt
                         last_finetune_prot.pt
                         finetune_history_prot.csv
                         final_test_metrics_prot.csv

    Stage 3 ── Inference Demo
                 Accepts a raw protein + peptide sequence
                 Generates ProtTrans embeddings on the fly
                 Runs binding-site prediction with the fine-tuned checkpoint
                 Prints colour-coded results to terminal
                 Saves an HTML report

Paths used here match the lab storage layout; update them via the
CONFIG block at the top if your directory structure differs.

Usage
-----
    python run_propepx_example.py                         # full pipeline
    python run_propepx_example.py --skip_pretrain         # skip Stage 1
    python run_propepx_example.py --skip_pretrain \\
                                   --skip_finetune        # inference only

Authors: Syed Kumail Hussain Naqvi et al.
"""

# ─────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────
import argparse
import os
import sys

import torch

# ─────────────────────────────────────────────────────────────
# ProPepX package path
# ─────────────────────────────────────────────────────────────
PROPEPX_CODE_DIR = "/home/kumail/Transformer Model/ProPepX_complete_code"
sys.path.append(PROPEPX_CODE_DIR)

from Import_libraries import set_seed
from pretrain_propepx import run_pretraining
from finetune_propepx import run_finetuning
from propepx_predict  import ProPepXPredictor


# ╔══════════════════════════════════════════════════════════════╗
# ║                     CONFIGURATION                           ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  All paths, hyperparameters, and mode settings live here.   ║
# ║  No other section of the script needs to be edited.         ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Hardware ──────────────────────────────────────────────────
GPU_ID   = "0"          # CUDA device ID
SEED     = 42

# ── Task mode ─────────────────────────────────────────────────
MODE     = "prot"       # "prot" | "pep" | "mode-GLOBAL"

# ── Embedding model ───────────────────────────────────────────
EMBEDDING_MODEL = "prottrans"   # "prottrans" | "esm" | "both"

# ── Fine-tune dataset (for Stage 3 inference) ────────────────
DATASET  = "peppi"      # "pepbdb" | "peppi" | "benchmark"

# ── Pre-training data (HDF5 embeddings, ProtTrans, prot mode) ─
PRETRAIN_TRAIN_H5 = (
    "/media/8TB_hardisk/kumail/Transformer Model/"
    "Embedding Bechmark Dtaatset/"
    "Transfer_Learning_Training_embeddings.h5"
)
PRETRAIN_VAL_H5 = (
    "/media/8TB_hardisk/kumail/Transformer Model/"
    "Embedding Bechmark Dtaatset/"
    "Transfer_Learning_Validation_embeddings.h5"
)

# ── Pre-training checkpoint output ────────────────────────────
PRETRAIN_CKPT_DIR = (
    "/home/kumail/Transformer Model/updatedModel-pepPI/"
    "Transfer Learning/Model_10M_parameters/check_Pre_Train_weight/"
)

# ── Fine-tuning data ──────────────────────────────────────────
FINETUNE_TRAIN_H5 = (
    "/media/8TB_hardisk/kumail/Transformer Model/"
    "Embedding Bechmark Dtaatset/pepPI/"
    "pepPI_prot_train_prottrans.h5"
)
FINETUNE_VAL_H5 = (
    "/media/8TB_hardisk/kumail/Transformer Model/"
    "Embedding Bechmark Dtaatset/pepPI/"
    "pepPI_prot_val_prottrans.h5"
)
FINETUNE_TEST_H5 = (
    "/media/8TB_hardisk/kumail/Transformer Model/"
    "Embedding Bechmark Dtaatset/pepPI/"
    "pepPI_prot_test_prottrans.h5"
)

# ── Fine-tuning checkpoint output ────────────────────────────
FINETUNE_CKPT_DIR = (
    "/home/kumail/Transformer Model/updatedModel-pepPI/"
    "Fine_Tune_Weights/prot/pepPI/prottrans/"
)

# ── HTML report output directory ─────────────────────────────
HTML_REPORT_DIR = (
    "/home/kumail/Transformer Model/updatedModel-pepPI/reports/"
)

# ─────────────────────────────────────────────────────────────
# Pre-Training Hyperparameters  (paper values)
# ─────────────────────────────────────────────────────────────
PRETRAIN_HP = dict(
    emb_dim          = 1024,
    hidden_dim       = 512,
    heads            = 8,
    dropout          = 0.35,
    num_self_layers  = 2,
    ff_dim           = 512,
    max_len_prot     = 1418,
    max_len_pep      = 50,
    epochs           = 200,
    train_batch_size = 64,
    val_batch_size   = 64,
    lr               = 1e-5,
    weight_decay     = 1e-4,
    warmup_ratio     = 0.01,
    num_workers      = 4,
    pin_memory       = True,
)

# ─────────────────────────────────────────────────────────────
# Fine-Tuning Hyperparameters  (paper values)
# ─────────────────────────────────────────────────────────────
FINETUNE_HP = dict(
    emb_dim              = 1024,
    hidden_dim           = 512,
    heads                = 8,
    dropout              = 0.35,
    num_self_layers      = 2,
    ff_dim               = 512,
    max_len_prot         = 1418,
    max_len_pep          = 50,
    epochs               = 40,
    patience             = 8,
    train_batch_size     = 32,
    val_batch_size       = 8,
    test_batch_size      = 32,
    lr                   = 1e-5,
    weight_decay         = 1e-4,
    warmup_ratio         = 0.01,
    freeze_first_n_epochs= 0,
    fixed_threshold      = 0.50,
    num_workers          = 4,
    pin_memory           = True,
)

# ─────────────────────────────────────────────────────────────
# Inference Demo Sequences
# ─────────────────────────────────────────────────────────────
#   Source: PDB 1IW4 – Trypsin/BPTI complex
#   Protein: Trypsin (first 50 residues shown; replace with any ≤1418 aa)
#   Peptide: BPTI (Kunitz domain; ≤50 aa)
DEMO_PROTEIN_SEQ = (
    "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"
)
DEMO_PEPTIDE_SEQ = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGG"


# ═══════════════════════════════════════════════════════════════
# Stage 1 – Pre-Training
# ═══════════════════════════════════════════════════════════════

def stage_pretrain(device: torch.device) -> str:
    """
    Run transfer pre-training and return the path to the best checkpoint.

    The best checkpoint is automatically loaded by ``run_pretraining``
    before returning the model.  We only need the path here for Stage 2.
    """
    print("\n" + "═" * 70)
    print("  STAGE 1  –  TRANSFER PRE-TRAINING")
    print("═" * 70)
    print(f"  Mode             : {MODE}")
    print(f"  Embedding        : {EMBEDDING_MODEL.upper()}")
    print(f"  Training data    : {PRETRAIN_TRAIN_H5}")
    print(f"  Validation data  : {PRETRAIN_VAL_H5}")
    print(f"  Checkpoint dir   : {PRETRAIN_CKPT_DIR}")
    print(f"  Epochs           : {PRETRAIN_HP['epochs']}")
    print(f"  LR               : {PRETRAIN_HP['lr']}")
    print(f"  Batch size       : {PRETRAIN_HP['train_batch_size']}")
    print("═" * 70 + "\n")

    _model, best_path, _last, _tl, _vl = run_pretraining(
        train_h5 = PRETRAIN_TRAIN_H5,
        val_h5   = PRETRAIN_VAL_H5,
        ckpt_dir = PRETRAIN_CKPT_DIR,
        mode     = MODE,
        device   = device,
        **PRETRAIN_HP,
    )

    print(f"\n  ✅  Stage 1 complete.  Best checkpoint → {best_path}\n")
    return best_path


# ═══════════════════════════════════════════════════════════════
# Stage 2 – Fine-Tuning
# ═══════════════════════════════════════════════════════════════

def stage_finetune(pretrained_ckpt: str, device: torch.device) -> str:
    """
    Fine-tune from the pre-trained checkpoint and return the best path.
    """
    print("\n" + "═" * 70)
    print("  STAGE 2  –  FINE-TUNING")
    print("═" * 70)
    print(f"  Mode             : {MODE}")
    print(f"  Embedding        : {EMBEDDING_MODEL.upper()}")
    print(f"  Pre-trained ckpt : {pretrained_ckpt}")
    print(f"  Train data       : {FINETUNE_TRAIN_H5}")
    print(f"  Val   data       : {FINETUNE_VAL_H5}")
    print(f"  Test  data       : {FINETUNE_TEST_H5}")
    print(f"  Checkpoint dir   : {FINETUNE_CKPT_DIR}")
    print(f"  Epochs           : {FINETUNE_HP['epochs']}")
    print(f"  Early-stop pat.  : {FINETUNE_HP['patience']}")
    print(f"  LR               : {FINETUNE_HP['lr']}")
    print(f"  Batch size       : {FINETUNE_HP['train_batch_size']}")
    print("═" * 70 + "\n")

    _model, best_path, _tl, _vl, test_metrics = run_finetuning(
        train_h5        = FINETUNE_TRAIN_H5,
        val_h5          = FINETUNE_VAL_H5,
        test_h5         = FINETUNE_TEST_H5,
        pretrained_ckpt = pretrained_ckpt,
        ckpt_dir        = FINETUNE_CKPT_DIR,
        mode            = MODE,
        device          = device,
        **FINETUNE_HP,
    )

    print("\n  ── Final Test Set Metrics ──")
    for k, v in test_metrics.items():
        tag = f"    {k:<25}"
        print(f"{tag} {v:.4f}" if isinstance(v, float) else f"{tag} {v}")

    print(f"\n  ✅  Stage 2 complete.  Best checkpoint → {best_path}\n")
    return best_path


# ═══════════════════════════════════════════════════════════════
# Stage 3 – Inference Demo
# ═══════════════════════════════════════════════════════════════

def stage_inference(finetune_ckpt: str) -> None:
    """
    Run binding-site prediction on the demo sequences.

    Embeddings are generated fresh from the raw sequences (no HDF5 needed).
    The fine-tuned checkpoint loaded here is the *exact* one produced by
    Stage 2, so the result is fully reproducible.
    """
    print("\n" + "═" * 70)
    print("  STAGE 3  –  INFERENCE  (DEMO SEQUENCES)")
    print("═" * 70)
    print(f"  Protein ({len(DEMO_PROTEIN_SEQ)} aa) :")
    print(f"    {DEMO_PROTEIN_SEQ[:80]}{'…' if len(DEMO_PROTEIN_SEQ) > 80 else ''}")
    print(f"  Peptide ({len(DEMO_PEPTIDE_SEQ)} aa) :")
    print(f"    {DEMO_PEPTIDE_SEQ}")
    print("═" * 70 + "\n")

    # Point the config registry at the checkpoint we just trained
    # (Normally propepx_config.py holds these paths statically;
    #  here we override programmatically for the example.)
    import propepx_config as cfg
    cfg.FINETUNE_WEIGHT_REGISTRY[MODE][DATASET][EMBEDDING_MODEL] = finetune_ckpt

    predictor = ProPepXPredictor(gpu_id=int(GPU_ID))

    result = predictor.predict(
        protein_seq     = DEMO_PROTEIN_SEQ,
        peptide_seq     = DEMO_PEPTIDE_SEQ,
        embedding_model = EMBEDDING_MODEL,
        propepx_mode    = MODE,
        dataset         = DATASET,
        render          = True,          # colour-coded terminal output
        save_html       = HTML_REPORT_DIR,
    )

    # Print structured summary
    prot_pred = result.get("prot_pred")
    pep_pred  = result.get("pep_pred")

    print("\n" + "─" * 60)
    print("  STRUCTURED OUTPUT  (per-residue 0/1 binding predictions)")
    print("─" * 60)

    if prot_pred is not None:
        binding_idx = [i + 1 for i, p in enumerate(prot_pred) if p == 1]
        print(f"\n  Protein binding residue indices ({len(binding_idx)} sites):")
        for i in range(0, len(binding_idx), 20):
            chunk = binding_idx[i:i + 20]
            print("    " + "  ".join(str(x) for x in chunk))
        print(f"\n  Binary vector (first 50 residues):  "
              f"{''.join(str(p) for p in prot_pred[:50])}")

    if pep_pred is not None:
        binding_idx = [i + 1 for i, p in enumerate(pep_pred) if p == 1]
        print(f"\n  Peptide binding residue indices ({len(binding_idx)} sites):")
        print("    " + "  ".join(str(x) for x in binding_idx))
        print(f"\n  Binary vector:  {''.join(str(p) for p in pep_pred)}")

    print("\n  ✅  Stage 3 complete.")
    if HTML_REPORT_DIR:
        html_name = f"propepx_{MODE}_{EMBEDDING_MODEL}_{DATASET}.html"
        print(f"  📄  HTML report → {os.path.join(HTML_REPORT_DIR, html_name)}")


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProPepX end-to-end example (pre-train → fine-tune → predict)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip_pretrain", action="store_true",
        help=(
            "Skip Stage 1 (pre-training) and use an existing checkpoint.\n"
            "Requires --pretrained_ckpt."
        ),
    )
    parser.add_argument(
        "--skip_finetune", action="store_true",
        help=(
            "Skip Stage 2 (fine-tuning) and use an existing fine-tuned checkpoint.\n"
            "Requires --finetuned_ckpt."
        ),
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default=os.path.join(PRETRAIN_CKPT_DIR, "best_pretrain_prot.pt"),
        help="Path to an existing pre-trained checkpoint (used when --skip_pretrain).",
    )
    parser.add_argument(
        "--finetuned_ckpt",
        default=os.path.join(FINETUNE_CKPT_DIR, "best_finetune_prot.pt"),
        help="Path to an existing fine-tuned checkpoint (used when --skip_finetune).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── GPU & seed setup ─────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)

    if torch.cuda.is_available():
        print(f"\n  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device: {device}")

    # ── Stage 1: Pre-Training ────────────────────────────────
    if args.skip_pretrain:
        pretrained_ckpt = args.pretrained_ckpt
        print(f"\n  ⏭  Skipping Stage 1  |  using checkpoint: {pretrained_ckpt}")
    else:
        pretrained_ckpt = stage_pretrain(device)

    # ── Stage 2: Fine-Tuning ─────────────────────────────────
    if args.skip_finetune:
        finetuned_ckpt = args.finetuned_ckpt
        print(f"\n  ⏭  Skipping Stage 2  |  using checkpoint: {finetuned_ckpt}")
    else:
        finetuned_ckpt = stage_finetune(pretrained_ckpt, device)

    # ── Stage 3: Inference Demo ───────────────────────────────
    stage_inference(finetuned_ckpt)

    print("\n" + "═" * 70)
    print("  🏁  ProPepX full pipeline complete.")
    print("═" * 70 + "\n")
