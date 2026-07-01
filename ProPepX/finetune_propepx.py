"""
finetune_propepx.py
====================
ProPepX Fine-Tuning Script
---------------------------
Supports all task modes:
    - prot        : protein-only binding-site fine-tuning
    - pep         : peptide-only binding-site fine-tuning
    - mode-GLOBAL : joint (protein + peptide) global fine-tuning with 5-fold CV
    - zero-shot   : direct evaluation of a pre-trained checkpoint (no gradient updates)

Entry Points
------------
``run_finetuning``          – standard train/val/test fine-tuning (prot / pep modes)
``run_5fold_cv_global``     – 5-fold cross-validated fine-tuning (mode-GLOBAL)
``run_zero_shot_evaluation``– zero-shot evaluation against a test set

Saved Artefacts (per run)
--------------------------
    best_finetune_<mode>.pt       – best validation checkpoint
    last_finetune_<mode>.pt       – final-epoch checkpoint
    finetune_history_<mode>.csv   – per-epoch train/val metrics
    final_test_metrics_<mode>.csv – held-out test metrics (standard run)
    all_5fold_<mode>_metrics.csv  – per-fold test metrics (CV run)
    mean_std_5fold_<mode>.csv     – mean ± std summary (CV run)
    zero_shot_metrics_<mode>.csv  – zero-shot evaluation metrics

Usage
-----
Fine-tune (prot / pep):
    python finetune_propepx.py --mode prot \\
        --train_h5 /path/train.h5 --val_h5 /path/val.h5 --test_h5 /path/test.h5 \\
        --pretrained_ckpt /path/best_pretrain_prot.pt \\
        --ckpt_dir /path/to/output --gpu_id 0

Fine-tune (mode-GLOBAL, 5-fold):
    python finetune_propepx.py --mode mode-GLOBAL \\
        --train_h5 /path/train.h5 --test_h5 /path/test.h5 \\
        --pretrained_ckpt /path/best_pretrain_mode-GLOBAL.pt \\
        --ckpt_dir /path/to/output --gpu_id 0

Zero-shot evaluation:
    python finetune_propepx.py --mode mode-GLOBAL --zero_shot \\
        --test_h5 /path/test.h5 \\
        --pretrained_ckpt /path/best_pretrain_mode-GLOBAL.pt \\
        --ckpt_dir /path/to/output --gpu_id 0

Authors: Syed Kumail Hussain Naqvi et al.
"""

# ============================================================
# Standard Library & Third-Party Imports
# ============================================================
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ============================================================
# ProPepX Package Imports
# ============================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CoBindingCNN import ProPepX
from compute_loss import compute_loss
from Dataset_Preprocessing import H5PairDataset
from Import_libraries import set_seed
from lr_scheduler import get_cosine_schedule_with_warmup
from model_load_weight_utils import load_weight
from pepPI_binding_site_collate import collate_fn
from propepx_metrics import compute_joint_mode_metrics, compute_prot_metrics

# ============================================================
# Fine-Tuning Hyperparameters
# These are the canonical parameters used in the ProPepX paper.
# ============================================================

# prot / pep mode
FINETUNE_DEFAULTS_PROT_PEP = dict(
    emb_dim=1024,
    hidden_dim=512,
    heads=8,
    dropout=0.35,
    num_self_layers=2,
    ff_dim=512,
    max_len_prot=1418,
    max_len_pep=50,
    epochs=40,
    patience=8,
    train_batch_size=32,
    val_batch_size=8,
    test_batch_size=32,
    lr=1e-5,
    weight_decay=1e-4,
    warmup_ratio=0.01,
    freeze_first_n_epochs=0,
    fixed_threshold=0.50,
    num_workers=4,
    pin_memory=True,
    seed=42,
)

# mode-GLOBAL (5-fold cross-validation)
FINETUNE_DEFAULTS_GLOBAL = dict(
    emb_dim=1024,
    hidden_dim=512,
    heads=8,
    dropout=0.35,
    num_self_layers=2,
    ff_dim=512,
    max_len_prot=1418,
    max_len_pep=50,
    epochs=40,
    patience=8,
    train_batch_size=16,
    val_batch_size=8,
    test_batch_size=16,
    lr=3e-6,
    weight_decay=1e-4,
    warmup_ratio=0.05,
    freeze_first_n_epochs=3,
    fixed_threshold=0.50,
    selection_metric="mean_MCC",  # mean_MCC | mean_AUPR | mean_AUROC
    n_splits=5,
    seed=42,
    num_workers=4,
    pin_memory=True,
)

# zero-shot (evaluation only)
ZEROSHOT_DEFAULTS = dict(
    emb_dim=1024,
    hidden_dim=512,
    heads=8,
    dropout=0.35,
    num_self_layers=2,
    ff_dim=512,
    max_len_prot=1418,
    max_len_pep=50,
    fixed_threshold=0.50,
    test_batch_size=16,
    num_workers=4,
    pin_memory=True,
    seed=42,
)


# ============================================================
# Utilities
# ============================================================

def print_model_parameters(model: torch.nn.Module) -> int:
    """Print all trainable parameter names, shapes, and counts."""
    total = 0
    print("\n" + "=" * 70)
    print("  ProPepX – Trainable Parameters")
    print("=" * 70)
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            print(f"  {name:<60s}  {str(list(param.shape)):<25s}  {count:>12,}")
            total += count
    print("=" * 70)
    print(f"  Total Trainable Parameters : {total:,}")
    print("=" * 70 + "\n")
    return total


def load_pretrained_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a ProPepX pre-trained checkpoint into *model* with positional-
    embedding size adaptation and partial-load (strict=False) support.
    """
    model = load_weight(
        model=model,
        weight_path=ckpt_path,
        device=device,
        strict=False,
        verbose=True,
        resize_positional_embeddings=True,
    )
    return model


def freeze_lower_layers(model: torch.nn.Module) -> None:
    """
    Freeze CNN feature-extraction and input-projection layers at the start
    of fine-tuning to protect transferred representations.
    """
    freeze_prefixes = [
        "prot_proj", "prot_pos", "prot_cnn",
        "pep_proj",  "pep_pos",  "pep_cnn",
    ]
    frozen = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in freeze_prefixes):
            param.requires_grad = False
            frozen.append(name)

    print(f"\n  {len(frozen)} parameter groups frozen:")
    for n in frozen:
        print(f"       {n}")
    print()


def unfreeze_all_layers(model: torch.nn.Module) -> None:
    """Restore gradient computation for all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print(" All layers unfrozen – full fine-tuning active.\n")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    extra: dict = None,
) -> None:
    """Serialise a full training state to disk."""
    payload = {
        "epoch":           epoch,
        "best_metric":     best_metric,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  ✔  Checkpoint saved → {path}")


# ============================================================
# Metrics
# ============================================================

def _compute_global_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """
    Residue-level metrics with padding-index (-100) filtering.
    Used for both ``prot`` and ``mode-GLOBAL`` evaluation.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    valid  = y_true != -100
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return dict(MCC=0.0, ACC=0.0, F1=0.0, AUROC=np.nan,
                    AUPR=np.nan, Positive=0, Negative=0, Total=0)

    y_pred = (y_prob >= threshold).astype(int)

    def _safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return np.nan

    return {
        "MCC":      _safe(matthews_corrcoef, y_true, y_pred)
                    if len(np.unique(y_true)) > 1 else 0.0,
        "ACC":      accuracy_score(y_true, y_pred),
        "F1":       f1_score(y_true, y_pred, zero_division=0),
        "AUROC":    _safe(roc_auc_score, y_true, y_prob)
                    if len(np.unique(y_true)) > 1 else np.nan,
        "AUPR":     _safe(average_precision_score, y_true, y_prob)
                    if len(np.unique(y_true)) > 1 else np.nan,
        "Recall":   recall_score(y_true, y_pred, zero_division=0),
        "Precision":precision_score(y_true, y_pred, zero_division=0),
        "Positive": int(np.sum(y_true == 1)),
        "Negative": int(np.sum(y_true == 0)),
        "Total":    int(len(y_true)),
    }


# ============================================================
# DataLoader Builders
# ============================================================

def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_finetune_loaders(
    train_h5: str,
    val_h5: str,
    test_h5: str,
    mode: str,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple:
    """Standard train / val / test loaders for prot or pep fine-tuning."""
    train_ds = H5PairDataset(train_h5, mode=mode)
    val_ds   = H5PairDataset(val_h5,   mode=mode)
    test_ds  = H5PairDataset(test_h5,  mode=mode)

    print("\n" + "=" * 50)
    print(f"  Fine-Tune Dataset Summary  [mode = {mode}]")
    print("=" * 50)
    print(f"  TRAIN : {len(train_ds):>8,}  |  batch = {train_batch_size}")
    print(f"  VAL   : {len(val_ds):>8,}  |  batch = {val_batch_size}")
    print(f"  TEST  : {len(test_ds):>8,}  |  batch = {test_batch_size}")
    print("=" * 50 + "\n")

    return (
        _make_loader(train_ds, train_batch_size, True,  num_workers, pin_memory),
        _make_loader(val_ds,   val_batch_size,   False, num_workers, pin_memory),
        _make_loader(test_ds,  test_batch_size,  False, num_workers, pin_memory),
    )


def build_cv_loaders(
    train_h5: str,
    test_h5: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    mode: str,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple:
    """Fold-specific loaders for 5-fold cross-validation (mode-GLOBAL)."""
    full_ds  = H5PairDataset(train_h5, mode=mode)
    test_ds  = H5PairDataset(test_h5,  mode=mode)

    train_sub = Subset(full_ds, train_indices)
    val_sub   = Subset(full_ds, val_indices)

    print(f"  TRAIN fold : {len(train_sub):>6,}  |  VAL fold : {len(val_sub):>6,}"
          f"  |  TEST : {len(test_ds):>6,}")

    return (
        _make_loader(train_sub, train_batch_size, True,  num_workers, pin_memory),
        _make_loader(val_sub,   val_batch_size,   False, num_workers, pin_memory),
        _make_loader(test_ds,   test_batch_size,  False, num_workers, pin_memory),
    )


# ============================================================
# Core Training / Evaluation Loops
# ============================================================

def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mode: str,
) -> float:
    """
    Single fine-tuning epoch.

    Performs forward pass, composite loss computation (CE + Focal + Tversky),
    gradient clipping (max-norm = 1.0), AdamW update, and LR scheduler step.
    """
    model.train()
    total_loss = 0.0

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="  Fine-Tune", leave=False):
        prot_emb   = prot_emb.to(device,   non_blocking=True)
        pep_emb    = pep_emb.to(device,    non_blocking=True)
        prot_label = prot_label.to(device, non_blocking=True)
        pep_label  = pep_label.to(device,  non_blocking=True)

        prot_mask = prot_emb.abs().sum(dim=-1) == 0
        pep_mask  = pep_emb.abs().sum(dim=-1)  == 0

        optimizer.zero_grad()

        prot_logits, pep_logits = model(
            prot_emb=prot_emb,
            pep_emb=pep_emb,
            prot_mask=prot_mask,
            pep_mask=pep_mask,
        )

        loss = compute_loss(
            prot_logits=prot_logits,
            pep_logits=pep_logits,
            prot_labels=prot_label,
            pep_labels=pep_label,
            MODE=mode,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    threshold: float,
) -> tuple:
    """
    Evaluation loop shared by validation and test phases.

    Computes loss and, for metrics:
      - prot / pep      → global residue-level metrics (MCC, AUROC, AUPR, F1 …)
      - mode-GLOBAL     → separate protein and peptide residue-level metrics

    Returns
    -------
    avg_loss   : float
    prot_metrics : dict  (None for pep-only mode)
    pep_metrics  : dict  (None for prot-only mode)
    """
    model.eval()
    total_loss = 0.0

    all_prot_true, all_prot_prob = [], []
    all_pep_true,  all_pep_prob  = [], []

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="  Evaluate ", leave=False):
        prot_emb   = prot_emb.to(device,   non_blocking=True)
        pep_emb    = pep_emb.to(device,    non_blocking=True)
        prot_label = prot_label.to(device, non_blocking=True)
        pep_label  = pep_label.to(device,  non_blocking=True)

        prot_mask = prot_emb.abs().sum(dim=-1) == 0
        pep_mask  = pep_emb.abs().sum(dim=-1)  == 0

        prot_logits, pep_logits = model(
            prot_emb=prot_emb,
            pep_emb=pep_emb,
            prot_mask=prot_mask,
            pep_mask=pep_mask,
        )

        loss = compute_loss(
            prot_logits=prot_logits,
            pep_logits=pep_logits,
            prot_labels=prot_label,
            pep_labels=pep_label,
            MODE=mode,
        )
        total_loss += loss.item()

        if prot_logits is not None and mode in ("prot", "both", "mode-GLOBAL"):
            prob = torch.softmax(prot_logits, dim=-1)[..., 1]
            all_prot_true.append(prot_label.cpu().numpy().reshape(-1))
            all_prot_prob.append(prob.cpu().numpy().reshape(-1))

        if pep_logits is not None and mode in ("pep", "both", "mode-GLOBAL"):
            prob = torch.softmax(pep_logits, dim=-1)[..., 1]
            all_pep_true.append(pep_label.cpu().numpy().reshape(-1))
            all_pep_prob.append(prob.cpu().numpy().reshape(-1))

    avg_loss = total_loss / len(loader)

    prot_metrics = (
        _compute_global_metrics(
            np.concatenate(all_prot_true),
            np.concatenate(all_prot_prob),
            threshold,
        ) if all_prot_true else None
    )

    pep_metrics = (
        _compute_global_metrics(
            np.concatenate(all_pep_true),
            np.concatenate(all_pep_prob),
            threshold,
        ) if all_pep_true else None
    )

    return avg_loss, prot_metrics, pep_metrics


def _build_optimizer_scheduler(model, lr, weight_decay, warmup_ratio, epochs, loader_len):
    """Build AdamW + cosine-warmup scheduler over trainable parameters."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    total_steps  = loader_len * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    return optimizer, scheduler


def _print_epoch_metrics(prefix, metrics):
    if metrics is None:
        return
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {prefix} {k:<15s}: {v:.4f}")
        else:
            print(f"    {prefix} {k:<15s}: {v}")


# ============================================================
# Standard Fine-Tuning  (prot / pep / mode-GLOBAL single split)
# ============================================================

def run_finetuning(
    train_h5: str,
    val_h5: str,
    test_h5: str,
    pretrained_ckpt: str,
    ckpt_dir: str,
    mode: str = "prot",
    device: torch.device = None,
    # --- Architecture ---
    emb_dim: int         = FINETUNE_DEFAULTS_PROT_PEP["emb_dim"],
    hidden_dim: int      = FINETUNE_DEFAULTS_PROT_PEP["hidden_dim"],
    heads: int           = FINETUNE_DEFAULTS_PROT_PEP["heads"],
    dropout: float       = FINETUNE_DEFAULTS_PROT_PEP["dropout"],
    num_self_layers: int = FINETUNE_DEFAULTS_PROT_PEP["num_self_layers"],
    ff_dim: int          = FINETUNE_DEFAULTS_PROT_PEP["ff_dim"],
    max_len_prot: int    = FINETUNE_DEFAULTS_PROT_PEP["max_len_prot"],
    max_len_pep: int     = FINETUNE_DEFAULTS_PROT_PEP["max_len_pep"],
    # --- Optimisation ---
    epochs: int              = FINETUNE_DEFAULTS_PROT_PEP["epochs"],
    patience: int            = FINETUNE_DEFAULTS_PROT_PEP["patience"],
    train_batch_size: int    = FINETUNE_DEFAULTS_PROT_PEP["train_batch_size"],
    val_batch_size: int      = FINETUNE_DEFAULTS_PROT_PEP["val_batch_size"],
    test_batch_size: int     = FINETUNE_DEFAULTS_PROT_PEP["test_batch_size"],
    lr: float                = FINETUNE_DEFAULTS_PROT_PEP["lr"],
    weight_decay: float      = FINETUNE_DEFAULTS_PROT_PEP["weight_decay"],
    warmup_ratio: float      = FINETUNE_DEFAULTS_PROT_PEP["warmup_ratio"],
    freeze_first_n_epochs: int = FINETUNE_DEFAULTS_PROT_PEP["freeze_first_n_epochs"],
    fixed_threshold: float   = FINETUNE_DEFAULTS_PROT_PEP["fixed_threshold"],
    # --- Misc ---
    num_workers: int     = FINETUNE_DEFAULTS_PROT_PEP["num_workers"],
    pin_memory: bool     = FINETUNE_DEFAULTS_PROT_PEP["pin_memory"],
) -> tuple:
    """
    Standard ProPepX fine-tuning for prot, pep, or mode-GLOBAL (single split).

    Model selection is performed on validation MCC (prot/pep) or mean MCC
    (mode-GLOBAL) using a fixed threshold of 0.50.

    Returns
    -------
    model            : fine-tuned ProPepX (best checkpoint loaded)
    best_ckpt_path   : str
    train_loss_history : list[float]
    val_loss_history   : list[float]
    test_metrics     : dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(ckpt_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. DataLoaders
    # ----------------------------------------------------------
    train_loader, val_loader, test_loader = build_finetune_loaders(
        train_h5=train_h5, val_h5=val_h5, test_h5=test_h5,
        mode=mode,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ----------------------------------------------------------
    # 2. Model  →  load pre-trained weights
    # ----------------------------------------------------------
    model = ProPepX(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=dropout,
        mode=mode,
        num_self_layers=num_self_layers,
        ff_dim=ff_dim,
        max_len_prot=max_len_prot,
        max_len_pep=max_len_pep,
    ).to(device)

    model = load_pretrained_weights(model, pretrained_ckpt, device)

    if freeze_first_n_epochs > 0:
        freeze_lower_layers(model)

    print_model_parameters(model)

    # ----------------------------------------------------------
    # 3. Optimiser & Scheduler
    # ----------------------------------------------------------
    optimizer, scheduler = _build_optimizer_scheduler(
        model, lr, weight_decay, warmup_ratio, epochs, len(train_loader)
    )

    # ----------------------------------------------------------
    # 4. Checkpoint paths
    # ----------------------------------------------------------
    best_ckpt_path    = os.path.join(ckpt_dir, f"best_finetune_{mode}.pt")
    last_ckpt_path    = os.path.join(ckpt_dir, f"last_finetune_{mode}.pt")
    history_csv_path  = os.path.join(ckpt_dir, f"finetune_history_{mode}.csv")
    test_csv_path     = os.path.join(ckpt_dir, f"final_test_metrics_{mode}.csv")

    # ----------------------------------------------------------
    # 5. Training Loop
    # ----------------------------------------------------------
    train_loss_history: list = []
    val_loss_history:   list = []
    history_rows:       list = []

    best_val_mcc      = -float("inf")
    best_epoch        = 0
    best_state        = None
    patience_counter  = 0

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*70}")
        print(f"  FINE-TUNE  EPOCH {epoch:>4d} / {epochs}   |   MODE = {mode}")
        print(f"{'='*70}")

        # -- Unfreeze lower layers after warm-up period ---------
        if freeze_first_n_epochs > 0 and epoch == freeze_first_n_epochs + 1:
            unfreeze_all_layers(model)
            # Rebuild optimiser over all params
            optimizer, scheduler = _build_optimizer_scheduler(
                model, lr, weight_decay, warmup_ratio,
                epochs - epoch + 1, len(train_loader)
            )

        # -- Forward / backward pass ----------------------------
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, device, mode)
        val_loss, val_prot, val_pep = _evaluate(model, val_loader, device, mode, fixed_threshold)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"  Train Loss : {train_loss:.6f}")
        print(f"  Val   Loss : {val_loss:.6f}")

        if val_prot:
            print("  [Protein]")
            _print_epoch_metrics("  ", val_prot)
        if val_pep:
            print("  [Peptide]")
            _print_epoch_metrics("  ", val_pep)

        # -- Compute selection score ----------------------------
        mcc_scores = [
            m["MCC"] for m in [val_prot, val_pep] if m is not None
        ]
        current_mcc = float(np.nanmean(mcc_scores)) if mcc_scores else -float("inf")
        print(f"  Val MCC (mean): {current_mcc:.4f}")

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_mean_mcc": current_mcc, "threshold": fixed_threshold}
        if val_prot:
            row.update({f"prot_{k}": v for k, v in val_prot.items()})
        if val_pep:
            row.update({f"pep_{k}":  v for k, v in val_pep.items()})
        history_rows.append(row)

        # -- Checkpoint management ------------------------------
        if current_mcc > best_val_mcc:
            best_val_mcc = current_mcc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_checkpoint(
                path=best_ckpt_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_metric=best_val_mcc,
                extra={"train_loss_history": train_loss_history,
                       "val_loss_history":   val_loss_history},
            )
            print(f"  🏆  New best at epoch {epoch}  |  mean MCC = {best_val_mcc:.4f}")
        else:
            patience_counter += 1
            print(f"   No improvement  ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n   Early stopping triggered after {patience} epochs.\n")
            break

    # ----------------------------------------------------------
    # 6. Load best state → evaluate on test set
    # ----------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(
        path=last_ckpt_path,
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=epoch, best_metric=best_val_mcc,
        extra={"train_loss_history": train_loss_history,
               "val_loss_history":   val_loss_history},
    )

    test_loss, test_prot, test_pep = _evaluate(model, test_loader, device, mode, fixed_threshold)

    print("\n" + "=" * 70)
    print("  FINAL TEST RESULTS")
    print("=" * 70)
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Best val MCC    : {best_val_mcc:.4f}")
    print(f"  Test loss       : {test_loss:.6f}")
    if test_prot:
        print("  [Protein]")
        _print_epoch_metrics("  ", test_prot)
    if test_pep:
        print("  [Peptide]")
        _print_epoch_metrics("  ", test_pep)
    print("=" * 70 + "\n")

    # ----------------------------------------------------------
    # 7. Save history & test metrics
    # ----------------------------------------------------------
    pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)

    test_row = {"best_epoch": best_epoch, "best_val_mcc": best_val_mcc,
                "test_loss": test_loss, "threshold": fixed_threshold}
    if test_prot:
        test_row.update({f"prot_{k}": v for k, v in test_prot.items()})
    if test_pep:
        test_row.update({f"pep_{k}":  v for k, v in test_pep.items()})
    pd.DataFrame([test_row]).to_csv(test_csv_path, index=False)

    print(f"   History CSV      → {history_csv_path}")
    print(f"   Test metrics CSV → {test_csv_path}")
    print(f"   Best checkpoint  → {best_ckpt_path}\n")

    test_metrics = {**(test_prot or {}), **(test_pep or {})}
    return model, best_ckpt_path, train_loss_history, val_loss_history, test_metrics


# ============================================================
# 5-Fold Cross-Validation Fine-Tuning  (mode-GLOBAL)
# ============================================================

def _run_one_cv_fold(
    fold_id: int,
    train_h5: str,
    test_h5: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    pretrained_ckpt: str,
    ckpt_dir: str,
    mode: str,
    device: torch.device,
    # Architecture
    emb_dim, hidden_dim, heads, dropout, num_self_layers, ff_dim,
    max_len_prot, max_len_pep,
    # Optimisation
    epochs, patience, train_batch_size, val_batch_size, test_batch_size,
    lr, weight_decay, warmup_ratio, freeze_first_n_epochs,
    # Evaluation
    fixed_threshold, selection_metric,
    num_workers, pin_memory,
) -> dict:
    """Run a single cross-validation fold for mode-GLOBAL fine-tuning."""

    fold_dir = os.path.join(ckpt_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_id}  –  mode-GLOBAL Fine-Tuning")
    print(f"{'='*70}")

    # -- DataLoaders ----------------------------------------
    train_loader, val_loader, test_loader = build_cv_loaders(
        train_h5=train_h5, test_h5=test_h5,
        train_indices=train_indices, val_indices=val_indices,
        mode=mode,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # -- Model ----------------------------------------------
    model = ProPepX(
        emb_dim=emb_dim, hidden_dim=hidden_dim, heads=heads,
        dropout=dropout, mode=mode, num_self_layers=num_self_layers,
        ff_dim=ff_dim, max_len_prot=max_len_prot, max_len_pep=max_len_pep,
    ).to(device)

    model = load_pretrained_weights(model, pretrained_ckpt, device)

    if freeze_first_n_epochs > 0:
        freeze_lower_layers(model)

    # -- Optimiser & Scheduler ------------------------------
    optimizer, scheduler = _build_optimizer_scheduler(
        model, lr, weight_decay, warmup_ratio, epochs, len(train_loader)
    )

    best_ckpt_path = os.path.join(fold_dir, f"best_fold_{fold_id}_global.pt")

    best_score       = -float("inf")
    best_epoch       = 0
    best_state       = None
    patience_counter = 0
    fold_history     = []

    # -- Epoch loop -----------------------------------------
    for epoch in range(1, epochs + 1):
        print(f"\n  ── FOLD {fold_id}  |  EPOCH {epoch}/{epochs}  ──")

        if freeze_first_n_epochs > 0 and epoch == freeze_first_n_epochs + 1:
            unfreeze_all_layers(model)
            optimizer, scheduler = _build_optimizer_scheduler(
                model, lr, weight_decay, warmup_ratio,
                epochs - epoch + 1, len(train_loader)
            )

        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, device, mode)
        val_loss, val_prot, val_pep = _evaluate(model, val_loader, device, mode, fixed_threshold)

        mean_mcc  = float(np.nanmean([val_prot["MCC"],  val_pep["MCC"]]))
        mean_aupr = float(np.nanmean([val_prot["AUPR"], val_pep["AUPR"]]))
        mean_auroc= float(np.nanmean([val_prot["AUROC"],val_pep["AUROC"]]))

        score_map = {
            "mean_MCC": mean_mcc, "mean_AUPR": mean_aupr, "mean_AUROC": mean_auroc
        }
        if selection_metric not in score_map:
            raise ValueError(f"selection_metric must be one of {list(score_map)}")

        global_score = score_map[selection_metric]

        row = {
            "fold": fold_id, "epoch": epoch,
            "train_loss": train_loss, "val_loss": val_loss,
            "prot_MCC": val_prot["MCC"],  "prot_AUPR": val_prot["AUPR"],
            "prot_AUROC": val_prot["AUROC"], "prot_F1": val_prot["F1"],
            "prot_ACC": val_prot["ACC"],
            "pep_MCC":  val_pep["MCC"],   "pep_AUPR": val_pep["AUPR"],
            "pep_AUROC": val_pep["AUROC"], "pep_F1":  val_pep["F1"],
            "pep_ACC":  val_pep["ACC"],
            "mean_MCC": mean_mcc, "mean_AUPR": mean_aupr, "mean_AUROC": mean_auroc,
        }
        fold_history.append(row)

        print(f"  Train Loss : {train_loss:.6f}  |  Val Loss : {val_loss:.6f}")
        print(f"  Prot MCC : {val_prot['MCC']:.4f}  |  Pep MCC : {val_pep['MCC']:.4f}"
              f"  |  Mean MCC : {mean_mcc:.4f}  |  Score ({selection_metric}) : {global_score:.4f}")

        if global_score > best_score:
            best_score = global_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_checkpoint(
                path=best_ckpt_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_metric=best_score,
                extra={"fold": fold_id, "selection_metric": selection_metric,
                       "fixed_threshold": fixed_threshold, "fold_history": fold_history},
            )
            print(f"   New best  |  {selection_metric} = {best_score:.4f}")
        else:
            patience_counter += 1
            print(f"   No improvement  ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"   Early stopping at fold {fold_id}, epoch {epoch}.\n")
            break

    # -- Test evaluation ------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_prot, test_pep = _evaluate(
        model, test_loader, device, mode, fixed_threshold
    )

    test_mean_mcc  = float(np.nanmean([test_prot["MCC"],   test_pep["MCC"]]))
    test_mean_aupr = float(np.nanmean([test_prot["AUPR"],  test_pep["AUPR"]]))
    test_mean_auroc= float(np.nanmean([test_prot["AUROC"], test_pep["AUROC"]]))

    fold_test_row = {
        "fold": fold_id, "best_epoch": best_epoch, "best_val_score": best_score,
        "test_loss": test_loss,
        "test_prot_MCC": test_prot["MCC"],   "test_prot_AUPR": test_prot["AUPR"],
        "test_prot_AUROC": test_prot["AUROC"],"test_prot_F1": test_prot["F1"],
        "test_prot_ACC": test_prot["ACC"],
        "test_prot_Positive": test_prot["Positive"],
        "test_prot_Negative": test_prot["Negative"],
        "test_pep_MCC": test_pep["MCC"],     "test_pep_AUPR": test_pep["AUPR"],
        "test_pep_AUROC": test_pep["AUROC"],  "test_pep_F1": test_pep["F1"],
        "test_pep_ACC": test_pep["ACC"],
        "test_pep_Positive": test_pep["Positive"],
        "test_pep_Negative": test_pep["Negative"],
        "test_mean_MCC": test_mean_mcc,
        "test_mean_AUPR": test_mean_aupr,
        "test_mean_AUROC": test_mean_auroc,
    }

    pd.DataFrame(fold_history).to_csv(
        os.path.join(fold_dir, f"fold_{fold_id}_history.csv"), index=False
    )
    pd.DataFrame([fold_test_row]).to_csv(
        os.path.join(fold_dir, f"fold_{fold_id}_test_metrics.csv"), index=False
    )

    print(f"\n  ── FOLD {fold_id}  RESULT ──"
          f"  Prot MCC : {test_prot['MCC']:.4f}"
          f"  |  Pep MCC : {test_pep['MCC']:.4f}"
          f"  |  Mean MCC : {test_mean_mcc:.4f}")

    return fold_test_row


def run_5fold_cv_global(
    train_h5: str,
    test_h5: str,
    pretrained_ckpt: str,
    ckpt_dir: str,
    mode: str = "mode-GLOBAL",
    device: torch.device = None,
    # Architecture
    emb_dim: int         = FINETUNE_DEFAULTS_GLOBAL["emb_dim"],
    hidden_dim: int      = FINETUNE_DEFAULTS_GLOBAL["hidden_dim"],
    heads: int           = FINETUNE_DEFAULTS_GLOBAL["heads"],
    dropout: float       = FINETUNE_DEFAULTS_GLOBAL["dropout"],
    num_self_layers: int = FINETUNE_DEFAULTS_GLOBAL["num_self_layers"],
    ff_dim: int          = FINETUNE_DEFAULTS_GLOBAL["ff_dim"],
    max_len_prot: int    = FINETUNE_DEFAULTS_GLOBAL["max_len_prot"],
    max_len_pep: int     = FINETUNE_DEFAULTS_GLOBAL["max_len_pep"],
    # Optimisation
    epochs: int              = FINETUNE_DEFAULTS_GLOBAL["epochs"],
    patience: int            = FINETUNE_DEFAULTS_GLOBAL["patience"],
    train_batch_size: int    = FINETUNE_DEFAULTS_GLOBAL["train_batch_size"],
    val_batch_size: int      = FINETUNE_DEFAULTS_GLOBAL["val_batch_size"],
    test_batch_size: int     = FINETUNE_DEFAULTS_GLOBAL["test_batch_size"],
    lr: float                = FINETUNE_DEFAULTS_GLOBAL["lr"],
    weight_decay: float      = FINETUNE_DEFAULTS_GLOBAL["weight_decay"],
    warmup_ratio: float      = FINETUNE_DEFAULTS_GLOBAL["warmup_ratio"],
    freeze_first_n_epochs: int = FINETUNE_DEFAULTS_GLOBAL["freeze_first_n_epochs"],
    fixed_threshold: float   = FINETUNE_DEFAULTS_GLOBAL["fixed_threshold"],
    selection_metric: str    = FINETUNE_DEFAULTS_GLOBAL["selection_metric"],
    n_splits: int            = FINETUNE_DEFAULTS_GLOBAL["n_splits"],
    seed: int                = FINETUNE_DEFAULTS_GLOBAL["seed"],
    num_workers: int         = FINETUNE_DEFAULTS_GLOBAL["num_workers"],
    pin_memory: bool         = FINETUNE_DEFAULTS_GLOBAL["pin_memory"],
) -> tuple:
    """
    5-Fold cross-validated fine-tuning for mode-GLOBAL.

    Each fold trains an independent ProPepX model initialised from the same
    pre-trained checkpoint. The held-out test set (test_h5) is evaluated once
    per fold using the best-validation checkpoint.

    Returns
    -------
    all_fold_df  : pd.DataFrame  – per-fold test metrics
    summary_df   : pd.DataFrame  – mean ± std across folds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(ckpt_dir, exist_ok=True)

    full_dataset = H5PairDataset(train_h5, mode=mode)
    indices = np.arange(len(full_dataset))

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(kfold.split(indices), start=1):
        fold_result = _run_one_cv_fold(
            fold_id=fold_id,
            train_h5=train_h5, test_h5=test_h5,
            train_indices=train_idx, val_indices=val_idx,
            pretrained_ckpt=pretrained_ckpt,
            ckpt_dir=ckpt_dir,
            mode=mode, device=device,
            emb_dim=emb_dim, hidden_dim=hidden_dim, heads=heads,
            dropout=dropout, num_self_layers=num_self_layers, ff_dim=ff_dim,
            max_len_prot=max_len_prot, max_len_pep=max_len_pep,
            epochs=epochs, patience=patience,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            lr=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio,
            freeze_first_n_epochs=freeze_first_n_epochs,
            fixed_threshold=fixed_threshold,
            selection_metric=selection_metric,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        all_fold_results.append(fold_result)

    all_fold_df = pd.DataFrame(all_fold_results)

    # -- Summary statistics -----------------------------------
    numeric_cols = all_fold_df.select_dtypes(include=[np.number]).columns.difference(["fold"])
    summary_rows = [
        {"Metric": col, "Mean": all_fold_df[col].mean(), "Std": all_fold_df[col].std()}
        for col in numeric_cols
    ]
    summary_df = pd.DataFrame(summary_rows)

    all_fold_csv  = os.path.join(ckpt_dir, f"all_{n_splits}fold_test_metrics.csv")
    summary_csv   = os.path.join(ckpt_dir, f"mean_std_{n_splits}fold_metrics.csv")
    all_fold_df.to_csv(all_fold_csv, index=False)
    summary_df.to_csv(summary_csv,   index=False)

    print("\n" + "=" * 70)
    print(f"   {n_splits}-Fold CV complete")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\n   All-fold CSV → {all_fold_csv}")
    print(f"   Summary CSV  → {summary_csv}\n")

    return all_fold_df, summary_df


# ============================================================
# Zero-Shot Evaluation  (no gradient updates)
# ============================================================

def run_zero_shot_evaluation(
    test_h5: str,
    pretrained_ckpt: str,
    ckpt_dir: str,
    mode: str = "mode-GLOBAL",
    device: torch.device = None,
    # Architecture
    emb_dim: int         = ZEROSHOT_DEFAULTS["emb_dim"],
    hidden_dim: int      = ZEROSHOT_DEFAULTS["hidden_dim"],
    heads: int           = ZEROSHOT_DEFAULTS["heads"],
    dropout: float       = ZEROSHOT_DEFAULTS["dropout"],
    num_self_layers: int = ZEROSHOT_DEFAULTS["num_self_layers"],
    ff_dim: int          = ZEROSHOT_DEFAULTS["ff_dim"],
    max_len_prot: int    = ZEROSHOT_DEFAULTS["max_len_prot"],
    max_len_pep: int     = ZEROSHOT_DEFAULTS["max_len_pep"],
    # Evaluation
    fixed_threshold: float = ZEROSHOT_DEFAULTS["fixed_threshold"],
    test_batch_size: int   = ZEROSHOT_DEFAULTS["test_batch_size"],
    num_workers: int       = ZEROSHOT_DEFAULTS["num_workers"],
    pin_memory: bool       = ZEROSHOT_DEFAULTS["pin_memory"],
) -> dict:
    """
    Zero-shot evaluation of a pre-trained ProPepX checkpoint.

    No gradient updates are performed.  The checkpoint is loaded and
    immediately evaluated on the supplied test HDF5.

    Returns
    -------
    metrics : dict  – {prot_MCC, prot_AUPR, pep_MCC, pep_AUPR, …}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ProPepX  ─  Zero-Shot Evaluation  [mode = {mode}]")
    print(f"{'='*70}")

    # -- DataLoader -------------------------------------------
    test_ds = H5PairDataset(test_h5, mode=mode)
    test_loader = _make_loader(test_ds, test_batch_size, False, num_workers, pin_memory)
    print(f"  TEST : {len(test_ds):,} samples  |  batch = {test_batch_size}\n")

    # -- Model ------------------------------------------------
    model = ProPepX(
        emb_dim=emb_dim, hidden_dim=hidden_dim, heads=heads,
        dropout=dropout, mode=mode, num_self_layers=num_self_layers,
        ff_dim=ff_dim, max_len_prot=max_len_prot, max_len_pep=max_len_pep,
    ).to(device)

    model = load_pretrained_weights(model, pretrained_ckpt, device)

    print_model_parameters(model)

    # -- Evaluate ---------------------------------------------
    test_loss, prot_metrics, pep_metrics = _evaluate(
        model, test_loader, device, mode, fixed_threshold
    )

    print("\n  ── Zero-Shot Results ──")
    print(f"  Test Loss  : {test_loss:.6f}")
    if prot_metrics:
        print("  [Protein]")
        _print_epoch_metrics("  ", prot_metrics)
    if pep_metrics:
        print("  [Peptide]")
        _print_epoch_metrics("  ", pep_metrics)

    # -- Save metrics -----------------------------------------
    results = {
        "mode": mode, "threshold": fixed_threshold, "test_loss": test_loss,
    }
    if prot_metrics:
        results.update({f"prot_{k}": v for k, v in prot_metrics.items()})
    if pep_metrics:
        results.update({f"pep_{k}":  v for k, v in pep_metrics.items()})

    if prot_metrics and pep_metrics:
        results["mean_MCC"]   = float(np.nanmean([prot_metrics["MCC"],   pep_metrics["MCC"]]))
        results["mean_AUPR"]  = float(np.nanmean([prot_metrics["AUPR"],  pep_metrics["AUPR"]]))
        results["mean_AUROC"] = float(np.nanmean([prot_metrics["AUROC"], pep_metrics["AUROC"]]))

    out_csv = os.path.join(ckpt_dir, f"zero_shot_metrics_{mode}.csv")
    pd.DataFrame([results]).to_csv(out_csv, index=False)
    print(f"\n   Zero-shot metrics → {out_csv}\n")

    return results


# ============================================================
# CLI Entry Point
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProPepX Fine-Tuning & Zero-Shot Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data paths -------------------------------------------
    parser.add_argument("--train_h5",      default=None, help="Training HDF5 (prot/pep/global).")
    parser.add_argument("--val_h5",        default=None, help="Validation HDF5 (prot/pep only).")
    parser.add_argument("--test_h5",       default=None, help="Test HDF5.")
    parser.add_argument("--pretrained_ckpt", required=True, help="Pre-trained ProPepX checkpoint (.pt).")
    parser.add_argument("--ckpt_dir",      required=True, help="Output directory for checkpoints & logs.")

    # -- Mode & flags -----------------------------------------
    parser.add_argument(
        "--mode", default="prot",
        choices=["prot", "pep", "mode-GLOBAL"],
        help="Task mode.",
    )
    parser.add_argument(
        "--zero_shot", action="store_true",
        help="Run zero-shot evaluation only (no training).",
    )

    # -- Hardware ---------------------------------------------
    parser.add_argument("--gpu_id", default="0", help="CUDA device ID.")

    # -- Architecture (paper defaults) -----------------------
    parser.add_argument("--emb_dim",         type=int,   default=1024)
    parser.add_argument("--hidden_dim",      type=int,   default=512)
    parser.add_argument("--heads",           type=int,   default=8)
    parser.add_argument("--dropout",         type=float, default=0.35)
    parser.add_argument("--num_self_layers", type=int,   default=2)
    parser.add_argument("--ff_dim",          type=int,   default=512)
    parser.add_argument("--max_len_prot",    type=int,   default=1418)
    parser.add_argument("--max_len_pep",     type=int,   default=50)

    # -- Fine-tuning optimisation (paper defaults) -----------
    parser.add_argument("--epochs",              type=int,   default=40)
    parser.add_argument("--patience",            type=int,   default=8)
    parser.add_argument("--train_batch_size",    type=int,   default=32)
    parser.add_argument("--val_batch_size",      type=int,   default=8)
    parser.add_argument("--test_batch_size",     type=int,   default=32)
    parser.add_argument("--lr",                  type=float, default=1e-5)
    parser.add_argument("--weight_decay",        type=float, default=1e-4)
    parser.add_argument("--warmup_ratio",        type=float, default=0.01)
    parser.add_argument("--freeze_first_n_epochs", type=int, default=0)
    parser.add_argument("--fixed_threshold",     type=float, default=0.50)

    # -- mode-GLOBAL / CV specific ---------------------------
    parser.add_argument("--n_splits",         type=int, default=5)
    parser.add_argument("--selection_metric", default="mean_MCC",
                        choices=["mean_MCC", "mean_AUPR", "mean_AUROC"])

    # -- Misc ------------------------------------------------
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  CUDA devices available : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    print(f"  Active device          : {device}  (GPU {args.gpu_id})\n")

    set_seed(args.seed)

    # --------------------------------------------------------
    # Zero-shot
    # --------------------------------------------------------
    if args.zero_shot:
        if args.test_h5 is None:
            raise ValueError("--test_h5 is required for zero-shot evaluation.")
        run_zero_shot_evaluation(
            test_h5=args.test_h5,
            pretrained_ckpt=args.pretrained_ckpt,
            ckpt_dir=args.ckpt_dir,
            mode=args.mode,
            device=device,
            emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, heads=args.heads,
            dropout=args.dropout, num_self_layers=args.num_self_layers, ff_dim=args.ff_dim,
            max_len_prot=args.max_len_prot, max_len_pep=args.max_len_pep,
            fixed_threshold=args.fixed_threshold,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
        )

    # --------------------------------------------------------
    # 5-fold CV (mode-GLOBAL)
    # --------------------------------------------------------
    elif args.mode == "mode-GLOBAL":
        if args.train_h5 is None or args.test_h5 is None:
            raise ValueError("--train_h5 and --test_h5 are required for mode-GLOBAL fine-tuning.")
        run_5fold_cv_global(
            train_h5=args.train_h5,
            test_h5=args.test_h5,
            pretrained_ckpt=args.pretrained_ckpt,
            ckpt_dir=args.ckpt_dir,
            mode=args.mode,
            device=device,
            emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, heads=args.heads,
            dropout=args.dropout, num_self_layers=args.num_self_layers, ff_dim=args.ff_dim,
            max_len_prot=args.max_len_prot, max_len_pep=args.max_len_pep,
            epochs=args.epochs, patience=args.patience,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            test_batch_size=args.test_batch_size,
            lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            freeze_first_n_epochs=args.freeze_first_n_epochs,
            fixed_threshold=args.fixed_threshold,
            selection_metric=args.selection_metric,
            n_splits=args.n_splits,
            seed=args.seed,
            num_workers=args.num_workers,
        )

    # --------------------------------------------------------
    # Standard fine-tuning (prot / pep)
    # --------------------------------------------------------
    else:
        if None in (args.train_h5, args.val_h5, args.test_h5):
            raise ValueError("--train_h5, --val_h5, and --test_h5 are all required for prot/pep fine-tuning.")
        run_finetuning(
            train_h5=args.train_h5,
            val_h5=args.val_h5,
            test_h5=args.test_h5,
            pretrained_ckpt=args.pretrained_ckpt,
            ckpt_dir=args.ckpt_dir,
            mode=args.mode,
            device=device,
            emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, heads=args.heads,
            dropout=args.dropout, num_self_layers=args.num_self_layers, ff_dim=args.ff_dim,
            max_len_prot=args.max_len_prot, max_len_pep=args.max_len_pep,
            epochs=args.epochs, patience=args.patience,
            train_batch_size=args.train_batch_size,
            val_batch_size=args.val_batch_size,
            test_batch_size=args.test_batch_size,
            lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            freeze_first_n_epochs=args.freeze_first_n_epochs,
            fixed_threshold=args.fixed_threshold,
            num_workers=args.num_workers,
        )
