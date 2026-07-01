"""
pretrain_propepx.py
===================
ProPepX Transfer Pre-Training Script
-------------------------------------
Supports all task modes:
    - prot        : protein-only binding-site pre-training
    - pep         : peptide-only binding-site pre-training
    - mode-GLOBAL : joint (protein + peptide) global binding-site pre-training

Workflow
--------
1. Build train / validation DataLoaders from HDF5 embedding files.
2. Instantiate ProPepX with mode-specific architecture parameters.
3. Train with AdamW + cosine-warmup LR schedule for a fixed number of epochs.
4. Save:
      checkpoint_<mode>.pt        – resume checkpoint (overwritten every epoch)
      best_pretrain_<mode>.pt     – best validation-loss checkpoint
      last_pretrain_<mode>.pt     – final-epoch checkpoint
      pretrain_history_<mode>.csv – per-epoch train / val loss

Usage
-----
    python pretrain_propepx.py \\
        --mode prot \\
        --train_h5  /path/to/train_embeddings.h5 \\
        --val_h5    /path/to/val_embeddings.h5 \\
        --ckpt_dir  /path/to/checkpoints \\
        --gpu_id    0

Authors: Syed Kumail Hussain Naqvi et al.
"""

# ============================================================
# Standard Library & Third-Party Imports
# ============================================================
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
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
from pepPI_binding_site_collate import collate_fn

# ============================================================
# Pre-Training Hyperparameters
# These are the canonical parameters used in the ProPepX paper.
# ============================================================
PRETRAIN_DEFAULTS = dict(
    # --- Architecture ---
    emb_dim=1024,          # ProtTrans-T5 embedding dimension
    hidden_dim=512,
    heads=8,
    dropout=0.35,
    num_self_layers=2,
    ff_dim=512,
    max_len_prot=1418,
    max_len_pep=50,

    # --- Optimisation ---
    epochs=200,
    train_batch_size=64,
    val_batch_size=64,
    lr=1e-5,
    weight_decay=1e-4,
    warmup_ratio=0.01,     # fraction of total steps used for linear warm-up

    # --- Misc ---
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


def build_pretrain_loaders(
    train_h5: str,
    val_h5: str,
    mode: str,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple:
    """
    Construct train and validation DataLoaders from HDF5 embedding files.

    Parameters
    ----------
    train_h5 : str
        Path to HDF5 file containing training embeddings and labels.
    val_h5 : str
        Path to HDF5 file containing validation embeddings and labels.
    mode : str
        Task mode – one of ``'prot'``, ``'pep'``, ``'mode-GLOBAL'``.
    train_batch_size, val_batch_size : int
    num_workers : int
    pin_memory : bool

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    train_dataset = H5PairDataset(train_h5, mode=mode)
    val_dataset   = H5PairDataset(val_h5,   mode=mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print("\n" + "=" * 50)
    print(f"  Pre-Training Dataset Summary  [mode = {mode}]")
    print("=" * 50)
    print(f"  TRAIN : {len(train_dataset):>8,} samples  |  batch = {train_batch_size}")
    print(f"  VAL   : {len(val_dataset):>8,} samples  |  batch = {val_batch_size}")
    print("=" * 50 + "\n")

    return train_loader, val_loader


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    train_loss_history: list,
    val_loss_history: list,
) -> None:
    """Serialise a full training state to disk."""
    torch.save(
        {
            "epoch":              epoch,
            "best_loss":          best_loss,
            "model_state":        model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "scheduler_state":    scheduler.state_dict() if scheduler is not None else None,
            "train_loss_history": train_loss_history,
            "val_loss_history":   val_loss_history,
        },
        path,
    )
    print(f"  ✔  Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple:
    """
    Resume training from a checkpoint if it exists.

    Returns
    -------
    start_epoch : int
    best_loss   : float
    train_history, val_history : list[float]
    """
    if not os.path.exists(path):
        print("  ℹ  No checkpoint found – starting fresh pre-training.\n")
        return 1, float("inf"), [], []

    print(f"  ↻  Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch        = ckpt["epoch"] + 1
    best_loss          = ckpt["best_loss"]
    train_loss_history = ckpt.get("train_loss_history", [])
    val_loss_history   = ckpt.get("val_loss_history",   [])

    print(f"  ↳  Resumed at epoch {start_epoch}  |  best_val_loss = {best_loss:.6f}\n")
    return start_epoch, best_loss, train_loss_history, val_loss_history


# ============================================================
# Core Training / Evaluation Loops
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mode: str,
) -> float:
    """
    Run one full training epoch.

    Performs forward pass, composite loss computation, gradient clipping
    (max-norm = 1.0), AdamW update, and cosine-warmup LR step.

    Returns
    -------
    float : mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="  Pre-Train", leave=False):
        prot_emb   = prot_emb.to(device,   non_blocking=True)
        pep_emb    = pep_emb.to(device,    non_blocking=True)
        prot_label = prot_label.to(device, non_blocking=True)
        pep_label  = pep_label.to(device,  non_blocking=True)

        # Padding masks: True where the embedding row is all-zero (padded position)
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
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> float:
    """
    Run one full validation epoch without gradient computation.

    Returns
    -------
    float : mean validation loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="  Validate ", leave=False):
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

    return total_loss / len(loader)


# ============================================================
# Main Pre-Training Orchestrator
# ============================================================

def run_pretraining(
    train_h5: str,
    val_h5: str,
    ckpt_dir: str,
    mode: str = "prot",
    device: torch.device = None,
    # --- Architecture ---
    emb_dim: int        = PRETRAIN_DEFAULTS["emb_dim"],
    hidden_dim: int     = PRETRAIN_DEFAULTS["hidden_dim"],
    heads: int          = PRETRAIN_DEFAULTS["heads"],
    dropout: float      = PRETRAIN_DEFAULTS["dropout"],
    num_self_layers: int= PRETRAIN_DEFAULTS["num_self_layers"],
    ff_dim: int         = PRETRAIN_DEFAULTS["ff_dim"],
    max_len_prot: int   = PRETRAIN_DEFAULTS["max_len_prot"],
    max_len_pep: int    = PRETRAIN_DEFAULTS["max_len_pep"],
    # --- Optimisation ---
    epochs: int             = PRETRAIN_DEFAULTS["epochs"],
    train_batch_size: int   = PRETRAIN_DEFAULTS["train_batch_size"],
    val_batch_size: int     = PRETRAIN_DEFAULTS["val_batch_size"],
    lr: float               = PRETRAIN_DEFAULTS["lr"],
    weight_decay: float     = PRETRAIN_DEFAULTS["weight_decay"],
    warmup_ratio: float     = PRETRAIN_DEFAULTS["warmup_ratio"],
    # --- Misc ---
    num_workers: int    = PRETRAIN_DEFAULTS["num_workers"],
    pin_memory: bool    = PRETRAIN_DEFAULTS["pin_memory"],
) -> tuple:
    """
    Full pre-training pipeline for ProPepX.

    Supports ``mode in {'prot', 'pep', 'mode-GLOBAL'}``.

    Returns
    -------
    model            : trained ProPepX instance (best checkpoint loaded)
    best_model_path  : str
    last_model_path  : str
    train_loss_history : list[float]
    val_loss_history   : list[float]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(ckpt_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 1. DataLoaders
    # ----------------------------------------------------------
    train_loader, val_loader = build_pretrain_loaders(
        train_h5=train_h5,
        val_h5=val_h5,
        mode=mode,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ----------------------------------------------------------
    # 2. Model
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

    print_model_parameters(model)

    # ----------------------------------------------------------
    # 3. Optimiser & Scheduler
    # ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )

    # ----------------------------------------------------------
    # 4. Checkpoint paths
    # ----------------------------------------------------------
    resume_ckpt_path  = os.path.join(ckpt_dir, f"checkpoint_{mode}.pt")
    best_model_path   = os.path.join(ckpt_dir, f"best_pretrain_{mode}.pt")
    last_model_path   = os.path.join(ckpt_dir, f"last_pretrain_{mode}.pt")
    history_csv_path  = os.path.join(ckpt_dir, f"pretrain_history_{mode}.csv")

    # ----------------------------------------------------------
    # 5. Resume if a checkpoint exists
    # ----------------------------------------------------------
    start_epoch, best_val_loss, train_loss_history, val_loss_history = load_checkpoint(
        path=resume_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # ----------------------------------------------------------
    # 6. Training Loop
    # ----------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*70}")
        print(f"  EPOCH {epoch:>4d} / {epochs}   |   MODE = {mode}")
        print(f"{'='*70}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            mode=mode,
        )

        val_loss = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mode=mode,
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"  Train Loss : {train_loss:.6f}")
        print(f"  Val   Loss : {val_loss:.6f}")

        # -- Resume checkpoint (overwritten every epoch) ----------
        save_checkpoint(
            path=resume_ckpt_path,
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_loss=best_val_loss,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
        )

        # -- Best model -------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=best_model_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_loss=best_val_loss,
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
            )
            print(f"  🏆  New best checkpoint  |  epoch = {epoch}  |  val_loss = {best_val_loss:.6f}")

    # ----------------------------------------------------------
    # 7. Save last-epoch model
    # ----------------------------------------------------------
    save_checkpoint(
        path=last_model_path,
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=epochs, best_loss=best_val_loss,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
    )

    # ----------------------------------------------------------
    # 8. Save loss history as CSV
    # ----------------------------------------------------------
    pd.DataFrame(
        {
            "epoch":      np.arange(1, len(train_loss_history) + 1),
            "train_loss": train_loss_history,
            "val_loss":   val_loss_history,
        }
    ).to_csv(history_csv_path, index=False)

    print("\n" + "=" * 70)
    print("  ✅  Pre-training completed.")
    print(f"  🏆  Best model  → {best_model_path}")
    print(f"  💾  Last model  → {last_model_path}")
    print(f"  📄  Loss CSV    → {history_csv_path}")
    print("=" * 70 + "\n")

    # Load best weights before returning
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    return model, best_model_path, last_model_path, train_loss_history, val_loss_history


# ============================================================
# CLI Entry Point
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProPepX Transfer Pre-Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument("--train_h5",  required=True,  help="Path to training HDF5 embedding file.")
    parser.add_argument("--val_h5",    required=True,  help="Path to validation HDF5 embedding file.")
    parser.add_argument("--ckpt_dir",  required=True,  help="Directory to save checkpoints and logs.")

    # Mode
    parser.add_argument(
        "--mode",
        default="prot",
        choices=["prot", "pep", "mode-GLOBAL"],
        help="Task mode for pre-training.",
    )

    # Hardware
    parser.add_argument("--gpu_id", default="0", help="CUDA device ID (e.g. '0' or '1').")

    # Architecture (paper defaults)
    parser.add_argument("--emb_dim",         type=int,   default=PRETRAIN_DEFAULTS["emb_dim"])
    parser.add_argument("--hidden_dim",      type=int,   default=PRETRAIN_DEFAULTS["hidden_dim"])
    parser.add_argument("--heads",           type=int,   default=PRETRAIN_DEFAULTS["heads"])
    parser.add_argument("--dropout",         type=float, default=PRETRAIN_DEFAULTS["dropout"])
    parser.add_argument("--num_self_layers", type=int,   default=PRETRAIN_DEFAULTS["num_self_layers"])
    parser.add_argument("--ff_dim",          type=int,   default=PRETRAIN_DEFAULTS["ff_dim"])
    parser.add_argument("--max_len_prot",    type=int,   default=PRETRAIN_DEFAULTS["max_len_prot"])
    parser.add_argument("--max_len_pep",     type=int,   default=PRETRAIN_DEFAULTS["max_len_pep"])

    # Optimisation (paper defaults)
    parser.add_argument("--epochs",            type=int,   default=PRETRAIN_DEFAULTS["epochs"])
    parser.add_argument("--train_batch_size",  type=int,   default=PRETRAIN_DEFAULTS["train_batch_size"])
    parser.add_argument("--val_batch_size",    type=int,   default=PRETRAIN_DEFAULTS["val_batch_size"])
    parser.add_argument("--lr",                type=float, default=PRETRAIN_DEFAULTS["lr"])
    parser.add_argument("--weight_decay",      type=float, default=PRETRAIN_DEFAULTS["weight_decay"])
    parser.add_argument("--warmup_ratio",      type=float, default=PRETRAIN_DEFAULTS["warmup_ratio"])

    # Misc
    parser.add_argument("--seed",        type=int, default=PRETRAIN_DEFAULTS["seed"])
    parser.add_argument("--num_workers", type=int, default=PRETRAIN_DEFAULTS["num_workers"])

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  CUDA devices available : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    print(f"  Active device          : {device}  (GPU {args.gpu_id})\n")

    set_seed(args.seed)

    run_pretraining(
        train_h5=args.train_h5,
        val_h5=args.val_h5,
        ckpt_dir=args.ckpt_dir,
        mode=args.mode,
        device=device,
        # Architecture
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        num_self_layers=args.num_self_layers,
        ff_dim=args.ff_dim,
        max_len_prot=args.max_len_prot,
        max_len_pep=args.max_len_pep,
        # Optimisation
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        # Misc
        num_workers=args.num_workers,
    )
