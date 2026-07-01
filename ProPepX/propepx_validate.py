"""
propepx_validate.py
====================
ProPepX Results Validation
----------------------------------------------------------------------
Reproduces all reported test-set metrics across every ProPepX mode,
dataset, and embedding model by:

  1. Downloading fine-tuned weights from HuggingFace
     (skipped automatically if already cached locally in ~/.cache/propepx/weights/)
  2. Downloading test-set HDF5 embeddings from HuggingFace
     (always fetched fresh; stored in ~/.cache/propepx/embeddings/)
  3. Running inference and computing metrics
  4. Saving per-run CSVs + a consolidated results table
  5. Generating a LaTeX table ready for the paper

Supported validation runs
--------------------------
  MODE        DATASET          EMBEDDING
  ----------  ---------------  ----------
  prot        ts092            prottrans / esm
  prot        ts125            prottrans / esm
  prot        ts251            prottrans / esm
  prot        ts639            prottrans / esm
  pep         camp_test231     prottrans / esm
  mode-GLOBAL leads_ts251      prottrans / esm
  mode-GLOBAL test167          prottrans / esm
  zero-shot   test167_zs       prottrans / esm

Usage
-----
  # Validate ALL runs (full paper results)
  python propepx_validate.py --gpu_id 0

  # Validate a specific mode/dataset/embedding only
  python propepx_validate.py --mode prot --dataset ts092 --embedding esm --gpu_id 0

  # List all available validation runs and exit
  python propepx_validate.py --list

  # Force re-download of weights even if cached
  python propepx_validate.py --force_weight_download

Authors: Syed Kumail Hussain Naqvi et al.
"""

# ─────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────
import argparse
import os
import sys
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# ProPepX package
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/kumail/Transformer Model/ProPepX_complete_code/")

from CoBindingCNN            import ProPepX
from Dataset_Preprocessing   import H5PairDataset
from model_load_weight_utils import load_weight
from pepPI_binding_site_collate import collate_fn
from propepx_metrics         import (
    compute_prot_metrics,
    compute_pep_pair_metrics,
    summarize_pep_pair_metrics,
    compute_joint_mode_metrics,
)
from propepx_config          import (
    ARCH, EMB_DIM, BINDING_THRESHOLD,
    FINETUNE_WEIGHT_REGISTRY,
    PRETRAIN_WEIGHT_REGISTRY,
    TEST_EMBEDDING_REGISTRY,
    DATASET_DESCRIPTIONS,
    MODE_DESCRIPTIONS,
    EMBEDDING_DESCRIPTIONS,
    resolve_weight,
    resolve_embedding,
    HF_WEIGHTS_CACHE,
)


# ═══════════════════════════════════════════════════════════════
# Validation run catalogue
# ═══════════════════════════════════════════════════════════════
#
# Each entry is a dict: {mode, dataset, embedding, zero_shot}
# This drives all loops in run_all_validations().
#
ALL_RUNS = [
    # ── Protein-side ─────────────────────────────────────────
    dict(mode="prot", dataset="ts092",       embedding="prottrans", zero_shot=False),
    dict(mode="prot", dataset="ts092",       embedding="esm",       zero_shot=False),
    dict(mode="prot", dataset="ts125",       embedding="prottrans", zero_shot=False),
    dict(mode="prot", dataset="ts125",       embedding="esm",       zero_shot=False),
    dict(mode="prot", dataset="ts251",       embedding="prottrans", zero_shot=False),
    dict(mode="prot", dataset="ts251",       embedding="esm",       zero_shot=False),
    dict(mode="prot", dataset="ts639",       embedding="prottrans", zero_shot=False),
    dict(mode="prot", dataset="ts639",       embedding="esm",       zero_shot=False),
    # ── Peptide-side ─────────────────────────────────────────
    dict(mode="pep",  dataset="camp_test231",embedding="prottrans", zero_shot=False),
    dict(mode="pep",  dataset="camp_test231",embedding="esm",       zero_shot=False),
    # ── Joint (mode-GLOBAL) ───────────────────────────────────
    dict(mode="mode-GLOBAL", dataset="leads_ts251", embedding="prottrans", zero_shot=False),
    dict(mode="mode-GLOBAL", dataset="leads_ts251", embedding="esm",       zero_shot=False),
    dict(mode="mode-GLOBAL", dataset="test167",     embedding="prottrans", zero_shot=False),
    dict(mode="mode-GLOBAL", dataset="test167",     embedding="esm",       zero_shot=False),
    # ── Zero-shot ─────────────────────────────────────────────
    dict(mode="mode-GLOBAL", dataset="test167_zs", embedding="prottrans", zero_shot=True),
    dict(mode="mode-GLOBAL", dataset="test167_zs", embedding="esm",       zero_shot=True),
]


# ═══════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════

def _print_banner(text: str, width: int = 72) -> None:
    bar = "═" * width
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}\n")


def _build_model(
    mode: str,
    emb_dim: int,
    device: torch.device,
    zero_shot: bool = False,
) -> ProPepX:

    arch_mode = "both" if "global" in mode.lower() else mode

    max_len_prot = 8000 if zero_shot else ARCH["max_len_prot"]

    return ProPepX(
        emb_dim=emb_dim,
        hidden_dim=ARCH["hidden_dim"],
        heads=ARCH["heads"],
        dropout=ARCH["dropout"],
        mode=arch_mode,
        num_self_layers=ARCH["num_self_layers"],
        ff_dim=ARCH["ff_dim"],
        max_len_prot=max_len_prot,
        max_len_pep=ARCH["max_len_pep"],
    ).to(device)


def _get_weight_path(
    mode: str,
    dataset: str,
    embedding: str,
    zero_shot: bool,
    force_download: bool,
) -> str:
    """
    Resolve the local path to the fine-tuned (or pre-trained) weight file.

    If the weight is already cached and force_download is False, the download
    is silently skipped.  When force_download is True, huggingface_hub still
    uses its own etag-based mechanism and will only re-download if the remote
    file has changed.
    """
    if zero_shot:
        registry = PRETRAIN_WEIGHT_REGISTRY
        key_mode = "zero-shot"
        hf_path  = registry[key_mode][embedding]
    else:
        registry = FINETUNE_WEIGHT_REGISTRY
        key_mode = mode
        try:
            hf_path = registry[key_mode][dataset][embedding]
        except KeyError:
            raise ValueError(
                f"No weight registered for mode='{mode}', "
                f"dataset='{dataset}', embedding='{embedding}'."
            )

    return resolve_weight(hf_path, verbose=True)


def _get_embedding_path(mode: str, dataset: str, embedding: str, zero_shot: bool = False) -> str:
    """
    Resolve test HDF5 embedding path.

    User/API may call zero-shot, but internally zero-shot still uses
    mode-GLOBAL architecture. For registry lookup, test167_zs is stored
    under TEST_EMBEDDING_REGISTRY["zero-shot"].
    """
    if zero_shot or dataset == "test167_zs":
        registry_key = "zero-shot"
    else:
        registry_key = mode if mode in TEST_EMBEDDING_REGISTRY else "mode-GLOBAL"

    try:
        hf_path = TEST_EMBEDDING_REGISTRY[registry_key][dataset][embedding]
    except KeyError:
        raise ValueError(
            f"No embedding registered for registry='{registry_key}', "
            f"mode='{mode}', dataset='{dataset}', embedding='{embedding}'."
        )

    return resolve_embedding(hf_path, verbose=True)


def _make_dataloader(
    h5_path: str,
    mode: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    ds_mode = "both" if "global" in mode.lower() else mode
    dataset = H5PairDataset(h5_path, mode=ds_mode)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )


# ═══════════════════════════════════════════════════════════════
# Evaluation loops (one per ProPepX mode)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_prot(
    model: ProPepX,
    loader: DataLoader,
    device: torch.device,
    threshold: float = BINDING_THRESHOLD,
) -> dict:
    """
    Protein-mode evaluation.
    Concatenates all valid (non-padded) residue labels and probabilities
    globally before computing metrics to match the paper's evaluation protocol.
    """
    model.eval()
    all_true, all_prob = [], []

    for prot_emb, pep_emb, prot_label, _ in tqdm(loader, desc="  Eval prot", leave=False):
        prot_emb   = prot_emb.to(device)
        pep_emb    = pep_emb.to(device)
        prot_label = prot_label.to(device)

        prot_mask = prot_emb.abs().sum(-1) == 0
        pep_mask  = pep_emb.abs().sum(-1)  == 0

        prot_logits, _ = model(
            prot_emb=prot_emb, pep_emb=pep_emb,
            prot_mask=prot_mask, pep_mask=pep_mask,
        )

        prob  = torch.softmax(prot_logits, dim=-1)[..., 1]
        valid = prot_label != -100

        y_t = prot_label[valid].cpu().numpy()
        y_p = prob[valid].cpu().numpy()

        if len(y_t) > 0:
            all_true.append(y_t)
            all_prob.append(y_p)

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    metrics = compute_prot_metrics(y_true, y_prob, threshold=threshold)
    metrics["Num_Residues"] = len(y_true)
    metrics["Num_Positive"] = int(y_true.sum())
    metrics["Num_Negative"] = int(len(y_true) - y_true.sum())
    return metrics


@torch.no_grad()
def _eval_pep(
    model: ProPepX,
    loader: DataLoader,
    device: torch.device,
    threshold: float = BINDING_THRESHOLD,
) -> tuple:
    """
    Peptide-mode evaluation.
    Computes per-peptide metrics then returns the mean summary dict
    and the full per-sample DataFrame (as in the notebook).
    """
    model.eval()
    per_sample = []
    sample_idx = 0

    for prot_emb, pep_emb, _, pep_label in tqdm(loader, desc="  Eval pep", leave=False):
        prot_emb  = prot_emb.to(device)
        pep_emb   = pep_emb.to(device)
        pep_label = pep_label.to(device)

        prot_mask = prot_emb.abs().sum(-1) == 0
        pep_mask  = pep_emb.abs().sum(-1)  == 0

        _, pep_logits = model(
            prot_emb=prot_emb, pep_emb=pep_emb,
            prot_mask=prot_mask, pep_mask=pep_mask,
        )

        pep_prob = torch.softmax(pep_logits, dim=-1)[..., 1]
        pep_pred = (pep_prob >= threshold).long()

        for i in range(pep_label.size(0)):
            true_i  = pep_label[i]
            pred_i  = pep_pred[i]
            valid_i = true_i != -100

            y_true = true_i[valid_i].cpu().numpy()
            y_pred = pred_i[valid_i].cpu().numpy()

            if len(y_true) == 0:
                continue

            mn = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:mn], y_pred[:mn]

            m = compute_pep_pair_metrics(y_true, y_pred)
            per_sample.append({
                "sample_index":   sample_idx + 1,
                "peptide_length": len(y_true),
                "num_positive":   int(np.sum(y_true)),
                "num_negative":   int(len(y_true) - np.sum(y_true)),
                **m,
            })
            sample_idx += 1

    summary, per_sample_df = summarize_pep_pair_metrics(per_sample)
    return summary, per_sample_df


@torch.no_grad()
def _eval_joint(
    model: ProPepX,
    loader: DataLoader,
    device: torch.device,
    threshold: float = BINDING_THRESHOLD,
) -> dict:
    """
    Joint (mode-GLOBAL / zero-shot) evaluation.
    Computes separate protein and peptide global residue-level metrics,
    then reports mean MCC, mean AUROC, and mean AUPR across both chains.
    """
    model.eval()
    all_prot_true, all_prot_prob = [], []
    all_pep_true,  all_pep_prob  = [], []

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="  Eval joint", leave=False):
        prot_emb   = prot_emb.to(device)
        pep_emb    = pep_emb.to(device)
        prot_label = prot_label.to(device)
        pep_label  = pep_label.to(device)

        prot_mask = prot_emb.abs().sum(-1) == 0
        pep_mask  = pep_emb.abs().sum(-1)  == 0

        prot_logits, pep_logits = model(
            prot_emb=prot_emb, pep_emb=pep_emb,
            prot_mask=prot_mask, pep_mask=pep_mask,
        )

        all_prot_true.append(prot_label.cpu().numpy().reshape(-1))
        all_prot_prob.append(torch.softmax(prot_logits, -1)[..., 1].cpu().numpy().reshape(-1))
        all_pep_true.append(pep_label.cpu().numpy().reshape(-1))
        all_pep_prob.append(torch.softmax(pep_logits,  -1)[..., 1].cpu().numpy().reshape(-1))

    prot_m = compute_joint_mode_metrics(
        np.concatenate(all_prot_true), np.concatenate(all_prot_prob), threshold
    )
    pep_m  = compute_joint_mode_metrics(
        np.concatenate(all_pep_true),  np.concatenate(all_pep_prob),  threshold
    )

    return {
        # Per-chain
        "Prot_MCC":   prot_m["MCC"],   "Prot_AUROC": prot_m["AUROC"],
        "Prot_AUPR":  prot_m["AUPR"],  "Prot_F1":    prot_m["F1"],
        "Prot_ACC":   prot_m["ACC"],
        "Prot_Pos":   prot_m["Positive"], "Prot_Neg": prot_m["Negative"],
        "Pep_MCC":    pep_m["MCC"],    "Pep_AUROC":  pep_m["AUROC"],
        "Pep_AUPR":   pep_m["AUPR"],   "Pep_F1":     pep_m["F1"],
        "Pep_ACC":    pep_m["ACC"],
        "Pep_Pos":    pep_m["Positive"],  "Pep_Neg":  pep_m["Negative"],
        # Aggregate
        "Mean_MCC":   float(np.nanmean([prot_m["MCC"],   pep_m["MCC"]])),
        "Mean_AUROC": float(np.nanmean([prot_m["AUROC"], pep_m["AUROC"]])),
        "Mean_AUPR":  float(np.nanmean([prot_m["AUPR"],  pep_m["AUPR"]])),
    }


# ═══════════════════════════════════════════════════════════════
# Single validation run
# ═══════════════════════════════════════════════════════════════

def run_one_validation(
    mode:           str,
    dataset:        str,
    embedding:      str,
    device:         torch.device,
    output_dir:     str,
    zero_shot:      bool  = False,
    batch_size:     int   = 16,
    num_workers:    int   = 4,
    threshold:      float = BINDING_THRESHOLD,
    force_download: bool  = False,
) -> dict:
    """
    Download weights + embedding, run inference, compute metrics.

    Returns
    -------
    dict with all metrics + run meta-data.
    """
    run_label = f"{mode} | {dataset} | {embedding}"
    _print_banner(f"Validating :  {run_label}")

    # ── 1. Resolve weights ───────────────────────────────────
    print("  [1/4]  Resolving model weights …")
    weight_path = _get_weight_path(mode, dataset, embedding, zero_shot, force_download)

    # ── 2. Resolve test embeddings ───────────────────────────
    print("\n  [2/4]  Resolving test embeddings …")
    h5_path = _get_embedding_path(mode, dataset, embedding, zero_shot)

    # ── 3. Build model + load weights ────────────────────────
    print("\n  [3/4]  Building model & loading checkpoint …")
    emb_dim = EMB_DIM[embedding]
    model   = _build_model(mode, emb_dim, device, zero_shot=zero_shot,)
    model   = load_weight(model, weight_path, device, strict=False, verbose=True)
    model.eval()

    # ── 4. DataLoader ────────────────────────────────────────
    loader = _make_dataloader(h5_path, mode, batch_size, num_workers)
    print(f"  Test samples : {len(loader.dataset):,}  |  batches : {len(loader)}")

    # ── 5. Evaluate ──────────────────────────────────────────
    print("\n  [4/4]  Running evaluation …")

    if mode == "prot":
        metrics = _eval_prot(model, loader, device, threshold)
        per_sample_df = None

    elif mode == "pep":
        metrics, per_sample_df = _eval_pep(model, loader, device, threshold)

    elif mode in ("mode-GLOBAL", "mode-global", "zero-shot"):
        metrics = _eval_joint(model, loader, device, threshold)
        per_sample_df = None

    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    # ── Free GPU memory ──────────────────────────────────────
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Meta-data ────────────────────────────────────────────
    run_meta = {
        "Mode":      mode,
        "Dataset":   DATASET_DESCRIPTIONS.get(dataset, dataset),
        "Embedding": EMBEDDING_DESCRIPTIONS.get(embedding, embedding),
        "Threshold": threshold,
        "Zero_shot": zero_shot,
        "Weight":    os.path.basename(weight_path),
        "H5":        os.path.basename(h5_path),
    }
    result = {**run_meta, **metrics}

    # ── Console summary ──────────────────────────────────────
    print(f"\n  ── Results : {run_label} ──")
    for k, v in metrics.items():
        tag = f"    {k:<22}"
        print(f"{tag} {v:.4f}" if isinstance(v, float) else f"{tag} {v}")

    return result


# ═══════════════════════════════════════════════════════════════
# Publication table generators
# ═══════════════════════════════════════════════════════════════

def _make_prot_latex(df: pd.DataFrame) -> str:
    """LaTeX table for protein-mode results."""
    cols = ["Dataset", "Embedding", "AUC", "AUPR", "MCC", "F1",
            "Recall", "Specificity", "Precision", "Accuracy"]
    sub  = df[df["Mode"] == "prot"].copy()
    if sub.empty:
        return ""
    sub["Dataset"]   = sub["Dataset"].str.replace("Protein-side  ", "")
    sub["Embedding"] = sub["Embedding"].str.split("(").str[0].str.strip()
    rows = []
    for _, r in sub.iterrows():
        row = " & ".join(
            r[c] if isinstance(r.get(c), str)
            else f"{float(r.get(c, 0)):.4f}" if c not in ("Dataset","Embedding")
            else str(r.get(c,""))
            for c in cols
        )
        rows.append(row + r" \\")

    header = " & ".join(cols) + r" \\"
    return (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{ProPepX Protein Binding Site Prediction Results}" + "\n"
        r"\begin{tabular}{ll" + "c" * (len(cols)-2) + "}\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )


def _make_joint_latex(df: pd.DataFrame) -> str:
    """LaTeX table for joint-mode and zero-shot results."""
    cols = ["Mode", "Dataset", "Embedding",
            "Mean_MCC", "Mean_AUROC", "Mean_AUPR",
            "Prot_MCC", "Prot_AUROC", "Pep_MCC", "Pep_AUROC"]
    sub  = df[df["Mode"].isin(["mode-GLOBAL","zero-shot"])].copy()
    if sub.empty:
        return ""
    sub["Embedding"] = sub["Embedding"].str.split("(").str[0].str.strip()
    rows = []
    for _, r in sub.iterrows():
        row = " & ".join(
            str(r.get(c,"")) if c in ("Mode","Dataset","Embedding")
            else f"{float(r.get(c, 0)):.4f}"
            for c in cols
        )
        rows.append(row + r" \\")

    header = " & ".join(cols) + r" \\"
    return (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{ProPepX Joint and Zero-Shot Prediction Results}" + "\n"
        r"\begin{tabular}{lll" + "c" * (len(cols)-3) + "}\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )


def _make_pep_latex(df: pd.DataFrame) -> str:
    """LaTeX table for peptide-mode results."""
    cols = ["Dataset", "Embedding",
            "Mean_AUC", "Mean_MCC", "Mean_F1",
            "Mean_Precision", "Mean_Recall", "Mean_Specificity", "Mean_Accuracy"]
    sub  = df[df["Mode"] == "pep"].copy()
    if sub.empty:
        return ""
    sub["Embedding"] = sub["Embedding"].str.split("(").str[0].str.strip()
    rows = []
    for _, r in sub.iterrows():
        row = " & ".join(
            str(r.get(c,"")) if c in ("Dataset","Embedding")
            else f"{float(r.get(c, 0)):.4f}"
            for c in cols
        )
        rows.append(row + r" \\")

    header = " & ".join(cols) + r" \\"
    return (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{ProPepX Peptide Binding Site Prediction Results}" + "\n"
        r"\begin{tabular}{ll" + "c" * (len(cols)-2) + "}\n"
        r"\hline" + "\n"
        + header + "\n"
        r"\hline" + "\n"
        + "\n".join(rows) + "\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )



# ═══════════════════════════════════════════════════════════════
# All-runs orchestrator
# ═══════════════════════════════════════════════════════════════

def run_all_validations(
    device:         torch.device,
    output_dir:     str,
    mode_filter:    str  = None,
    dataset_filter: str  = None,
    emb_filter:     str  = None,
    batch_size:     int  = 16,
    num_workers:    int  = 4,
    threshold:      float = BINDING_THRESHOLD,
    force_download: bool  = False,
) -> list:
    """
    Run all (or a filtered subset of) ProPepX validation experiments.

    Parameters
    ----------
    device         : torch.device
    output_dir     : str    Root directory for all output CSVs and LaTeX files.
    mode_filter    : str    If set, only runs whose mode equals this string.
    dataset_filter : str    If set, only runs whose dataset equals this string.
    emb_filter     : str    If set, only runs whose embedding equals this string.
    batch_size     : int    DataLoader batch size.
    num_workers    : int    DataLoader workers.
    threshold      : float  Binding probability threshold (default 0.50).
    force_download : bool   Force re-download weights even if cached.

    Returns
    -------
    list[dict] : all metric dictionaries.
    """
    internal_mode_filter = mode_filter

    if mode_filter == "zero-shot":
        internal_mode_filter = "mode-GLOBAL"
        dataset_filter = dataset_filter or "test167_zs"

    runs = [
        r for r in ALL_RUNS
        if (internal_mode_filter is None or r["mode"] == internal_mode_filter)
        and (dataset_filter is None or r["dataset"] == dataset_filter)
        and (emb_filter is None or r["embedding"] == emb_filter)
    ]

    if not runs:
        print("\n  No validation runs match the given filters.")
        return []

    _print_banner(
        f"ProPepX Validation  ·  {len(runs)} run(s)  "
        f"·  device = {device}  "
        f"·  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    all_results = []
    failed      = []

    for idx, run in enumerate(runs, 1):
        print(f"\n  RUN {idx}/{len(runs)}")
        try:
            result = run_one_validation(
                mode           = run["mode"],
                dataset        = run["dataset"],
                embedding      = run["embedding"],
                device         = device,
                output_dir     = output_dir,
                zero_shot      = run.get("zero_shot", False),
                batch_size     = batch_size,
                num_workers    = num_workers,
                threshold      = threshold,
                force_download = force_download,
            )
            all_results.append(result)
        except Exception as exc:
            label = f"{run['mode']}|{run['dataset']}|{run['embedding']}"
            print(f"\n  ERROR in run [{label}]: {exc}")
            failed.append({**run, "error": str(exc)})

    # ── Failed runs report ────────────────────────────────────
    if failed:
        print(f"\n  FAILED RUNS ({len(failed)}):")
        for f in failed:
            print(f"    {f['mode']} | {f['dataset']} | {f['embedding']}  ->  {f['error']}")
        fail_csv = os.path.join(output_dir, "failed_runs.csv")
        pd.DataFrame(failed).to_csv(fail_csv, index=False)

    _print_banner(
        f"Validation complete  ·  "
        f"{len(all_results)} succeeded  ·  {len(failed)} failed"
    )

    return all_results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _list_runs() -> None:
    print("\n  Available ProPepX validation runs:")
    print(f"  {'#':<4} {'Mode':<14} {'Dataset':<16} {'Embedding':<12} {'Zero-shot'}")
    print("  " + "─" * 62)
    for i, r in enumerate(ALL_RUNS, 1):
        print(
            f"  {i:<4} {r['mode']:<14} {r['dataset']:<16} "
            f"{r['embedding']:<12} {'yes' if r.get('zero_shot') else 'no'}"
        )
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProPepX Results Validation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", default=None,
        choices=["prot", "pep", "mode-GLOBAL", "zero-shot"],
        help="Validate only this mode (default: all modes).",
    )
    parser.add_argument(
        "--dataset", default=None,
        help=(
            "Validate only this dataset key. Available:\n"
            "  prot        : ts092, ts125, ts251, ts639\n"
            "  pep         : camp_test231\n"
            "  mode-GLOBAL : leads_ts251, test167\n"
            "  zero-shot   : test167_zs\n"
            "(default: all datasets)"
        ),
    )
    parser.add_argument(
        "--embedding", default=None,
        choices=["prottrans", "esm"],
        help="Validate only this embedding model (default: both).",
    )
    parser.add_argument(
        "--output_dir", default="./propepx_validation_results",
        help="Directory for output CSVs and LaTeX tables.",
    )
    parser.add_argument(
        "--gpu_id", default="0",
        help="CUDA device ID (default: 0).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="DataLoader batch size (default: 16).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--threshold", type=float, default=BINDING_THRESHOLD,
        help=f"Binding probability threshold (default: {BINDING_THRESHOLD}).",
    )
    parser.add_argument(
        "--force_weight_download", action="store_true",
        help=(
            "Force re-download of model weights even if already cached.\n"
            "huggingface_hub will still use etag-based deduplication."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available validation runs and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list:
        _list_runs()
        sys.exit(0)

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"\n  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Device: {device}\n")

    run_all_validations(
        device         = device,
        output_dir     = args.output_dir,
        mode_filter    = args.mode,
        dataset_filter = args.dataset,
        emb_filter     = args.embedding,
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        threshold      = args.threshold,
        force_download = args.force_weight_download,
    )
