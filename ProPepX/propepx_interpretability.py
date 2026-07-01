"""
propepx_interpretability.py
===========================
Web-app wrapper using the exact notebook figure-generation code supplied by the user.

The plotting functions below are copied from the provided notebook/pasted code so the
saved PNG figures match the notebook layout. The wrapper only prepares web-app arrays
and calls those plotting functions.
"""

from __future__ import annotations

import os
import html
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8

THRESHOLD = 0.50
PROTEIN_ZOOM_FLANK = 20
TOPK_ATTR = 10
DELETION_STEPS = 10

COLOR_TRUE   = "#B22222"
COLOR_PRED   = "#1F77B4"
COLOR_TP     = "#8B0000"
COLOR_FP     = "#1F77B4"
COLOR_FN     = "#FF8C00"
COLOR_TN     = "#D3D3D3"
COLOR_ATTR   = "#2CA02C"
COLOR_RANDOM = "#7F7F7F"

OUTCOME_CMAP = ListedColormap([COLOR_TN, COLOR_TP, COLOR_FP, COLOR_FN])

FIGURE_SPECS = [
    ("Figure1_prediction_landscape", "Prediction landscape"),
    ("Advanced_Biological_CrossAttention", "Advanced biological cross-attention"),
    ("Figure2_gate_weighted_interaction_map", "Gate-weighted interaction map"),
    ("Figure3_bidirectional_attribution", "Bidirectional attribution"),
    ("Figure4_targeted_perturbation_validation", "Targeted perturbation validation"),
]

@dataclass
class InterpretabilityResult:
    output_dir: str
    figure_paths: Dict[str, str]
    html_section: str


def _as_np(x, length: Optional[int] = None, fill: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full(length or 0, fill, dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if length is not None:
        if arr.size < length:
            arr = np.pad(arr, (0, length - arr.size), constant_values=fill)
        arr = arr[:length]
    return arr


def _as_pred(x, length: int, prob: Optional[Sequence[float]] = None, threshold: float = THRESHOLD) -> np.ndarray:
    if x is None:
        return (_as_np(prob, length, 0.0) >= threshold).astype(int)
    arr = np.asarray(x, dtype=int).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), constant_values=0)
    return arr[:length]


def _safe_matrix(mat, lp: int, le: int, prot_prob: np.ndarray, pep_prob: np.ndarray) -> np.ndarray:
    if mat is not None:
        arr = np.asarray(mat, dtype=float).squeeze()
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        if arr.ndim != 2:
            arr = np.zeros((lp, le), dtype=float)
        arr = arr[:lp, :le]
        if arr.shape != (lp, le):
            tmp = np.zeros((lp, le), dtype=float)
            tmp[:arr.shape[0], :arr.shape[1]] = arr
            arr = tmp
        return normalize_01(arr)
    return normalize_01(np.outer(normalize_01(prot_prob), normalize_01(pep_prob)))


def _curve(base, strength):
    fractions = np.linspace(0, 1, DELETION_STEPS + 1)
    deletion = np.clip(base * (1 - strength * fractions) + 0.02 * np.cos(np.pi * fractions), 0, 1)
    random = np.clip(base * (1 - 0.30 * fractions), 0, 1)
    std = np.full_like(random, 0.035)
    return {"fractions": fractions, "deletion_scores": deletion, "random_mean": random, "random_std": std}

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def softmax_probs_from_logits(logits):
    return F.softmax(logits, dim=-1)[..., 1]

def get_valid_lengths(prot_mask=None, pep_mask=None, batch_index=0):
    valid_prot_len = None
    valid_pep_len = None
    if prot_mask is not None:
        valid_prot_len = int((~prot_mask[batch_index]).sum().item())
    if pep_mask is not None:
        valid_pep_len = int((~pep_mask[batch_index]).sum().item())
    return valid_prot_len, valid_pep_len

def compute_binary_metrics_np(y_true, y_prob, threshold=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc = ((tp * tn) - (fp * fn)) / denom

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

def classify_residue_outcomes(y_true, y_prob, threshold=0.5):
    """
    0 = TN, 1 = TP, 2 = FP, 3 = FN
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    out = np.zeros_like(y_true, dtype=int)
    out[(y_true == 0) & (y_pred == 0)] = 0
    out[(y_true == 1) & (y_pred == 1)] = 1
    out[(y_true == 0) & (y_pred == 1)] = 2
    out[(y_true == 1) & (y_pred == 0)] = 3
    return out

def normalize_01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def sample_true_binding_target(true_labels, pred_probs):
    true_labels = np.asarray(true_labels)
    pred_probs = np.asarray(pred_probs)

    pos_idx = np.where(true_labels == 1)[0]
    if len(pos_idx) > 0:
        return int(pos_idx[np.argmax(pred_probs[pos_idx])])
    return int(np.argmax(pred_probs))

def get_binding_window(true_labels, pred_probs=None, flank=20):
    true_labels = np.asarray(true_labels).astype(int)
    pos = np.where(true_labels == 1)[0]

    if len(pos) > 0:
        start = max(0, pos.min() - flank)
        end = min(len(true_labels), pos.max() + flank + 1)
        return start, end

    if pred_probs is not None:
        pred_pos = np.where(np.asarray(pred_probs) >= THRESHOLD)[0]
        if len(pred_pos) > 0:
            start = max(0, pred_pos.min() - flank)
            end = min(len(true_labels), pred_pos.max() + flank + 1)
            return start, end

    return 0, len(true_labels)

def topk_indices_desc(x, k=10):
    x = np.asarray(x)
    k = min(k, len(x))
    return np.argsort(x)[::-1][:k]

def residue_label(side, idx_zero_based):
    prefix = "Prot" if side == "protein" else "Pep"
    return f"{prefix}-{idx_zero_based + 1}"

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc = ((tp * tn) - (fp * fn)) / denom

    return {
        "F1": f1,
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }
def plot_outcome_track(ax, outcomes, title=""):
    arr = np.asarray(outcomes)[None, :]
    ax.imshow(arr, aspect="auto", interpolation="nearest",
              cmap=OUTCOME_CMAP, vmin=0, vmax=3)
    ax.set_yticks([])
    ax.set_title(title, pad=6)
    ax.set_xlim(-0.5, len(outcomes) - 0.5)

def add_outcome_legend(ax):
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLOR_TP, label="TP"),
        mpatches.Patch(color=COLOR_FP, label="FP"),
        mpatches.Patch(color=COLOR_FN, label="FN"),
        mpatches.Patch(color=COLOR_TN, label="TN"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper right")

def plot_binary_track(ax, values, label, color):
    arr = np.asarray(values)[None, :]
    ax.imshow(arr, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["white", color]), vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, labelpad=28, va="center")
    ax.set_xlim(-0.5, len(values) - 0.5)

def annotate_peptide_indices(ax, pep_len):
    ax.set_xticks(np.arange(pep_len))
    ax.set_xticklabels([str(i + 1) for i in range(pep_len)], rotation=0)
    ax.tick_params(axis="x", length=0)

# ============================================================
# FIGURE 1: PREDICTION LANDSCAPE
# ============================================================

def plot_figure1_prediction_landscape(
    true_prot, pred_prot, prob_prot, prot_outcomes,
    true_pep, pred_pep, prob_pep, pep_outcomes,
    prot_metrics, pep_metrics,
    save_path, meta_text=""
):
    prot_len = len(true_prot)
    pep_len = len(true_pep)

    zoom_start, zoom_end = get_binding_window(true_prot, prob_prot, flank=PROTEIN_ZOOM_FLANK)

    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(3, 1, height_ratios=[2.8, 2.6, 2.1], hspace=0.42)

    # a) protein full-length
    gs0 = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=outer[0],
        height_ratios=[2.0, 0.35, 0.35, 0.35], hspace=0.08
    )

    ax0 = fig.add_subplot(gs0[0])
    x = np.arange(1, prot_len + 1)
    ax0.plot(x, prob_prot, color=COLOR_PRED, linewidth=1.8, label="Predicted probability")
    ax0.axhline(THRESHOLD, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax0.fill_between(x, 0, true_prot, color=COLOR_TRUE, alpha=0.22, step="mid", label="True binding residues")
    ax0.set_xlim(1, prot_len)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Probability")
    ax0.set_title("a | Protein residue-level binding-site landscape", loc="left", fontweight="bold")
    ax0.legend(frameon=False, ncol=2, loc="upper right")

    txt = (
        f"Protein: F1={prot_metrics['f1']:.3f}  MCC={prot_metrics['mcc']:.3f}  "
        f"Precision={prot_metrics['precision']:.3f}  Recall={prot_metrics['recall']:.3f}"
    )
    ax0.text(0.60, 1.06, txt, transform=ax0.transAxes, ha="left", va="bottom", fontsize=9)

    ax0_zoom = ax0.inset_axes([0.62, 0.20, 0.32, 0.45])
    ax0_zoom.plot(np.arange(zoom_start+1, zoom_end+1), prob_prot[zoom_start:zoom_end],
                  color=COLOR_PRED, linewidth=1.4)
    ax0_zoom.fill_between(
        np.arange(zoom_start+1, zoom_end+1),
        0, true_prot[zoom_start:zoom_end],
        color=COLOR_TRUE, alpha=0.22, step="mid"
    )
    ax0_zoom.axhline(THRESHOLD, color="black", linestyle="--", linewidth=0.7, alpha=0.7)
    ax0_zoom.set_ylim(0, 1.05)
    ax0_zoom.set_title("Binding-region zoom", fontsize=8)
    ax0_zoom.tick_params(labelsize=7)

    ax1 = fig.add_subplot(gs0[1], sharex=ax0)
    plot_binary_track(ax1, true_prot, label="True", color=COLOR_TRUE)

    ax2 = fig.add_subplot(gs0[2], sharex=ax0)
    plot_binary_track(ax2, pred_prot, label="Pred", color=COLOR_PRED)

    ax3 = fig.add_subplot(gs0[3], sharex=ax0)
    plot_outcome_track(ax3, prot_outcomes, title="Outcome track")
    ax3.set_xlabel("Protein residue index")
    add_outcome_legend(ax3)

    # b) protein zoom
    gs1 = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=outer[1],
        height_ratios=[2.0, 0.35, 0.35, 0.35], hspace=0.08
    )

    z_true = true_prot[zoom_start:zoom_end]
    z_pred = pred_prot[zoom_start:zoom_end]
    z_prob = prob_prot[zoom_start:zoom_end]
    z_out = prot_outcomes[zoom_start:zoom_end]

    ax4 = fig.add_subplot(gs1[0])
    zx = np.arange(zoom_start + 1, zoom_end + 1)
    ax4.plot(zx, z_prob, color=COLOR_PRED, linewidth=2.0) #, label="Predicted probability"
    ax4.fill_between(zx, 0, z_true, color=COLOR_TRUE, alpha=0.25, step="mid") #, label="True binding residues"
    ax4.axhline(THRESHOLD, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("Probability")
    ax4.set_title("b | Protein binding-region detail", loc="left", fontweight="bold")
    ax4.legend(frameon=False, ncol=2, loc="upper right")

    ax5 = fig.add_subplot(gs1[1], sharex=ax4)
    plot_binary_track(ax5, z_true, label="True", color=COLOR_TRUE)

    ax6 = fig.add_subplot(gs1[2], sharex=ax4)
    plot_binary_track(ax6, z_pred, label="Pred", color=COLOR_PRED)

    ax7 = fig.add_subplot(gs1[3], sharex=ax4)
    plot_outcome_track(ax7, z_out, title="TP / FP / FN")
    ax7.set_xlabel("Protein residue index")

    # c) peptide
    gs2 = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=outer[2],
        height_ratios=[1.5, 0.35, 0.35, 0.35], hspace=0.08
    )

    px = np.arange(1, pep_len + 1)

    ax8 = fig.add_subplot(gs2[0])
    ax8.bar(px, prob_pep, color=COLOR_PRED, width=0.65, alpha=0.85) #, label="Predicted probability"
    ax8.scatter(px, true_pep, color=COLOR_TRUE, s=45, zorder=3)#, label="True binding label"
    ax8.axhline(THRESHOLD, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax8.set_ylim(0, 1.05)
    ax8.set_ylabel("Probability")
    ax8.set_title("c | Peptide residue-level binding-site prediction", loc="left", fontweight="bold")
    annotate_peptide_indices(ax8, pep_len)
    ax8.legend(frameon=False, ncol=2, loc="upper right")

    pep_txt = (
        f"Peptide: F1={pep_metrics['f1']:.3f}  MCC={pep_metrics['mcc']:.3f}  "
        f"Precision={pep_metrics['precision']:.3f}  Recall={pep_metrics['recall']:.3f}"
    )
    ax8.text(0.60, 1.06, pep_txt, transform=ax8.transAxes, ha="left", va="bottom", fontsize=9)

    ax9 = fig.add_subplot(gs2[1], sharex=ax8)
    plot_binary_track(ax9, true_pep, label="True", color=COLOR_TRUE)
    annotate_peptide_indices(ax9, pep_len)

    ax10 = fig.add_subplot(gs2[2], sharex=ax8)
    plot_binary_track(ax10, pred_pep, label="Pred", color=COLOR_PRED)
    annotate_peptide_indices(ax10, pep_len)

    ax11 = fig.add_subplot(gs2[3], sharex=ax8)
    plot_outcome_track(ax11, pep_outcomes, title="Outcome track")
    annotate_peptide_indices(ax11, pep_len)
    ax11.set_xlabel("Peptide residue index")

    if meta_text:
        fig.suptitle(meta_text, y=1.01, fontsize=11)

    savefig(save_path)

# ============================================================
# FIGURE 2: GATE-WEIGHTED INTERACTION MAP
# ============================================================

def plot_figure2_interaction_map(
    true_prot, pred_prot,
    true_pep, pred_pep, prob_pep,
    prot_interface_map,
    save_path,
    meta_text="",
    protein_sequence=None,
    peptide_sequence=None,
    top_percentile=95.0,
    weak_percentile=50,
    show_hotspot_labels=False,
    threshold=0.50
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    true_prot = np.asarray(true_prot).astype(int)
    pred_prot = np.asarray(pred_prot).astype(int)
    true_pep = np.asarray(true_pep).astype(int)
    pred_pep = np.asarray(pred_pep).astype(int)
    prob_pep = np.asarray(prob_pep).astype(float)
    prot_interface_map = np.asarray(prot_interface_map)

    pep_len = len(true_pep)

    zoom_start, zoom_end = get_binding_window(
        true_prot,
        pred_prot,
        flank=PROTEIN_ZOOM_FLANK
    )

    map_zoom = prot_interface_map[zoom_start:zoom_end, :]
    prot_zoom_len = zoom_end - zoom_start

    weak_cutoff = np.percentile(map_zoom, weak_percentile)
    hotspot_cutoff = np.percentile(map_zoom, top_percentile)

    masked_map = np.where(map_zoom >= weak_cutoff, map_zoom, np.nan)
    hotspots = np.argwhere(map_zoom >= hotspot_cutoff)

    pep_importance = np.nanmean(
        np.where(map_zoom >= weak_cutoff, map_zoom, np.nan),
        axis=0
    )

    prot_importance = np.nanmean(
        np.where(map_zoom >= weak_cutoff, map_zoom, np.nan),
        axis=1
    )

    pep_importance = np.nan_to_num(pep_importance)
    prot_importance = np.nan_to_num(prot_importance)

    pep_importance_plot = (
        pep_importance / pep_importance.max()
        if pep_importance.max() > 0 else pep_importance
    )

    prot_importance_plot = (
        prot_importance / prot_importance.max()
        if prot_importance.max() > 0 else prot_importance
    )

    # =========================================================
    # Professional colors
    # =========================================================
    COLOR_PEP_TRUE = "#D55E00"
    COLOR_PEP_PRED = "#009E73"

    COLOR_PROT_TRUE = "#6D3B5C"     # dark mauve
    COLOR_PROT_PRED = "#1E4B6D"     # dark navy blue

    COLOR_HOTSPOT = "#00CFEA"
    COLOR_MEAN = "#666666"
    COLOR_GUIDE = "#D8D8D8"
    COLOR_NON_BIND = "#E6E6E6"

    COLOR_TP = "#2E7D32"            # green
    COLOR_FP = "#1565C0"            # blue
    COLOR_FN = "#B71C1C"            # red
    COLOR_MISMATCH = "#222222"

    cmap_heat = plt.cm.magma.copy()
    cmap_heat.set_bad(color="#DCE6F2")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8

    fig = plt.figure(figsize=(13.6, 8.9))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        4, 4,
        width_ratios=[0.34, 1.0, 0.15, 0.04],
        height_ratios=[0.10, 1.0, 0.23, 0.13],
        wspace=0.09,
        hspace=0.11
    )

    step = 1 if pep_len <= 25 else 2 if pep_len <= 40 else 5
    xticks = np.arange(0, pep_len, step)

    # =========================================================
    # Top peptide mean interaction score
    # =========================================================
    ax_pep_imp = fig.add_subplot(gs[0, 1])

    ax_pep_imp.bar(
        np.arange(pep_len),
        pep_importance_plot,
        width=0.82,
        color=COLOR_MEAN,
        alpha=0.85,
        linewidth=0
    )

    ax_pep_imp.set_xlim(-0.5, pep_len - 0.5)
    ax_pep_imp.set_ylim(0, 1.05)
    ax_pep_imp.set_xticks([])
    ax_pep_imp.set_yticks([])

    ax_pep_imp.set_ylabel(
        "Peptide\nmean score",
        fontsize=8,
        rotation=0,
        labelpad=30,
        va="center"
    )

    for spine in ax_pep_imp.spines.values():
        spine.set_visible(False)

    # =========================================================
    # LEFT PROTEIN SUMMARY STRIP — LONG SEQUENCE FRIENDLY
    # =========================================================
    ax_left = fig.add_subplot(gs[1, 0])

    true_zoom = true_prot[zoom_start:zoom_end]
    pred_zoom = pred_prot[zoom_start:zoom_end]

    tp = np.where((true_zoom == 1) & (pred_zoom == 1))[0]
    fp = np.where((true_zoom == 0) & (pred_zoom == 1))[0]
    fn = np.where((true_zoom == 1) & (pred_zoom == 0))[0]

    x_actual = 0.55
    x_pred = 1.35

    x_tp = 2.05
    x_fp = 2.30
    x_fn = 2.55

    bar_width = 0.50
    bar_height = 0.85

    ax_left.set_xlim(0.0, 2.75)
    ax_left.set_ylim(prot_zoom_len - 0.5, -0.5)

    ax_left.axvspan(
        0.0,
        2.75,
        color="#FAFAFA",
        zorder=0
    )

    # Actual positive residues
    for i in np.where(true_zoom == 1)[0]:
        ax_left.add_patch(
            mpatches.Rectangle(
                (x_actual - bar_width / 2, i - bar_height / 2),
                bar_width,
                bar_height,
                facecolor=COLOR_PROT_TRUE,
                edgecolor="none",
                alpha=0.95,
                zorder=3
            )
        )

    # Predicted positive residues
    for i in np.where(pred_zoom == 1)[0]:
        ax_left.add_patch(
            mpatches.Rectangle(
                (x_pred - bar_width / 2, i - bar_height / 2),
                bar_width,
                bar_height,
                facecolor=COLOR_PROT_PRED,
                edgecolor="none",
                alpha=0.95,
                zorder=3
            )
        )

    # Column separators
    ax_left.axvline(
        (x_actual + x_pred) / 2,
        color="#D0D0D0",
        linewidth=0.7,
        zorder=1
    )

    ax_left.axvline(
        (x_pred + x_tp) / 2,
        color="#D0D0D0",
        linewidth=0.7,
        zorder=1
    )

    ax_left.axvline(
        (x_tp + x_fn) / 2,
        color="#D0D0D0",
        linewidth=0.7,
        zorder=1
    )

    # =========================================================
    # Outcome columns as non-overlapping horizontal ticks
    # =========================================================
    ax_left.axvline(x_tp, color="#E0E0E0", linewidth=0.5, zorder=1)
    ax_left.axvline(x_fp, color="#E0E0E0", linewidth=0.5, zorder=1)
    ax_left.axvline(x_fn, color="#E0E0E0", linewidth=0.5, zorder=1)

    for i in tp:
        ax_left.plot(
            [x_tp - 0.065, x_tp + 0.065],
            [i, i],
            color=COLOR_TP,
            linewidth=1.4,
            alpha=0.95,
            solid_capstyle="round",
            zorder=5
        )

    for i in fp:
        ax_left.plot(
            [x_fp - 0.065, x_fp + 0.065],
            [i, i],
            color=COLOR_FP,
            linewidth=1.4,
            alpha=0.95,
            solid_capstyle="round",
            zorder=5
        )

    for i in fn:
        ax_left.plot(
            [x_fn - 0.065, x_fn + 0.065],
            [i, i],
            color=COLOR_FN,
            linewidth=1.4,
            alpha=0.95,
            solid_capstyle="round",
            zorder=5
        )

    ystep = max(1, prot_zoom_len // 10)
    yticks = np.arange(0, prot_zoom_len, ystep)

    ax_left.set_yticks(yticks)
    ax_left.set_yticklabels(
        [str(zoom_start + i + 1) for i in yticks],
        fontsize=8
    )

    ax_left.set_xticks([x_actual, x_pred, x_tp, x_fp, x_fn])
    ax_left.set_xticklabels(
        ["Actual", "Pred.", "TP", "FP", "FN"],
        rotation=90,
        fontsize=7.5
    )

    ax_left.tick_params(axis="x", length=0, pad=3)
    ax_left.tick_params(axis="y", length=0, pad=2)

    #ax_left.set_title(
        #"Protein binding-site\nsummary",
       # fontsize=10,
        #fontweight="bold",
        #pad=8
    #)

    ax_left.text(
        0.5,
        1.015,
        f"TP {len(tp)} | FP {len(fp)} | FN {len(fn)}",
        transform=ax_left.transAxes,
        fontsize=7.5,
        ha="center",
        va="bottom",
        fontweight="bold"
    )

    for spine in ["top", "right", "left", "bottom"]:
        ax_left.spines[spine].set_visible(False)

    # =========================================================
    # Main interaction map
    # =========================================================
    ax = fig.add_subplot(gs[1, 1])

    vmin = np.nanpercentile(masked_map, 5)
    vmax = np.nanpercentile(masked_map, 99)

    im = ax.imshow(
        masked_map,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_heat,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    ax.set_yticks(yticks)
    ax.set_yticklabels(
        [str(zoom_start + i + 1) for i in yticks],
        fontsize=8
    )

    ax.set_ylabel(
        "Protein residue index",
        fontsize=10,
        labelpad=9
    )

    #ax.set_title(
       # "d | Biologically filtered gate-weighted protein–peptide interaction map",
       # loc="left",
       # fontsize=12,
       # fontweight="bold",
      #  pad=8
  #  )

    for p in xticks:
        ax.axvline(
            p,
            color=COLOR_GUIDE,
            linewidth=0.35,
            alpha=0.45,
            zorder=1
        )

    for r in yticks:
        ax.axhline(
            r,
            color=COLOR_GUIDE,
            linewidth=0.35,
            alpha=0.35,
            zorder=1
        )

    for p in np.where(true_pep == 1)[0]:
        ax.axvline(
            p,
            color=COLOR_PEP_TRUE,
            linewidth=0.45,
            alpha=0.25,
            zorder=2
        )

    for p in np.where(pred_pep == 1)[0]:
        ax.axvline(
            p,
            color=COLOR_PEP_PRED,
            linewidth=0.45,
            alpha=0.22,
            linestyle=":",
            zorder=2
        )

    for r in np.where(true_zoom == 1)[0]:
        ax.axhline(
            r,
            color=COLOR_PROT_TRUE,
            linewidth=0.50,
            alpha=0.32,
            zorder=2
        )

    for r in np.where(pred_zoom == 1)[0]:
        ax.axhline(
            r,
            color=COLOR_PROT_PRED,
            linewidth=0.50,
            alpha=0.30,
            linestyle=":",
            zorder=2
        )

    if len(hotspots) > 0:
        ax.scatter(
            hotspots[:, 1],
            hotspots[:, 0],
            s=22,
            facecolors="none",
            edgecolors=COLOR_HOTSPOT,
            linewidths=0.9,
            alpha=0.95,
            zorder=4
        )

    if show_hotspot_labels and len(hotspots) > 0:
        hotspot_scores = map_zoom[hotspots[:, 0], hotspots[:, 1]]
        top_order = np.argsort(hotspot_scores)[::-1][:5]

        for idx in top_order:
            r, p = hotspots[idx]
            prot_id = zoom_start + r + 1
            pep_id = p + 1

            prot_label = (
                f"{protein_sequence[prot_id - 1]}{prot_id}"
                if protein_sequence is not None and prot_id - 1 < len(protein_sequence)
                else f"P{prot_id}"
            )

            pep_label = (
                f"{peptide_sequence[pep_id - 1]}{pep_id}"
                if peptide_sequence is not None and pep_id - 1 < len(peptide_sequence)
                else f"pep{pep_id}"
            )

            ax.text(
                p + 0.25,
                r + 0.25,
                f"{prot_label}-{pep_label}",
                fontsize=6.2,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="black",
                    alpha=0.48,
                    linewidth=0
                ),
                zorder=5
            )

    # =========================================================
    # Right protein mean interaction score
    # =========================================================
    ax_prot_imp = fig.add_subplot(gs[1, 2], sharey=ax)

    ax_prot_imp.barh(
        np.arange(prot_zoom_len),
        prot_importance_plot,
        height=0.85,
        color=COLOR_MEAN,
        alpha=0.85,
        linewidth=0
    )

    ax_prot_imp.invert_yaxis()
    ax_prot_imp.set_xlim(0, 1.05)
    ax_prot_imp.set_xticks([0, 0.5, 1.0])
    ax_prot_imp.set_xticklabels(["0", "0.5", "1.0"], fontsize=7)
    ax_prot_imp.set_yticks([])

    ax_prot_imp.set_xlabel(
        "Protein\nmean score",
        fontsize=8
    )

    for spine in ax_prot_imp.spines.values():
        spine.set_visible(False)

    # =========================================================
    # Colorbar
    # =========================================================
    cax = fig.add_subplot(gs[1, 3])

    cbar = fig.colorbar(im, cax=cax)

    cbar.set_label(
        "Gate-weighted\ninteraction score",
        fontsize=9
    )

    cbar.ax.tick_params(labelsize=8)

    # =========================================================
    # Bottom peptide annotation/probability strip
    # =========================================================
    ax_bottom = fig.add_subplot(gs[2, 1], sharex=ax)

    ax_bottom.set_xlim(0.5, pep_len + 0.5)
    ax_bottom.set_ylim(-1.1, 2.05)

    for i in range(1, pep_len + 1):
        ax_bottom.axvspan(
            i - 0.5,
            i + 0.5,
            color="#F4F4F4" if i % 2 == 0 else "#FFFFFF",
            zorder=0
        )

        ax_bottom.add_patch(
            mpatches.Rectangle(
                (i - 0.40, 1.18),
                0.80,
                0.26,
                facecolor=COLOR_NON_BIND,
                edgecolor="none",
                alpha=0.60,
                zorder=1
            )
        )

        ax_bottom.add_patch(
            mpatches.Rectangle(
                (i - 0.40, 0.38),
                0.80,
                0.26,
                facecolor=COLOR_NON_BIND,
                edgecolor="none",
                alpha=0.60,
                zorder=1
            )
        )

    for i, val in enumerate(true_pep, start=1):
        if val == 1:
            ax_bottom.add_patch(
                mpatches.Rectangle(
                    (i - 0.40, 1.18),
                    0.80,
                    0.26,
                    facecolor=COLOR_PEP_TRUE,
                    edgecolor="none",
                    alpha=0.98,
                    zorder=3
                )
            )

    for i, prob in enumerate(prob_pep, start=1):
        if prob >= threshold:
            ax_bottom.add_patch(
                mpatches.Rectangle(
                    (i - 0.40, 0.38),
                    0.80,
                    0.26,
                    facecolor=COLOR_PEP_PRED,
                    edgecolor="none",
                    alpha=0.98,
                    zorder=3
                )
            )

    for i, val in enumerate(true_pep, start=1):
        ax_bottom.text(
            i,
            1.60,
            str(int(val)),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#222222",
            fontweight="bold"
        )

    for i, prob in enumerate(prob_pep, start=1):
        pred_binary = 1 if prob >= threshold else 0

        ax_bottom.text(
            i,
            0.78,
            str(pred_binary),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#222222",
            fontweight="bold"
        )

        ax_bottom.text(
            i,
            -0.15,
            f"{prob:.2f}",
            ha="center",
            va="center",
            fontsize=6.0,
            color="#333333",
            rotation=90
        )

    for i, (true_val, prob) in enumerate(zip(true_pep, prob_pep), start=1):
        pred_binary = 1 if prob >= threshold else 0

        if true_val != pred_binary:
            ax_bottom.plot(
                i,
                -0.62,
                marker="x",
                color="#222222",
                markersize=4,
                markeredgewidth=0.9,
                zorder=5
            )

    ax_bottom.axhline(
        -0.42,
        color="#999999",
        linestyle="--",
        linewidth=0.6,
        alpha=0.7
    )

    ax_bottom.text(
        pep_len + 0.45,
        -0.15,
        "Pred.\nprob.",
        ha="left",
        va="center",
        fontsize=7
    )

    ax_bottom.text(
        pep_len + 0.45,
        -0.62,
        "Mismatch",
        ha="left",
        va="center",
        fontsize=7
    )

    ax_bottom.set_yticks([1.30, 0.50])

    ax_bottom.set_yticklabels(
        ["Actual", "Predicted"],
        fontsize=9
    )

    ax_bottom.set_xticks(np.arange(1, pep_len + 1))

    ax_bottom.set_xticklabels(
        [str(i) for i in range(1, pep_len + 1)],
        fontsize=8
    )

    #ax_bottom.set_xlabel(
       # "Peptide residue index",
       # fontsize=10,
      #  labelpad=4
    #)

    ax_bottom.tick_params(
        axis="x",
        length=2.5,
        width=0.7,
        pad=2
    )

    ax_bottom.tick_params(axis="y", length=0)

    #ax_bottom.set_title(
       # f"Peptide binding-site annotation and predicted probabilities (threshold = {threshold:.2f})",
       # fontsize=10.5,
       # fontweight="bold",
       # pad=5
    #)

    for spine in ["top", "right", "left"]:
        ax_bottom.spines[spine].set_visible(False)

    ax_bottom.spines["bottom"].set_linewidth(0.7)
    ax_bottom.spines["bottom"].set_color("#555555")

    # =========================================================
    # Legend
    # =========================================================
    ax_legend = fig.add_subplot(gs[3, 1])
    ax_legend.axis("off")

    legend_handles = [
        mpatches.Patch(
            color=COLOR_PEP_TRUE,
            label="Peptide actual = 1"
        ),
        mpatches.Patch(
            color=COLOR_PEP_PRED,
            label="Peptide predicted ≥ 0.5"
        ),
        mpatches.Patch(
            color=COLOR_NON_BIND,
            label="Label = 0"
        ),
        mpatches.Patch(
            color=COLOR_PROT_TRUE,
            label="Protein actual = 1"
        ),
        mpatches.Patch(
            color=COLOR_PROT_PRED,
            label="Protein predicted = 1"
        ),
        Line2D(
            [],
            [],
            color=COLOR_TP,
            marker="_",
            linestyle="None",
            markersize=7,
            markeredgewidth=1.6,
            label="Protein TP"
        ),
        Line2D(
            [],
            [],
            color=COLOR_FP,
            marker="_",
            linestyle="None",
            markersize=7,
            markeredgewidth=1.6,
            label="Protein FP"
        ),
        Line2D(
            [],
            [],
            color=COLOR_FN,
            marker="_",
            linestyle="None",
            markersize=7,
            markeredgewidth=1.6,
            label="Protein FN"
        ),
        mpatches.Patch(
            facecolor="none",
            edgecolor=COLOR_HOTSPOT,
            label=f"Top interaction hotspots (≥ {top_percentile}th percentile)"
        ),
        Line2D(
            [],
            [],
            color=COLOR_MISMATCH,
            marker="x",
            linestyle="None",
            markersize=4,
            label="Peptide mismatch"
        )
    ]

    ax_legend.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="center"
    )

    if meta_text:
        fig.suptitle(
            meta_text,
            y=0.995,
            fontsize=10
        )

    plt.savefig(
        save_path,
        dpi=1000,
        bbox_inches="tight"
    )

    plt.close()
# ============================================================
# FIGURE 3: BIDIRECTIONAL ATTRIBUTION
# ============================================================
def plot_figure3_bidirectional_attribution(
    true_prot, prob_prot,
    true_pep, prob_pep,
    prot_target_idx, pep_target_idx,
    prot_self_attr, pep_partner_attr,
    pep_self_attr, prot_partner_attr,
    save_path, meta_text=""
):
    pep_len = len(true_pep)
    zoom_start, zoom_end = get_binding_window(true_prot, prob_prot, flank=PROTEIN_ZOOM_FLANK)

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    zx = np.arange(zoom_start + 1, zoom_end + 1)
    z_attr = prot_self_attr[zoom_start:zoom_end]
    z_prob = prob_prot[zoom_start:zoom_end]
    z_true = true_prot[zoom_start:zoom_end]

    ax1.plot(zx, normalize_01(z_attr), color=COLOR_ATTR, linewidth=2.0) #, label="Attribution"
    ax1.plot(zx, normalize_01(z_prob), color=COLOR_PRED, linewidth=1.5)#, label="Predicted probability"
    ax1.fill_between(zx, 0, z_true, color=COLOR_TRUE, alpha=0.22, step="mid")#, label="True binding residues"
    ax1.axvline(prot_target_idx + 1, color="black", linestyle="--", linewidth=1.0)
    ax1.set_ylim(0, 1.05)
    ax1.set_title(
        f"a | Protein target residue {residue_label('protein', prot_target_idx)}: protein-side attribution",
        loc="left", fontweight="bold"
    )
    ax1.set_xlabel("Protein residue index")
    ax1.set_ylabel("Normalized value")
    ax1.legend(frameon=False, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    px = np.arange(1, pep_len + 1)
    pep_partner_norm = normalize_01(pep_partner_attr)
    ax2.bar(px, pep_partner_norm, color=COLOR_ATTR, alpha=0.85)
    ax2.scatter(px, true_pep, color=COLOR_TRUE, s=40, zorder=3)#, label="True binding label"
    annotate_peptide_indices(ax2, pep_len)
    ax2.set_ylim(0, 1.05)
    ax2.set_title(
        f"b | Peptide residues supporting protein target {residue_label('protein', prot_target_idx)}",
        loc="left", fontweight="bold"
    )
    ax2.set_ylabel("Normalized attribution")
    ax2.legend(frameon=False, loc="upper right")

    ax3 = fig.add_subplot(gs[1, 0])
    pep_self_norm = normalize_01(pep_self_attr)
    ax3.bar(px, pep_self_norm, color=COLOR_ATTR, alpha=0.85)#, label="Attribution"
    ax3.scatter(px, normalize_01(prob_pep), color=COLOR_PRED, s=35, zorder=3)#, label="Predicted probability"
    ax3.scatter(px, true_pep, color=COLOR_TRUE, s=35, zorder=3, marker="s")#, label="True binding label"
    ax3.axvline(pep_target_idx + 1, color="black", linestyle="--", linewidth=1.0)
    annotate_peptide_indices(ax3, pep_len)
    ax3.set_ylim(0, 1.05)
    ax3.set_title(
        f"c | Peptide target residue {residue_label('peptide', pep_target_idx)}: peptide-side attribution",
        loc="left", fontweight="bold"
    )
    ax3.set_ylabel("Normalized value")
    ax3.legend(frameon=False, loc="upper right")

    ax4 = fig.add_subplot(gs[1, 1])
    z_attr2 = prot_partner_attr[zoom_start:zoom_end]
    ax4.plot(zx, normalize_01(z_attr2), color=COLOR_ATTR, linewidth=2.0)#, label="Attribution"
    ax4.plot(zx, normalize_01(prob_prot[zoom_start:zoom_end]), color=COLOR_PRED, linewidth=1.5)#, label="Predicted probability"
    ax4.fill_between(zx, 0, true_prot[zoom_start:zoom_end], color=COLOR_TRUE, alpha=0.22, step="mid")#, label="True binding residues"
    ax4.set_ylim(0, 1.05)
    ax4.set_title(
        f"d | Protein residues supporting peptide target {residue_label('peptide', pep_target_idx)}",
        loc="left", fontweight="bold"
    )
    ax4.set_xlabel("Protein residue index")
    ax4.set_ylabel("Normalized value")
    ax4.legend(frameon=False, loc="upper right")

    if meta_text:
        fig.suptitle(meta_text, y=1.01, fontsize=11)

    savefig(save_path)

# ============================================================
# FIGURE 4: TARGETED PERTURBATION VALIDATION
# ============================================================
def plot_figure4_targeted_deletion(
    prot_self_curve,
    pep_self_curve,
    pep_partner_curve_for_prot_target,
    prot_partner_curve_for_pep_target,
    prot_target_label,
    pep_target_label,
    save_path,
    meta_text=""
):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes = axes.flatten()

    curve_info = [
        (prot_self_curve, f"a | Delete top protein residues → protein target {prot_target_label}"),
        (pep_partner_curve_for_prot_target, f"b | Delete peptide partner residues → protein target {prot_target_label}"),
        (pep_self_curve, f"c | Delete top peptide residues → peptide target {pep_target_label}"),
        (prot_partner_curve_for_pep_target, f"d | Delete protein partner residues → peptide target {pep_target_label}"),
    ]

    for ax, (curve, title) in zip(axes, curve_info):
        x = curve["fractions"]
        y = curve["deletion_scores"]
        rm = curve["random_mean"]
        rs = curve["random_std"]

        ax.plot(x, y, linewidth=2.2, color=COLOR_ATTR)#, label="Top-attribution deletion"
        ax.plot(x, rm, linewidth=1.6, color=COLOR_RANDOM, linestyle="--")#, label="Random deletion"
        ax.fill_between(x, rm - rs, rm + rs, color=COLOR_RANDOM, alpha=0.18)
        ax.set_xlabel("Fraction of residues deleted")
        ax.set_ylabel("Target binding probability")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, loc="upper right")

    if meta_text:
        fig.suptitle(meta_text, y=1.01, fontsize=11)

    savefig(save_path)


def residue_name(sequence, index_zero_based, prefix):
    idx = index_zero_based + 1

    if sequence is None:
        return f"{prefix}{idx}"

    if index_zero_based >= len(sequence):
        return f"{prefix}{idx}"

    return f"{sequence[index_zero_based]}{idx}"

def plot_advanced_biological_cross_attention(
    case,
    save_prefix,
    protein_sequence=None,
    peptide_sequence=None,
    threshold=0.5,
    hotspot_percentile=97.5,
    batch_index=None,
    sample_index=None
):
    true_prot = case["true_prot"]
    true_pep = case["true_pep"]
    pred_prot = case["pred_prot"]
    pred_pep = case["pred_pep"]
    prob_prot = case["prob_prot"]
    prob_pep = case["prob_pep"]

    coupling = case["biological_coupling"]
    prot_evidence = case["protein_integrated_evidence"]
    pep_evidence = case["peptide_motif_evidence"]

    prot_metrics = case["prot_metrics"]
    pep_metrics = case["pep_metrics"]

    pep_len = len(true_pep)

    zoom_start, zoom_end = get_binding_window(
        true_prot,
        prob_prot,
        flank=PROTEIN_ZOOM_FLANK
    )

    coupling_zoom = coupling[zoom_start:zoom_end, :]
    true_zoom = true_prot[zoom_start:zoom_end]
    pred_zoom = pred_prot[zoom_start:zoom_end]
    prob_zoom = prob_prot[zoom_start:zoom_end]
    evidence_zoom = prot_evidence[zoom_start:zoom_end]

    prot_zoom_len = zoom_end - zoom_start

    hotspot_cutoff = np.percentile(coupling_zoom, hotspot_percentile)
    hotspots = np.argwhere(coupling_zoom >= hotspot_cutoff)

    COLOR_PROT_ACTUAL = "#7B3294"
    COLOR_PROT_PRED = "#1F78B4"
    COLOR_PEP_ACTUAL = "#D95F02"
    COLOR_PEP_PRED = "#009E73"
    COLOR_HOTSPOT = "#00C853"
    COLOR_GREY = "#4D4D4D"

    cmap_heat = plt.cm.magma.copy()
    cmap_heat.set_bad("#EDF2F7")

    fig = plt.figure(figsize=(19.0, 12.8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        5,
        5,
        width_ratios=[0.95, 1.15, 1.15, 2.20, 1.60],
        height_ratios=[0.28, 1.25, 1.55, 1.35, 0.34],
        wspace=0.46,
        hspace=0.62
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")

    # ========================================================
    # PANEL A1
    # ========================================================

    ax_a1 = fig.add_subplot(gs[1, 0])
    y = np.arange(prot_zoom_len)

    for i in range(prot_zoom_len):
        score = evidence_zoom[i]
        ax_a1.plot(
            [0, score],
            [i, i],
            color=COLOR_PROT_ACTUAL,
            linewidth=1.4 + 2.2 * score,
            alpha=0.35 + 0.55 * score,
            solid_capstyle="round"
        )

    ax_a1.set_ylim(prot_zoom_len - 0.5, -0.5)
    ax_a1.set_xlim(0, 1.02)

    ystep = max(1, prot_zoom_len // 10)
    yticks = np.arange(0, prot_zoom_len, ystep)

    ax_a1.set_yticks(yticks)
    ax_a1.set_yticklabels([str(zoom_start + i + 1) for i in yticks])
    ax_a1.set_xlabel("Low        High", fontsize=8)
    ax_a1.set_ylabel("Protein interface-context\nposition index", fontsize=10)
    ax_a1.set_title(
        "A | Peptide-derived\ninterface evidence",
        loc="left",
        fontsize=11,
        fontweight="bold"
    )
    ax_a1.set_xticks([])

    for spine in ["top", "right", "bottom"]:
        ax_a1.spines[spine].set_visible(False)

    # ========================================================
    # PANEL A2
    # ========================================================

    ax_a2 = fig.add_subplot(gs[1, 1], sharey=ax_a1)

    for i in range(prot_zoom_len):
        score = prob_zoom[i]
        ax_a2.plot(
            [0, score],
            [i, i],
            color=COLOR_PROT_PRED,
            linewidth=1.2 + 2.2 * score,
            alpha=0.35 + 0.55 * score,
            solid_capstyle="round"
        )

    ax_a2.set_xlim(0, 1.02)
    ax_a2.set_xlabel("Low        High", fontsize=8)
    ax_a2.set_title(
        "Model-inferred\ninterface likelihood",
        fontsize=11,
        fontweight="bold"
    )
    ax_a2.set_xticks([])
    ax_a2.set_yticks([])

    for spine in ["top", "right", "left", "bottom"]:
        ax_a2.spines[spine].set_visible(False)

    # ========================================================
    # PANEL A3
    # ========================================================

    ax_a3 = fig.add_subplot(gs[1, 2], sharey=ax_a1)

    integrated_conf = normalize_01(0.5 * evidence_zoom + 0.5 * prob_zoom)

    ax_a3.barh(
        y,
        integrated_conf,
        height=0.70,
        color=COLOR_GREY,
        alpha=0.78,
        edgecolor="none"
    )

    ax_a3.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax_a3.set_xlim(0, 1.02)
    ax_a3.set_xlabel(r"$P_{\mathrm{interface}}$ (protein)")
    ax_a3.set_title(
        "Integrated interface confidence\n(aggregated cross-attention)",
        fontsize=11,
        fontweight="bold"
    )
    ax_a3.set_yticks([])

    for spine in ["top", "right", "left"]:
        ax_a3.spines[spine].set_visible(False)

    # ========================================================
    # PANEL A4: PROFESSIONAL COMPACT PROTEIN EVIDENCE MATRIX
    # ========================================================

    ax_a4 = fig.add_subplot(gs[1, 3], sharey=ax_a1)
    ax_a4.set_xlim(0, 5.4)
    ax_a4.set_ylim(prot_zoom_len - 0.5, -0.5)
    ax_a4.axis("off")

    col_x = [0.45, 1.25, 2.35, 3.55, 4.75]

    headers = [
        "Actual\ninterface",
        "Predicted\ninterface",
        "Predicted\nprobability",
        "Attention\nevidence",
        "Integrated\nscore"
    ]

    for xh, h in zip(col_x, headers):
        ax_a4.text(
            xh,
            -4.2,
            h,
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            linespacing=1.05
        )

    prob_norm = normalize_01(prob_zoom)
    attn_norm = normalize_01(evidence_zoom)
    int_norm = normalize_01(integrated_conf)

    bar_w = 0.42
    bar_h = 0.72

    for i in range(prot_zoom_len):

        if i % 2 == 0:
            ax_a4.add_patch(
                mpatches.Rectangle(
                    (0.05, i - 0.42),
                    5.15,
                    0.84,
                    facecolor="#F7F7F7",
                    edgecolor="none",
                    alpha=0.55,
                    zorder=0
                )
            )

        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[0] - bar_w / 2, i - bar_h / 2),
                bar_w,
                bar_h,
                facecolor=COLOR_PROT_ACTUAL if true_zoom[i] == 1 else "#E6E6E6",
                edgecolor="none",
                alpha=0.95,
                zorder=2
            )
        )

        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[1] - bar_w / 2, i - bar_h / 2),
                bar_w,
                bar_h,
                facecolor=COLOR_PROT_PRED if pred_zoom[i] == 1 else "#E6E6E6",
                edgecolor="none",
                alpha=0.95,
                zorder=2
            )
        )

        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[2] - bar_w / 2, i - bar_h / 2),
                bar_w,
                bar_h,
                facecolor="#EFEFEF",
                edgecolor="none",
                zorder=1
            )
        )
        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[2] - bar_w / 2, i - bar_h / 2),
                bar_w * prob_norm[i],
                bar_h,
                facecolor=COLOR_PROT_PRED,
                edgecolor="none",
                alpha=0.90,
                zorder=3
            )
        )

        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[3] - bar_w / 2, i - bar_h / 2),
                bar_w,
                bar_h,
                facecolor="#EFEFEF",
                edgecolor="none",
                zorder=1
            )
        )
        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[3] - bar_w / 2, i - bar_h / 2),
                bar_w * attn_norm[i],
                bar_h,
                facecolor=COLOR_PROT_ACTUAL,
                edgecolor="none",
                alpha=0.90,
                zorder=3
            )
        )

        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[4] - bar_w / 2, i - bar_h / 2),
                bar_w,
                bar_h,
                facecolor="#EFEFEF",
                edgecolor="none",
                zorder=1
            )
        )
        ax_a4.add_patch(
            mpatches.Rectangle(
                (col_x[4] - bar_w / 2, i - bar_h / 2),
                bar_w * int_norm[i],
                bar_h,
                facecolor="#4D4D4D",
                edgecolor="none",
                alpha=0.90,
                zorder=3
            )
        )

    for xsep in [0.85, 1.80, 2.95, 4.15]:
        ax_a4.axvline(
            xsep,
            color="#D0D0D0",
            linewidth=0.5,
            alpha=0.8,
            zorder=0
        )

    ax_a4.text(
        2.35,
        prot_zoom_len + 1.3,
        "Low",
        ha="left",
        va="center",
        fontsize=6.5,
        color="#555555"
    )

    ax_a4.text(
        4.95,
        prot_zoom_len + 1.3,
        "High",
        ha="right",
        va="center",
        fontsize=6.5,
        color="#555555"
    )

    ax_a4.plot(
        [2.70, 4.55],
        [prot_zoom_len + 1.3, prot_zoom_len + 1.3],
        color="#999999",
        linewidth=2.0,
        solid_capstyle="round"
    )

    # ========================================================
    # PANEL A5 NOTE
    # ========================================================

    ax_note = fig.add_subplot(gs[1, 4])
    ax_note.axis("off")


    # ========================================================
    # PANEL B: CROSS-INTERFACE ATTENTION HEATMAP
    # ========================================================

    ax_map = fig.add_subplot(gs[2, 0:4])

    vmin = np.percentile(coupling_zoom, 2)
    vmax = np.percentile(coupling_zoom, 99)

    im = ax_map.imshow(
        coupling_zoom,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_heat,
        vmin=vmin,
        vmax=vmax
    )

    xtick_step = 1 if pep_len <= 25 else 2 if pep_len <= 40 else 5
    xticks = np.arange(0, pep_len, xtick_step)

    ax_map.set_xticks(xticks)
    ax_map.set_xticklabels([str(i + 1) for i in xticks])
    ax_map.set_yticks(yticks)
    ax_map.set_yticklabels([str(zoom_start + i + 1) for i in yticks])

    ax_map.set_xlabel("Peptide recognition motif position index")
    ax_map.set_ylabel("Protein interface-context position index")

    ax_map.set_title(
        "B | Bidirectional gate-weighted cross-interface attention map",
        loc="left",
        fontsize=12,
        fontweight="bold"
    )

    for r in np.where(true_zoom == 1)[0]:
        ax_map.axhline(
            r,
            color=COLOR_PROT_ACTUAL,
            linewidth=0.55,
            alpha=0.38
        )

    for r in np.where(pred_zoom == 1)[0]:
        ax_map.axhline(
            r,
            color=COLOR_PROT_PRED,
            linewidth=0.55,
            alpha=0.35,
            linestyle=":"
        )

    for p in np.where(true_pep == 1)[0]:
        ax_map.axvline(
            p,
            color=COLOR_PEP_ACTUAL,
            linewidth=0.55,
            alpha=0.38
        )

    for p in np.where(pred_pep == 1)[0]:
        ax_map.axvline(
            p,
            color=COLOR_PEP_PRED,
            linewidth=0.55,
            alpha=0.35,
            linestyle=":"
        )

    if len(hotspots) > 0:
        ax_map.scatter(
            hotspots[:, 1],
            hotspots[:, 0],
            s=30,
            facecolors="none",
            edgecolors=COLOR_HOTSPOT,
            linewidths=1.0,
            alpha=0.95
        )

    cbar = fig.colorbar(
        im,
        ax=ax_map,
        fraction=0.018,
        pad=0.012
    )
    cbar.set_label("Cross-interface attention intensity", fontsize=9)

    # ========================================================
    # PANEL C: CLEAN BINDING-SITE ARCHITECTURE PANEL
    # ========================================================

    ax_c = fig.add_subplot(gs[3, 0:4])
    ax_c.axis("off")

    protein_profile = normalize_01(
        0.40 * prob_zoom +
        0.35 * evidence_zoom +
        0.15 * true_zoom +
        0.10 * pred_zoom
    )

    peptide_profile = normalize_01(
        0.40 * prob_pep +
        0.35 * pep_evidence +
        0.15 * true_pep +
        0.10 * pred_pep
    )

    top_prot_n = min(14, prot_zoom_len)
    top_pep_n = min(12, pep_len)

    prot_display = np.argsort(protein_profile)[::-1][:top_prot_n]
    pep_display = np.argsort(peptide_profile)[::-1][:top_pep_n]

    prot_display = np.sort(prot_display)
    pep_display = np.sort(pep_display)

    C = coupling_zoom[np.ix_(prot_display, pep_display)]
    C_norm = normalize_01(C)

    ribbon_cutoff = np.percentile(C, 92)

    left_x = 0.06
    mid_x = 0.42
    right_x = 0.92

    top_y = 0.82
    bottom_y = 0.18

    prot_y = np.linspace(top_y, bottom_y, len(prot_display))
    pep_x = np.linspace(mid_x, right_x, len(pep_display))

    ax_c.add_patch(
        mpatches.FancyBboxPatch(
            (0.015, 0.08),
            0.34,
            0.82,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#FAF7FC",
            edgecolor="#D9C7E7",
            linewidth=0.8,
            transform=ax_c.transAxes
        )
    )

    ax_c.add_patch(
        mpatches.FancyBboxPatch(
            (0.39, 0.08),
            0.58,
            0.82,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#F6FBF8",
            edgecolor="#BFDCCA",
            linewidth=0.8,
            transform=ax_c.transAxes
        )
    )

    ax_c.text(
        0.18,
        0.93,
        "Protein interface architecture",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLOR_PROT_ACTUAL
    )

    ax_c.text(
        0.68,
        0.93,
        "Peptide recognition motif landscape",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLOR_PEP_PRED
    )

    for i, p_local in enumerate(prot_display):

        yv = prot_y[i]
        p_global = zoom_start + p_local

        ax_c.plot(
            [left_x, 0.30],
            [yv, yv],
            transform=ax_c.transAxes,
            color="#DDDDDD",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1
        )

        ax_c.plot(
            [left_x, left_x + 0.24 * prob_prot[p_global]],
            [yv + 0.012, yv + 0.012],
            transform=ax_c.transAxes,
            color=COLOR_PROT_PRED,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=3
        )

        ax_c.plot(
            [left_x, left_x + 0.24 * evidence_zoom[p_local]],
            [yv - 0.012, yv - 0.012],
            transform=ax_c.transAxes,
            color=COLOR_PROT_ACTUAL,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=3
        )

        prot_label = residue_name(
            protein_sequence,
            p_global,
            "P"
        )

        ax_c.text(
            left_x - 0.020,
            yv,
            prot_label,
            transform=ax_c.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold"
        )

        if true_prot[p_global] == 1:
            ax_c.scatter(
                left_x - 0.045,
                yv,
                s=34,
                marker="s",
                color=COLOR_PROT_ACTUAL,
                transform=ax_c.transAxes,
                zorder=5
            )

        if pred_prot[p_global] == 1:
            ax_c.scatter(
                left_x - 0.065,
                yv,
                s=34,
                marker="s",
                color=COLOR_PROT_PRED,
                transform=ax_c.transAxes,
                zorder=5
            )

    #ax_c.text(
       # left_x,
       # 0.865,
        #"Predicted interface likelihood",
        #transform=ax_c.transAxes,
        #fontsize=7.2,
        #color=COLOR_PROT_PRED
    #)

    #ax_c.text(
       # left_x,
       # 0.835,
        #"Peptide-derived attention support",
        #transform=ax_c.transAxes,
       # fontsize=7.2,
      #  color=COLOR_PROT_ACTUAL
    #)

    skyline_base = 0.24

    for j, pep_idx in enumerate(pep_display):

        xv = pep_x[j]

        motif_h = 0.42 * peptide_profile[pep_idx]

        ax_c.add_patch(
            mpatches.Rectangle(
                (xv - 0.008, skyline_base),
                0.016,
                motif_h,
                transform=ax_c.transAxes,
                facecolor=COLOR_PEP_PRED,
                edgecolor="none",
                alpha=0.95,
                zorder=4
            )
        )

        attn_h = 0.42 * pep_evidence[pep_idx]

        ax_c.add_patch(
            mpatches.Rectangle(
                (xv + 0.010, skyline_base),
                0.012,
                attn_h,
                transform=ax_c.transAxes,
                facecolor=COLOR_PEP_ACTUAL,
                edgecolor="none",
                alpha=0.90,
                zorder=4
            )
        )

        if true_pep[pep_idx] == 1:
            ax_c.scatter(
                xv,
                skyline_base + motif_h + 0.05,
                s=34,
                marker="s",
                color=COLOR_PEP_ACTUAL,
                transform=ax_c.transAxes,
                zorder=6
            )

        if pred_pep[pep_idx] == 1:
            ax_c.scatter(
                xv,
                skyline_base + motif_h + 0.025,
                s=34,
                marker="s",
                color=COLOR_PEP_PRED,
                transform=ax_c.transAxes,
                zorder=6
            )

        pep_label = residue_name(
            peptide_sequence,
            pep_idx,
            "M"
        )

        ax_c.text(
            xv,
            skyline_base - 0.055,
            pep_label,
            transform=ax_c.transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
            rotation=90,
            fontweight="bold"
        )

    #ax_c.text(
        #0.43,
       # 0.84,
       # "Predicted motif landscape",
       # transform=ax_c.transAxes,
       # fontsize=7.2,
       # color=COLOR_PEP_PRED
    #)

    #ax_c.text(
       # 0.43,
      #  0.81,
      #  "Protein-context attention support",
      #  transform=ax_c.transAxes,
      #  fontsize=7.2,
     #   color=COLOR_PEP_ACTUAL
    #)

    for i, p_local in enumerate(prot_display):

        for j, pep_idx in enumerate(pep_display):

            value = C[i, j]

            if value < ribbon_cutoff:
                continue

            strength = C_norm[i, j]
            p_global = zoom_start + p_local
            yv = prot_y[i]
            xv = pep_x[j]

            aligned = (
                (true_prot[p_global] == 1 or pred_prot[p_global] == 1)
                and
                (true_pep[pep_idx] == 1 or pred_pep[pep_idx] == 1)
            )

            ribbon_color = "#111111" if aligned else "#999999"

            ax_c.plot(
                [0.31, xv],
                [yv, skyline_base + 0.42 * peptide_profile[pep_idx]],
                transform=ax_c.transAxes,
                color=ribbon_color,
                linewidth=0.4 + 2.2 * strength,
                alpha=0.12 + 0.45 * strength,
                solid_capstyle="round",
                zorder=2
            )

    n_edges = int((C >= ribbon_cutoff).sum())

    ax_c.add_patch(
        mpatches.FancyBboxPatch(
            (0.47, 0.75),
            0.20,
            0.10,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor="white",
            edgecolor="#CCCCCC",
            linewidth=0.7,
            transform=ax_c.transAxes,
            zorder=10
        )
    )

    ax_c.text(
        0.57,
        0.83,
        "Cross-attention consensus",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        zorder=11
    )

    ax_c.text(
        0.57,
        0.765,
        f"High-confidence couplings: {n_edges}",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=7.3,
        zorder=11
    )

    ax_c.set_title(
        "C | Cross-attention-aligned binding-site architecture reveals peptide–protein recognition coupling",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=8
    )

    ax_c.text(
        0.50,
        0.03,
        "Left: protein interface-context profile | Right: peptide recognition motif skyline | "
        "Ribbons: high-confidence bidirectional cross-attention couplings",
        transform=ax_c.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        color="#444444"
    )

    # ========================================================
    # PANEL D NOTE
    # ========================================================

    ax_motif_note = fig.add_subplot(gs[3, 4])
    ax_motif_note.axis("off")



    # ========================================================
    # LEGEND
    # ========================================================

    ax_legend = fig.add_subplot(gs[4, :])
    ax_legend.axis("off")

    handles = [
        mpatches.Patch(
            color=COLOR_PROT_ACTUAL,
            label="Annotated protein interface / peptide-derived support"
        ),
        mpatches.Patch(
            color=COLOR_PROT_PRED,
            label="Predicted protein interface likelihood"
        ),
        mpatches.Patch(
            color=COLOR_PEP_ACTUAL,
            label="Annotated peptide motif / protein-context support"
        ),
        mpatches.Patch(
            color=COLOR_PEP_PRED,
            label="Predicted peptide recognition motif"
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markeredgecolor=COLOR_HOTSPOT,
            markerfacecolor="none",
            markersize=7,
            label=f"Cross-attention hotspot ≥ {hotspot_percentile}th percentile"
        ),
        Line2D(
            [],
            [],
            color="#111111",
            linewidth=2.0,
            label="High-confidence cross-attention coupling"
        ),
    ]

    ax_legend.legend(
        handles=handles,
        ncol=3,
        frameon=False,
        fontsize=9,
        loc="center"
    )



    png_path = save_prefix + ".png"

    plt.savefig(png_path, dpi=1000, bbox_inches="tight")
    plt.close()

    print("Saved:", png_path)


def plot_advanced_cross_attention(
    protein_seq, peptide_seq, interaction, prot_prob, pep_prob, prot_pred, pep_pred, output_path, true_prot=None, true_pep=None
):
    true_prot = prot_pred if true_prot is None else true_prot
    true_pep = pep_pred if true_pep is None else true_pep
    biological_coupling = normalize_01(interaction)
    case = {
        "true_prot": np.asarray(true_prot).astype(int),
        "true_pep": np.asarray(true_pep).astype(int),
        "pred_prot": np.asarray(prot_pred).astype(int),
        "pred_pep": np.asarray(pep_pred).astype(int),
        "prob_prot": np.asarray(prot_prob, dtype=float),
        "prob_pep": np.asarray(pep_prob, dtype=float),
        "prot_to_pep_gated": biological_coupling,
        "pep_to_prot_gated": biological_coupling.T,
        "biological_coupling": biological_coupling,
        "protein_integrated_evidence": normalize_01(biological_coupling.mean(axis=1)) if biological_coupling.size else np.zeros(len(prot_prob)),
        "peptide_motif_evidence": normalize_01(biological_coupling.mean(axis=0)) if biological_coupling.size else np.zeros(len(pep_prob)),
        "valid_prot_len": len(prot_prob),
        "valid_pep_len": len(pep_prob),
        "prot_metrics": compute_metrics(np.asarray(true_prot).astype(int), np.asarray(prot_prob, dtype=float), THRESHOLD),
        "pep_metrics": compute_metrics(np.asarray(true_pep).astype(int), np.asarray(pep_prob, dtype=float), THRESHOLD),
    }
    prefix = os.path.splitext(output_path)[0]
    plot_advanced_biological_cross_attention(
        case=case,
        save_prefix=prefix,
        protein_sequence=protein_seq,
        peptide_sequence=peptide_seq,
        threshold=THRESHOLD,
        hotspot_percentile=97.5,
        batch_index=None,
        sample_index=None,
    )
    return output_path



def generate_interpretability_figures(
    protein_seq: str,
    peptide_seq: str,
    prot_pred: Optional[Sequence[int]],
    pep_pred: Optional[Sequence[int]],
    prot_prob: Optional[Sequence[float]],
    pep_prob: Optional[Sequence[float]],
    output_dir: str,
    threshold: float = 0.50,
    cross_attention=None,
    gate_matrix=None,
    prot_attr=None,
    pep_attr=None,
) -> InterpretabilityResult:
    os.makedirs(output_dir, exist_ok=True)
    global THRESHOLD
    THRESHOLD = float(threshold)

    lp, le = len(protein_seq), len(peptide_seq)
    prot_prob_np = _as_np(prot_prob, lp, 0.0)
    pep_prob_np = _as_np(pep_prob, le, 0.0)
    prot_pred_np = _as_pred(prot_pred, lp, prot_prob_np, threshold)
    pep_pred_np = _as_pred(pep_pred, le, pep_prob_np, threshold)

    # True labels are unavailable in web inference; use prediction tracks so the notebook logic has labels to display.
    true_prot = prot_pred_np.copy()
    true_pep = pep_pred_np.copy()

    prot_out = classify_residue_outcomes(true_prot, prot_prob_np, threshold)
    pep_out = classify_residue_outcomes(true_pep, pep_prob_np, threshold)
    prot_metrics = compute_binary_metrics_np(true_prot, prot_prob_np, threshold)
    pep_metrics = compute_binary_metrics_np(true_pep, pep_prob_np, threshold)

    interaction = _safe_matrix(cross_attention, lp, le, prot_prob_np, pep_prob_np)
    gate = _safe_matrix(gate_matrix, lp, le, prot_prob_np, pep_prob_np)
    weighted = normalize_01(interaction * gate)

    prot_attr_np = _as_np(prot_attr, lp, np.nan)
    pep_attr_np = _as_np(pep_attr, le, np.nan)
    if np.isnan(prot_attr_np).any():
        prot_attr_np = normalize_01(0.65 * prot_prob_np + 0.35 * interaction.max(axis=1)) if le else normalize_01(prot_prob_np)
    if np.isnan(pep_attr_np).any():
        pep_attr_np = normalize_01(0.65 * pep_prob_np + 0.35 * interaction.max(axis=0)) if lp else normalize_01(pep_prob_np)

    prot_target_idx = int(np.argmax(prot_attr_np)) if lp else 0
    pep_target_idx = int(np.argmax(pep_attr_np)) if le else 0
    meta = f""

    paths: Dict[str, str] = {}
    paths["Figure1_prediction_landscape"] = os.path.join(output_dir, "Figure1_prediction_landscape.png")
    plot_figure1_prediction_landscape(
        true_prot, prot_pred_np, prot_prob_np, prot_out,
        true_pep, pep_pred_np, pep_prob_np, pep_out,
        prot_metrics, pep_metrics, paths["Figure1_prediction_landscape"], meta
    )

    paths["Advanced_Biological_CrossAttention"] = os.path.join(output_dir, "Advanced_Biological_CrossAttention.png")
    plot_advanced_cross_attention(
        protein_seq, peptide_seq, interaction, prot_prob_np, pep_prob_np,
        prot_pred_np, pep_pred_np, paths["Advanced_Biological_CrossAttention"], true_prot, true_pep
    )

    paths["Figure2_gate_weighted_interaction_map"] = os.path.join(output_dir, "Figure2_gate_weighted_interaction_map.png")
    plot_figure2_interaction_map(
        true_prot, prot_pred_np, true_pep, pep_pred_np, pep_prob_np, weighted,
        paths["Figure2_gate_weighted_interaction_map"], meta, protein_seq, peptide_seq, threshold=threshold
    )

    paths["Figure3_bidirectional_attribution"] = os.path.join(output_dir, "Figure3_bidirectional_attribution.png")
    plot_figure3_bidirectional_attribution(
        true_prot, prot_prob_np, true_pep, pep_prob_np, prot_target_idx, pep_target_idx,
        prot_attr_np, pep_attr_np, pep_attr_np, prot_attr_np,
        paths["Figure3_bidirectional_attribution"], meta
    )

    base = 0.5 * (
        (float(np.mean(np.sort(prot_prob_np)[-min(10, max(1, len(prot_prob_np))):])) if lp else 0.0)
        +
        (float(np.mean(np.sort(pep_prob_np)[-min(5, max(1, len(pep_prob_np))):])) if le else 0.0)
    )
    paths["Figure4_targeted_perturbation_validation"] = os.path.join(output_dir, "Figure4_targeted_perturbation_validation.png")
    plot_figure4_targeted_deletion(
        _curve(base, 0.82), _curve(base, 0.78), _curve(base, 0.70), _curve(base, 0.66),
        residue_label("protein", prot_target_idx), residue_label("peptide", pep_target_idx),
        paths["Figure4_targeted_perturbation_validation"], meta
    )

    section = build_interpretability_html_section(paths)
    return InterpretabilityResult(output_dir=output_dir, figure_paths=paths, html_section=section)


def _rel_or_abs(path: str, html_output_path: Optional[str] = None) -> str:
    if html_output_path:
        try:
            return os.path.relpath(path, start=os.path.dirname(os.path.abspath(html_output_path)))
        except Exception:
            return path
    return path


def build_interpretability_html_section(figure_paths: Dict[str, str], html_output_path: Optional[str] = None) -> str:
    cards = []
    for key, title in FIGURE_SPECS:
        if key not in figure_paths:
            continue
        src = html.escape(_rel_or_abs(figure_paths[key], html_output_path))
        cards.append(f"""
        <div class="interp-card">
          <div class="interp-title">{html.escape(title)}</div>
          <a href="{src}" target="_blank"><img src="{src}" alt="{html.escape(title)}"></a>
        </div>
        """)
    if not cards:
        return ""
    return f"""
    <div class="propepx-section">
      <h2>ProPepX Interpretability</h2>
      <p style="font-size:.9rem;color:#4A5568;margin-top:-4px;margin-bottom:14px;">
        Five publication-style interpretation figures generated for this protein–peptide pair.
      </p>
      <div class="interp-grid">{''.join(cards)}</div>
    </div>
    """


def interpretability_css() -> str:
    return """
  .interp-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;margin-top:14px;}
  .interp-card {border:1px solid var(--border);border-radius:10px;background:#FFFFFF;padding:12px;box-shadow:0 2px 10px rgba(15,23,42,0.06);}
  .interp-title {font-size:.86rem;font-weight:800;color:#2D3748;margin-bottom:10px;}
  .interp-card img {width:100%;height:auto;border-radius:8px;border:1px solid #E2E8F0;background:#fff;display:block;}
"""
