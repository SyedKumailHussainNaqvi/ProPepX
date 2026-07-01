"""
propepx_predict.py
===================
ProPepX Prediction Engine
--------------------------
Core inference module that:
    1. Accepts raw protein + peptide sequences (strings)
    2. Generates per-residue PLM embeddings (ProtTrans / ESM / both)
    3. Loads the appropriate ProPepX fine-tuned (or pre-trained) checkpoint
    4. Runs binding-site prediction for the selected mode
    5. Returns structured results and renders visualisations

Supported ProPepX Modes
------------------------
    "prot"        – protein binding-site prediction only
    "pep"         – peptide binding-site prediction only
    "mode-GLOBAL" – joint protein–peptide prediction (best checkpoint)
    "zero-shot"   – pre-trained weights only, no fine-tuning

Supported Embedding Models
---------------------------
    "prottrans" – ProtTrans T5-XL-UniRef50  (1024-dim)
    "esm"       – ESM-2 600M               (1280-dim)
    "both"      – run both independently and return both results

Supported Fine-Tune Datasets
------------------------------
    "pepbdb"    – pepBDB
    "peppi"     – pepPI
    "benchmark" – combined benchmark

Usage (as a library)
---------------------
    from propepx_predict import ProPepXPredictor

    predictor = ProPepXPredictor(gpu_id=0)
    results = predictor.predict(
        protein_seq   = "MKTIIALSYIFCLVFA...",
        peptide_seq   = "ACDEFGHIKLM",
        embedding_model = "prottrans",
        propepx_mode  = "prot",
        dataset       = "peppi",
        render        = True,
    )

Usage (CLI)
-----------
    python propepx_predict.py \\
        --protein  "MKTIIALSYIFCLVFA..." \\
        --peptide  "ACDEFGHIKLM" \\
        --embedding prottrans \\
        --mode      prot \\
        --dataset   peppi \\
        --gpu_id    0 \\
        --save_html ./results/

Authors: Syed Kumail Hussain Naqvi et al.
"""

# ─────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────
import argparse
import os
import sys

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────
# ProPepX package
# ─────────────────────────────────────────────────────────────
sys.path.append("/home/kumail/Transformer Model/ProPepX_complete_code/")

from CoBindingCNN  import ProPepX
from model_load_weight_utils import load_weight
from propepx_config import (
    ARCH, EMB_DIM, BINDING_THRESHOLD,
    FINETUNE_WEIGHT_REGISTRY, PRETRAIN_WEIGHT_REGISTRY,
    DATASET_DESCRIPTIONS, MODE_DESCRIPTIONS, EMBEDDING_DESCRIPTIONS,
    resolve_weight,
)
from propepx_embedding    import embed_sequences
from propepx_visualizer   import render_terminal, render_html
from propepx_visualizer import render_terminal, render_html
from propepx_structure_viewer import *
from propepx_interpretability import generate_interpretability_figures
from propepx_visualizer_interpretability import render_interpretability_html

# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _build_model(mode: str, emb_dim: int, device: torch.device) -> ProPepX:
    model = ProPepX(
        emb_dim       = emb_dim,
        hidden_dim    = ARCH["hidden_dim"],
        heads         = ARCH["heads"],
        dropout       = ARCH["dropout"],
        mode          = mode,
        num_self_layers = ARCH["num_self_layers"],
        ff_dim        = ARCH["ff_dim"],
        max_len_prot  = ARCH["max_len_prot"],
        max_len_pep   = ARCH["max_len_pep"],
    )
    return model.to(device)


def _resolve_checkpoint(
    propepx_mode: str,
    dataset: str,
    embedding_model: str,
) -> str:

    if propepx_mode.startswith("zero-shot"):
        parts = propepx_mode.split("/")
        key_mode = parts[1] if len(parts) > 1 else "mode-GLOBAL"

        repo_path = PRETRAIN_WEIGHT_REGISTRY[key_mode][embedding_model]

    else:
        key_mode = propepx_mode

        repo_path = FINETUNE_WEIGHT_REGISTRY[key_mode][dataset][embedding_model]

    return resolve_weight(repo_path, verbose=True)


def _embedding_to_tensor(emb: np.ndarray, device: torch.device) -> torch.Tensor:
    """(L, D) numpy → (1, L, D) float32 tensor on *device*."""
    return torch.from_numpy(emb).float().unsqueeze(0).to(device)


@torch.no_grad()
def _run_inference(
    model: ProPepX,
    prot_emb: np.ndarray,
    pep_emb: np.ndarray,
    device: torch.device,
    threshold: float = BINDING_THRESHOLD,
) -> dict:
    """
    Forward pass → per-residue binding probability and binary prediction.

    Returns
    -------
    dict with keys:
        "prot_prob"  : np.ndarray (L_prot,)  | None
        "prot_pred"  : list[int] (0/1)       | None
        "pep_prob"   : np.ndarray (L_pep,)   | None
        "pep_pred"   : list[int] (0/1)       | None
    """
    model.eval()

    prot_t = _embedding_to_tensor(prot_emb, device)   # (1, Lp, D)
    pep_t  = _embedding_to_tensor(pep_emb,  device)   # (1, Le, D)

    prot_mask = prot_t.abs().sum(dim=-1) == 0          # (1, Lp)
    pep_mask  = pep_t.abs().sum(dim=-1)  == 0          # (1, Le)

    prot_logits, pep_logits = model(
        prot_emb  = prot_t,
        pep_emb   = pep_t,
        prot_mask = prot_mask,
        pep_mask  = pep_mask,
    )

    result = {"prot_prob": None, "prot_pred": None,
              "pep_prob":  None, "pep_pred":  None}

    if prot_logits is not None:
        prob = torch.softmax(prot_logits, dim=-1)[0, :len(prot_emb), 1]
        prob_np = prob.cpu().numpy()
        result["prot_prob"] = prob_np
        result["prot_pred"] = (prob_np >= threshold).astype(int).tolist()

    if pep_logits is not None:
        prob = torch.softmax(pep_logits, dim=-1)[0, :len(pep_emb), 1]
        prob_np = prob.cpu().numpy()
        result["pep_prob"] = prob_np
        result["pep_pred"] = (prob_np >= threshold).astype(int).tolist()

    return result


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

class ProPepXPredictor:
    """
    High-level ProPepX inference interface.

    Example
    -------
    >>> predictor = ProPepXPredictor(gpu_id=0)
    >>> result = predictor.predict(
    ...     protein_seq     = "MKTII...",
    ...     peptide_seq     = "ACDE..",
    ...     embedding_model = "prottrans",
    ...     propepx_mode    = "prot",
    ...     dataset         = "peppi",
    ...     render          = True,
    ... )
    >>> print(result["prot_pred"])   # list of 0/1 per residue
    """

    def __init__(self, gpu_id: int = 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n  ProPepX Predictor initialised  |  device = {self.device}\n")

    # ──────────────────────────────────────────────────────────

    def predict(
        self,
        protein_seq:     str,
        peptide_seq:     str,
        embedding_model: str = "prottrans",   # "prottrans" | "esm" | "both"
        propepx_mode:    str = "prot",        # "prot" | "pep" | "mode-GLOBAL" | "zero-shot"
        dataset:         str = "peppi",       # "pepbdb" | "peppi" | "benchmark"
        render:          bool = True,
        save_html:       str = None,          # dir path or None
    ) -> dict:
        """
        End-to-end prediction for a single protein–peptide pair.

        Parameters
        ----------
        protein_seq     : str   Raw amino-acid sequence (≤ 1418 residues)
        peptide_seq     : str   Raw amino-acid sequence (≤ 50 residues)
        embedding_model : str   "prottrans" | "esm" | "both"
        propepx_mode    : str   "prot" | "pep" | "mode-GLOBAL" | "zero-shot"
        dataset         : str   Fine-tune dataset key
        render          : bool  Print coloured output to terminal
        save_html       : str   Directory to write HTML report (None = skip)

        Returns
        -------
        dict
            When *embedding_model* is "prottrans" or "esm":
                {
                  "prot_prob":  np.ndarray | None,
                  "prot_pred":  list[int]  | None,   # 0 = non-binding, 1 = binding
                  "pep_prob":   np.ndarray | None,
                  "pep_pred":   list[int]  | None,
                  "embedding":  str,
                  "mode":       str,
                  "dataset":    str,
                }
            When *embedding_model* is "both":
                {
                  "prottrans": <above dict>,
                  "esm":       <above dict>,
                }
        """
        emb_model = embedding_model.lower().strip()
        mode      = propepx_mode.lower().strip()

        # Determine actual ProPepX arch mode (strip zero-shot prefix)
        emb_keys = ["prottrans", "esm"] if emb_model == "both" else [emb_model]

        # zero-shot uses mode-GLOBAL architecture internally
        if mode.lower().startswith("zero-shot"):
            arch_mode = "mode-GLOBAL"
        else:
            arch_mode = mode

        # Generate embeddings (may load both PLMs)
        print("\n" + "─" * 60)
        print("  [1/3]  Generating embeddings …")
        emb_dict = embed_sequences(
            protein_seq     = protein_seq,
            peptide_seq     = peptide_seq,
            embedding_model = emb_model,
            device          = self.device,
        )

        all_results = {}

        for ek in emb_keys:
            print(f"\n  [2/3]  Running ProPepX  "
                  f"[embedding={ek.upper()}  mode={arch_mode}  dataset={dataset}] …")

            prot_emb_np, pep_emb_np = emb_dict[ek]
            emb_dim = EMB_DIM[ek]

            # Build & load model
            model = _build_model(arch_mode, emb_dim, self.device)
            ckpt  = _resolve_checkpoint(mode, dataset, ek)
            model = load_weight(model, ckpt, self.device, strict=False, verbose=True)

            # Inference
            infer = _run_inference(model, prot_emb_np, pep_emb_np, self.device)

            result = {
                **infer,
                "embedding": ek,
                "mode":      arch_mode,
                "dataset":   dataset,
            }
            all_results[ek] = result

            # Free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  [3/3]  Rendering output …")

            if render:
                render_terminal(
                    protein_seq     = protein_seq,
                    peptide_seq     = peptide_seq,
                    prot_pred       = infer["prot_pred"],
                    pep_pred        = infer["pep_pred"],
                    mode            = arch_mode,
                    embedding_model = ek,
                    dataset         = dataset,
                    metrics         = None,
                )

            if save_html:
                os.makedirs(save_html, exist_ok=True)
                html_path = os.path.join(
                    save_html,
                    f"propepx_{arch_mode}_{ek}_{dataset}.html"
                )
                render_html(
                    protein_seq     = protein_seq,
                    peptide_seq     = peptide_seq,
                    prot_pred       = infer["prot_pred"],
                    pep_pred        = infer["pep_pred"],
                    prot_prob       = infer["prot_prob"],
                    pep_prob        = infer["pep_prob"],
                    mode            = arch_mode,
                    embedding_model = ek,
                    dataset         = dataset,
                    metrics         = None,
                    output_path     = html_path,
                )
                                # ======================================================
                # 3D STRUCTURE VIEWER
                # Uses real ProPepX predictions, NOT demo predictions
                # ======================================================

                pdb_path = os.path.join(
                    save_html,
                    "fold_2026_06_28_23_47_model_0.cif"
                )

                structure_dir = os.path.join(
                    save_html,
                    f"structure_{arch_mode}_{ek}_{dataset}"
                )

                os.makedirs(structure_dir, exist_ok=True)

                structure_html_path = os.path.join(
                    structure_dir,
                    "propepx_3d_viewer.html"
                )

                if os.path.exists(pdb_path):
                    generate_html_3d_viewer(
                        pdb_path        = pdb_path,
                        protein_seq     = protein_seq,
                        peptide_seq     = peptide_seq,
                        prot_pred       = infer["prot_pred"],
                        pep_pred        = infer["pep_pred"],
                        prot_prob       = infer["prot_prob"],
                        pep_prob        = infer["pep_prob"],
                        prot_chain_id   = "A",
                        pep_chain_id    = "B",
                        output_html     = structure_html_path,
                        mode            = arch_mode,
                        embedding_model = ek,
                        dataset         = dataset,
                    )

                    print(f"   3D viewer saved → {structure_html_path}")

                else:
                    print(f"   3D viewer skipped. CIF/PDB file not found:")
                    print(f"      {pdb_path}")
                
                interp_dir = os.path.join(
                    save_html,
                    f"interpretability_{arch_mode}_{ek}_{dataset}"
                )

                os.makedirs(interp_dir, exist_ok=True)

                interp = generate_interpretability_figures(
                    protein_seq = protein_seq,
                    peptide_seq = peptide_seq,
                    prot_pred   = infer["prot_pred"],
                    pep_pred    = infer["pep_pred"],
                    prot_prob   = infer["prot_prob"],
                    pep_prob    = infer["pep_prob"],
                    output_dir  = interp_dir,
                    threshold   = BINDING_THRESHOLD,
                )

                interp_html_path = os.path.join(
                    interp_dir,
                    "propepx_interpretability.html"
                )

                render_interpretability_html(
                    figure_paths    = interp.figure_paths,
                    output_path     = interp_html_path,
                    protein_seq     = protein_seq,
                    peptide_seq     = peptide_seq,
                    mode            = arch_mode,
                    embedding_model = ek,
                    dataset         = dataset,
                )

                print(f"   Interpretability report saved → {interp_html_path}")
                
                

        # Return flat dict for single embedding, nested for "both"
        if emb_model == "both":
            return all_results
        return all_results[emb_model]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _print_available_options() -> None:
    from propepx_config import (
        DATASET_DESCRIPTIONS, MODE_DESCRIPTIONS, EMBEDDING_DESCRIPTIONS,
        FINETUNE_WEIGHT_REGISTRY,
    )
    print("\n  ═══════════════════════════════════════")
    print("    ProPepX  ─  Available Options")
    print("  ═══════════════════════════════════════")
    print("\n  MODES:")
    for k, v in MODE_DESCRIPTIONS.items():
        print(f"    {k:<16} {v}")
    print("\n  EMBEDDING MODELS:")
    for k, v in EMBEDDING_DESCRIPTIONS.items():
        print(f"    {k:<16} {v}")
    print("\n  FINE-TUNE DATASETS:")
    for k, v in DATASET_DESCRIPTIONS.items():
        print(f"    {k:<16} {v}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProPepX – Protein–Peptide Binding Site Prediction",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--protein", required=True,
        help="Protein amino-acid sequence (max 1418 residues, 20 canonical AA)."
    )
    parser.add_argument(
        "--peptide", required=True,
        help="Peptide amino-acid sequence (max 50 residues, 20 canonical AA)."
    )
    parser.add_argument(
        "--embedding", default="prottrans",
        choices=["prottrans", "esm", "both"],
        help=(
            "Protein language model for embedding generation.\n"
            "  prottrans  ProtTrans T5-XL-UniRef50 (1024-dim)\n"
            "  esm        ESM-2 600M               (1280-dim)\n"
            "  both       Run both models independently\n"
            "(default: prottrans)"
        ),
    )
    parser.add_argument(
        "--mode", default="mode-GLOBAL",
        choices=["prot", "pep", "mode-GLOBAL", "zero-shot"],
        help=(
            "ProPepX prediction mode.\n"
            "  prot        Protein binding-site prediction\n"
            "  pep         Peptide binding-site prediction\n"
            "  mode-GLOBAL Joint protein–peptide prediction\n"
            "  zero-shot   Pre-trained weights only (no fine-tuning)\n"
            "(default: prot)"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="leads_ts251",
        choices=[
            # Joint mode
            "leads_ts251",
            "test167",
            
            # Joint mode
            "test167_zs", 

            # Protein mode
            "ts092",
            "ts125",
            "ts251",
            "ts639",

            # Peptide mode
            "camp_test231",
        ],
        help=(
            "Fine-tune dataset to load weights from.\n"
            "  Joint mode     leads_ts251 or test167 \n"
            "  rotein mode      ts092  or ts125 or ts251 or ts639\n"
            "  Peptide mode  camp_test231\n"
            "(default: leads_ts251)"
        ),
    )
    parser.add_argument(
        "--gpu_id", default="0",
        help="CUDA device ID (default: 0)."
    )
    parser.add_argument(
        "--save_html", default=None,
        help="Directory to save HTML report(s). Skipped if not provided."
    )
    parser.add_argument(
        "--list_options", action="store_true",
        help="Print all available modes, datasets, and embedding models, then exit."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list_options:
        _print_available_options()
        sys.exit(0)

    predictor = ProPepXPredictor(gpu_id=int(args.gpu_id))

    result = predictor.predict(
        protein_seq     = args.protein,
        peptide_seq     = args.peptide,
        embedding_model = args.embedding,
        propepx_mode    = args.mode,
        dataset         = args.dataset,
        render          = True,
        save_html       = args.save_html,
    )

    # Final summary to stdout
    if args.embedding != "both":
        pp = result.get("prot_pred")
        ep = result.get("pep_pred")
        if pp:
            n_bind = sum(pp)
            print(f"  Protein: {n_bind}/{len(pp)} binding residues predicted.")
        if ep:
            n_bind = sum(ep)
            print(f"  Peptide: {n_bind}/{len(ep)} binding residues predicted.")
    else:
        for emb, r in result.items():
            pp = r.get("prot_pred")
            ep = r.get("pep_pred")
            print(f"  [{emb.upper()}]")
            if pp:
                print(f"    Protein: {sum(pp)}/{len(pp)} binding residues predicted.")
            if ep:
                print(f"    Peptide: {sum(ep)}/{len(ep)} binding residues predicted.")
