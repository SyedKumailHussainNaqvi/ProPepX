"""
propepx_config.py
==================
ProPepX Model Configuration & HuggingFace Checkpoint Registry
--------------------------------------------------------------
Design principles
-----------------
  LAZY     -- weights and embeddings are only downloaded when first used.
  IDEMPOTENT -- if cached locally, download is silently skipped every time.
  SPLIT CACHES -- weights and embeddings use separate local directories.

Install:  pip install huggingface_hub

Authors: Syed Kumail Hussain Naqvi et al.
"""

import os
from huggingface_hub import hf_hub_download, try_to_load_from_cache

HF_REPO_ID        = "syedkumailhussain/ProPepX"
HF_WEIGHTS_CACHE  = os.path.expanduser("~/.cache/propepx/weights")
HF_EMBEDDINGS_CACHE = os.path.expanduser("~/.cache/propepx/embeddings")

os.makedirs(HF_WEIGHTS_CACHE,    exist_ok=True)
os.makedirs(HF_EMBEDDINGS_CACHE, exist_ok=True)


def _is_cached(repo_path, cache_dir):
    local = try_to_load_from_cache(
        repo_id=HF_REPO_ID, filename=repo_path,
        repo_type="model",  cache_dir=cache_dir,
    )
    return local is not None and os.path.isfile(local)


def resolve_weight(repo_path, verbose=True):
    """
    Return local path to a ProPepX weight file.
    Downloads from HuggingFace only if not already cached.
    """
    cached = _is_cached(repo_path, HF_WEIGHTS_CACHE)
    if verbose and not cached:
        print(f"  DOWNLOAD weight : {repo_path}")
    elif verbose and cached:
        print(f"  CACHED   weight : {os.path.basename(repo_path)}")
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=repo_path,
        repo_type="model",  cache_dir=HF_WEIGHTS_CACHE,
    )


def resolve_embedding(repo_path, verbose=True):
    """
    Return local path to a test-embedding HDF5 file.
    Downloads from HuggingFace only if not already cached.
    Embeddings are ALWAYS re-fetched if cache is absent (no skip).
    """
    cached = _is_cached(repo_path, HF_EMBEDDINGS_CACHE)
    if verbose and not cached:
        print(f"  DOWNLOAD embedding : {repo_path}")
    elif verbose and cached:
        print(f"  CACHED   embedding : {os.path.basename(repo_path)}")
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=repo_path,
        repo_type="model",  cache_dir=HF_EMBEDDINGS_CACHE,
    )


# Architecture (must match all saved checkpoints)
ARCH = dict(
    hidden_dim=512, heads=8, dropout=0.35,
    num_self_layers=2, ff_dim=512,
    max_len_prot=1418, max_len_pep=50,
)

EMB_DIM = {"prottrans": 1024, "esm": 1152}

BINDING_THRESHOLD = 0.50


# Fine-Tuned Weight Registry  (LAZY -- strings, resolved by resolve_weight())
FINETUNE_WEIGHT_REGISTRY = {
    "mode-GLOBAL": {
        "leads_ts251": {
            "esm":       "Joint-ProPep/ESM-3(600M)/Fine_tine_LEADs_TS251_ProPepX_weight/fold_5/best_fold_5_joint_mode_ProPepX.pt",
            "prottrans": "Joint-ProPep/ProttransT5/LEADs_TS251_ProPepX_weight/fold_5/best_fold_5_joint_mode_ProPepX.pt",
        },
        "test167": {
            "esm":       "Joint-ProPep/ESM-3(600M)/Fine_tine_Test167_ProPepX_weight/fold_5/best_fold_5_joint_mode_ProPepX.pt",
            "prottrans": "Joint-ProPep/ProttransT5/Test167_ProPepX_weight/fold_5/best_fold_5_joint_mode_ProPepX.pt",
        },
    },
    "prot": {
        "ts092": {
            "esm":       "Protein-side/ESM-3(600M)/Fine_Tune_TS092_weight/Fine_tine_TS092_ProPepX_weight.pt",
            "prottrans": "Protein-side/ProttransT5/Fine_Tune_TS092_weight/Fine_tine_TS092_ProPepX_weight.pt",
        },
        "ts125": {
            "esm":       "Protein-side/ESM-3(600M)/Fine_Tune_TS125_weight/ESM-3_Test_Dataset_TS125_embedding.pt",
            "prottrans": "Protein-side/ProttransT5/Fine_Tune_TS125_weight/Fine_tine_TS125_ProPepX_weight.pt",
        },
        "ts251": {
            "esm":       "Protein-side/ESM-3(600M)/Fine_Tune_TS251_weight/ESM-3_Test_Dataset_TS251_embedding.pt",
            "prottrans": "Protein-side/ProttransT5/Fine_Tune_TS251_weight/Fine_tine_TS251_ProPepX_weight.pt",
        },
        "ts639": {
            "esm":       "Protein-side/ESM-3(600M)/Fine_Tune_TS639_weight/ESM-3_Test_Dataset_TS639_embedding.pt",
            "prottrans": "Protein-side/ProttransT5/Fine_Tune_TS639_weight/Fine_tine_TS639_ProPepX_weight.pt",
        },
    },
    "pep": {
        "camp_test231": {
            "esm":       "Peptide-side/ESM-3(600M)/best_finetune_pep_mode_ProPepX_weight.pt",
            "prottrans": "Peptide-side/ProttransT5/best_finetune_pep_mode_ProPepX_weight.pt",
        },
    },
}
FINETUNE_WEIGHT_REGISTRY["mode-global"] = FINETUNE_WEIGHT_REGISTRY["mode-GLOBAL"]


# Zero-Shot Weight Registry
PRETRAIN_WEIGHT_REGISTRY = {
    "zero-shot": {
        "esm":       "Zero-shot/ESM-3(600M)/Zero_shot_ProPepX_weight/best_pretrain_zero_shot_mode_ProPepX.pt",
        "prottrans": "Zero-shot/ProttransT5/Zero_shot_ProPepX_weight/best_pretrain_zero_shot_mode_ProPepX.pt",
    },
    "mode-GLOBAL": {
        "esm":       "Zero-shot/ESM-3(600M)/Zero_shot_ProPepX_weight/best_pretrain_zero_shot_mode_ProPepX.pt",
        "prottrans": "Zero-shot/ProttransT5/Zero_shot_ProPepX_weight/best_pretrain_zero_shot_mode_ProPepX.pt",
    },
}
PRETRAIN_WEIGHT_REGISTRY["mode-global"] = PRETRAIN_WEIGHT_REGISTRY["mode-GLOBAL"]


# Test-Embedding Registry  (LAZY -- strings, resolved by resolve_embedding())
TEST_EMBEDDING_REGISTRY = {
    "mode-GLOBAL": {
        "leads_ts251": {
            "esm":       "Joint-ProPep/ESM-3(600M)/Test_Data_LEADs_TS251/Test251_LEADSPEP_ESM-600M_embeddings.h5",
            "prottrans": "Joint-ProPep/ProttransT5/Test_Data_LEADs_TS251/Test251_LEADSPEP_embedding.h5",
        },
        "test167": {
            "esm":       "Joint-ProPep/ESM-3(600M)/Test_Data_TS167/Test167_ESM-600M_embeddings.h5",
            "prottrans": "Joint-ProPep/ProttransT5/Test_Data_TS167/Test167_embeddings.h5",
        },
    },
    "prot": {
        "ts092": {
            "esm":       "Protein-side/ESM-3(600M)/Test_Data_TS092/ESM-3_Test_Dataset_TS092_embedding.h5",
            "prottrans": "Protein-side/ProttransT5/Test_Data_TS092/Test_Dataset_TS092_embedding.h5",
        },
        "ts125": {
            "esm":       "Protein-side/ESM-3(600M)/Test_Data_TS125/ESM-3_Test_Dataset_TS125_embedding.h5",
            "prottrans": "Protein-side/ProttransT5/Test_Data_TS125/Test_Dataset_TS125_embedding.h5",
        },
        "ts251": {
            "esm":       "Protein-side/ESM-3(600M)/Test_Data_TS251/ESM-3_Test_Dataset_TS251_embedding.h5",
            "prottrans": "Protein-side/ProttransT5/Test_Data_TS251/Test_Dataset_TS251_embedding.h5",
        },
        "ts639": {
            "esm":       "Protein-side/ESM-3(600M)/Test_Data_TS639/ESM-3_Test_Dataset_TS639_embedding.h5",
            "prottrans": "Protein-side/ProttransT5/Test_Data_TS639/Test_Dataset_TS639_embedding.h5",
        },
    },
    "pep": {
        "camp_test231": {
            "esm":       "Peptide-side/ESM-3(600M)/Test_Data_TS231/CAMP_Test_ESM-600_embeddings.h5",
            "prottrans": "Peptide-side/ProttransT5/Test_Data_TS231/CAMP_Test_ProttranT5_embeddings.h5",
        },
    },
    "zero-shot": {
        "test167_zs": {
            "esm":       "Zero-shot/ESM-3(600M)/Test_Data_TS167_ZS/Test_Dataset_TS167_ESM-3_embedding.h5",
            "prottrans": "Zero-shot/ProttransT5/Test_Data_TS167_ZS/Test_Dataset_TS167_ProttransT5_embedding.h5",
        },
    },
}
TEST_EMBEDDING_REGISTRY["mode-global"] = TEST_EMBEDDING_REGISTRY["mode-GLOBAL"]


DATASET_DESCRIPTIONS = {
    "leads_ts251":  "Joint-mode  LEADS TS251",
    "test167":      "Joint-mode  Test167",
    "ts092":        "Protein-side  TS092",
    "ts125":        "Protein-side  TS125",
    "ts251":        "Protein-side  TS251",
    "ts639":        "Protein-side  TS639",
    "camp_test231": "Peptide-side  CAMP Test231",
    "test167_zs":   "Zero-shot  TS167",
}
MODE_DESCRIPTIONS = {
    "prot":        "Protein binding-site prediction",
    "pep":         "Peptide binding-site prediction",
    "mode-GLOBAL": "Joint protein-peptide binding-site prediction",
    "mode-global": "Joint protein-peptide binding-site prediction",
    "zero-shot":   "Zero-shot (pre-trained only)",
}
EMBEDDING_DESCRIPTIONS = {
    "prottrans": "ProtTrans T5-XL-UniRef50  (1024-dim)",
    "esm":       "ESM-3 / ESM-2 600M        (1152-dim)",
    "both":      "Both ProtTrans and ESM",
}
