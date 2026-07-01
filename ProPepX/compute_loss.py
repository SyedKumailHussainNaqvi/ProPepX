from Import_libraries import *
from focal_tversky_loss import *

PRINTED_MODES = set()


def compute_loss(
    prot_logits,
    pep_logits,
    prot_labels,
    pep_labels,
    MODE="mode",
    alpha_prot=0.5,
    beta_pep=0.5,
    ignore_index=-100,

    prot_class_pos_weight=2.0,
    prot_label_smoothing=0.02,
    prot_focal_gamma=1.5,
    prot_tv_alpha=0.65,
    prot_tv_beta=0.35,
    lambda_prot_ce=0.55,
    lambda_prot_focal=0.20,
    lambda_prot_tv=0.25,

    gamma=2.0,
    pep_class_pos_weight=1.8,
    pep_label_smoothing=0.10,
    pep_focal_gamma=1.5,
    pep_tv_alpha=0.5,
    pep_tv_beta=0.5,
    lambda_pep_ce=0.60,
    lambda_pep_focal=0.15,
    lambda_pep_tv=0.25
):
    """
    MODE:
        "prot"        -> protein-only training
        "pep"         -> peptide-only training
        "both"        -> original joint training
        "mode-GLOBAL" -> global residue-pooled joint training
    """

    MODE = MODE.lower()

    if prot_logits is not None:
        device = prot_logits.device
    elif pep_logits is not None:
        device = pep_logits.device
    else:
        raise ValueError("Both prot_logits and pep_logits are None.")

    prot_class_weights = torch.tensor([1.0, prot_class_pos_weight], device=device)
    prot_focal_alpha = torch.tensor([1.0, prot_class_pos_weight], device=device)

    pep_class_weights = torch.tensor([1.0, pep_class_pos_weight], device=device)
    pep_focal_alpha = torch.tensor([1.0, pep_class_pos_weight], device=device)

    # ============================================================
    # PROTEIN-ONLY MODE
    # ============================================================
    if MODE == "prot":
        if "prot" not in PRINTED_MODES:
            print("Enter in prot mode")
            PRINTED_MODES.add("prot")

        prot_ce = protein_ce_loss_global(
            prot_logits,
            prot_labels,
            label_smoothing=prot_label_smoothing,
            ignore_index=ignore_index,
            class_weights=prot_class_weights
        )

        prot_focal = focal_loss(
            prot_logits,
            prot_labels,
            gamma=prot_focal_gamma,
            ignore_index=ignore_index,
            alpha=prot_focal_alpha,
            reduction="global"
        )

        prot_tv = tversky_loss(
            prot_logits,
            prot_labels,
            alpha=prot_tv_alpha,
            beta=prot_tv_beta,
            ignore_index=ignore_index,
            reduction="per_pair"
        )

        return (
            lambda_prot_ce * prot_ce +
            lambda_prot_focal * prot_focal +
            lambda_prot_tv * prot_tv
        )

    # ============================================================
    # PEPTIDE-ONLY MODE
    # ============================================================
    elif MODE == "pep":
        if "pep" not in PRINTED_MODES:
            print("Enter in pep mode")
            PRINTED_MODES.add("pep")

        pep_ce = peptide_ce_loss_per_pair(
            pep_logits,
            pep_labels,
            label_smoothing=pep_label_smoothing,
            ignore_index=ignore_index,
            class_weights=pep_class_weights
        )

        pep_focal = focal_loss(
            pep_logits,
            pep_labels,
            gamma=pep_focal_gamma,
            ignore_index=ignore_index,
            alpha=pep_focal_alpha,
            reduction="per_pair"
        )

        pep_tv = tversky_loss(
            pep_logits,
            pep_labels,
            alpha=pep_tv_alpha,
            beta=pep_tv_beta,
            ignore_index=ignore_index,
            reduction="per_pair"
        )

        return (
            lambda_pep_ce * pep_ce +
            lambda_pep_focal * pep_focal +
            lambda_pep_tv * pep_tv
        )

    # ============================================================
    # ORIGINAL BOTH MODE
    # ============================================================
    elif MODE == "both":
        if "both" not in PRINTED_MODES:
            print("Enter in both mode: original joint training")
            PRINTED_MODES.add("both")

        total_loss = 0.0

        if prot_logits is not None:
            prot_ce = protein_ce_loss_global(
                prot_logits,
                prot_labels,
                label_smoothing=prot_label_smoothing,
                ignore_index=ignore_index,
                class_weights=prot_class_weights
            )

            prot_focal = focal_loss(
                prot_logits,
                prot_labels,
                gamma=gamma,
                ignore_index=ignore_index,
                alpha=prot_focal_alpha,
                reduction="global"
            )

            prot_tv = tversky_loss(
                prot_logits,
                prot_labels,
                alpha=prot_tv_alpha,
                beta=prot_tv_beta,
                ignore_index=ignore_index,
                reduction="per_pair"
            )

            prot_loss = (
                lambda_prot_ce * prot_ce +
                lambda_prot_focal * prot_focal +
                lambda_prot_tv * prot_tv
            )

            total_loss += alpha_prot * prot_loss

        if pep_logits is not None:
            pep_ce = peptide_ce_loss_per_pair(
                pep_logits,
                pep_labels,
                label_smoothing=pep_label_smoothing,
                ignore_index=ignore_index,
                class_weights=pep_class_weights
            )

            pep_focal = focal_loss(
                pep_logits,
                pep_labels,
                gamma=pep_focal_gamma,
                ignore_index=ignore_index,
                alpha=pep_focal_alpha,
                reduction="per_pair"
            )

            pep_tv = tversky_loss(
                pep_logits,
                pep_labels,
                alpha=pep_tv_alpha,
                beta=pep_tv_beta,
                ignore_index=ignore_index,
                reduction="per_pair"
            )

            pep_loss = (
                lambda_pep_ce * pep_ce +
                lambda_pep_focal * pep_focal +
                lambda_pep_tv * pep_tv
            )

            total_loss += beta_pep * pep_loss

        return total_loss

    # ============================================================
    # NEW GLOBAL TRAINING MODE
    # ============================================================
    elif MODE == "mode-global":
        if "mode-global" not in PRINTED_MODES:
            print("Enter in mode-GLOBAL: GLOBAL residue-pooled joint training")
            PRINTED_MODES.add("mode-global")

        total_loss = 0.0

        if prot_logits is not None:
            prot_ce = protein_ce_loss_global(
                prot_logits,
                prot_labels,
                label_smoothing=prot_label_smoothing,
                ignore_index=ignore_index,
                class_weights=prot_class_weights
            )

            prot_focal = focal_loss(
                prot_logits,
                prot_labels,
                gamma=prot_focal_gamma,
                ignore_index=ignore_index,
                alpha=prot_focal_alpha,
                reduction="global"
            )

            prot_tv = tversky_loss(
                prot_logits,
                prot_labels,
                alpha=prot_tv_alpha,
                beta=prot_tv_beta,
                ignore_index=ignore_index,
                reduction="global"
            )

            prot_loss = (
                lambda_prot_ce * prot_ce +
                lambda_prot_focal * prot_focal +
                lambda_prot_tv * prot_tv
            )

            total_loss += alpha_prot * prot_loss

        if pep_logits is not None:
            pep_ce = peptide_ce_loss_global(
                pep_logits,
                pep_labels,
                label_smoothing=pep_label_smoothing,
                ignore_index=ignore_index,
                class_weights=pep_class_weights
            )

            pep_focal = focal_loss(
                pep_logits,
                pep_labels,
                gamma=pep_focal_gamma,
                ignore_index=ignore_index,
                alpha=pep_focal_alpha,
                reduction="global"
            )

            pep_tv = tversky_loss(
                pep_logits,
                pep_labels,
                alpha=pep_tv_alpha,
                beta=pep_tv_beta,
                ignore_index=ignore_index,
                reduction="global"
            )

            pep_loss = (
                lambda_pep_ce * pep_ce +
                lambda_pep_focal * pep_focal +
                lambda_pep_tv * pep_tv
            )

            total_loss += beta_pep * pep_loss

        return total_loss

    else:
        raise ValueError(f"Unknown MODE: {MODE}")