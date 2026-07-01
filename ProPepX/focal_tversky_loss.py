from Import_libraries import *


# ============================================================
# CROSS-ENTROPY LOSSES
# ============================================================

def protein_ce_loss_global(
    logits,
    labels,
    label_smoothing=0.05,
    ignore_index=-100,
    class_weights=None
):
    B, L, C = logits.shape

    return F.cross_entropy(
        logits.reshape(B * L, C),
        labels.reshape(B * L),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="mean",
        weight=class_weights
    )


def peptide_ce_loss_global(
    logits,
    labels,
    label_smoothing=0.10,
    ignore_index=-100,
    class_weights=None
):
    B, L, C = logits.shape

    return F.cross_entropy(
        logits.reshape(B * L, C),
        labels.reshape(B * L),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="mean",
        weight=class_weights
    )


def peptide_ce_loss_per_pair(
    logits,
    labels,
    label_smoothing=0.10,
    ignore_index=-100,
    class_weights=None
):
    B, L, C = logits.shape
    losses = []

    for b in range(B):
        valid = labels[b] != ignore_index
        if valid.sum() == 0:
            continue

        logits_b = logits[b][valid]
        labels_b = labels[b][valid]

        ce_b = F.cross_entropy(
            logits_b,
            labels_b,
            reduction="mean",
            label_smoothing=label_smoothing,
            weight=class_weights
        )
        losses.append(ce_b)

    if len(losses) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return torch.stack(losses).mean()


# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(
    logits,
    labels,
    gamma=2.0,
    ignore_index=-100,
    alpha=None,
    reduction="global"
):
    B, L, C = logits.shape

    if reduction == "global":
        valid = labels != ignore_index

        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_v = logits[valid]
        labels_v = labels[valid]

        log_probs = F.log_softmax(logits_v, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, labels_v.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, labels_v.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** gamma
        loss = -focal_weight * target_log_probs

        if alpha is not None:
            alpha = alpha.to(logits.device)
            loss = alpha[labels_v] * loss

        return loss.mean()

    elif reduction == "per_pair":
        losses = []

        for b in range(B):
            valid = labels[b] != ignore_index
            if valid.sum() == 0:
                continue

            logits_b = logits[b][valid]
            labels_b = labels[b][valid]

            log_probs = F.log_softmax(logits_b, dim=-1)
            probs = torch.exp(log_probs)

            target_log_probs = log_probs.gather(1, labels_b.unsqueeze(1)).squeeze(1)
            target_probs = probs.gather(1, labels_b.unsqueeze(1)).squeeze(1)

            focal_weight = (1.0 - target_probs) ** gamma
            loss_b = -focal_weight * target_log_probs

            if alpha is not None:
                alpha = alpha.to(logits.device)
                loss_b = alpha[labels_b] * loss_b

            losses.append(loss_b.mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return torch.stack(losses).mean()

    else:
        raise ValueError("reduction must be 'global' or 'per_pair'")


def focal_loss_per_pair(
    logits,
    labels,
    gamma=2.0,
    ignore_index=-100,
    alpha=None
):
    return focal_loss(
        logits=logits,
        labels=labels,
        gamma=gamma,
        ignore_index=ignore_index,
        alpha=alpha,
        reduction="per_pair"
    )


# ============================================================
# TVERSKY LOSS
# ============================================================

def tversky_loss(
    logits,
    labels,
    alpha=0.3,
    beta=0.7,
    ignore_index=-100,
    eps=1e-8,
    reduction="per_pair"
):
    probs = torch.softmax(logits, dim=-1)[..., 1]

    if reduction == "global":
        valid = labels != ignore_index

        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        p = probs[valid]
        y = labels[valid].float()

        TP = (p * y).sum()
        FP = (p * (1.0 - y)).sum()
        FN = ((1.0 - p) * y).sum()

        tv = TP / (TP + alpha * FP + beta * FN + eps)
        return 1.0 - tv

    elif reduction == "per_pair":
        losses = []

        for b in range(logits.size(0)):
            valid = labels[b] != ignore_index
            if valid.sum() == 0:
                continue

            p = probs[b][valid]
            y = labels[b][valid].float()

            TP = (p * y).sum()
            FP = (p * (1.0 - y)).sum()
            FN = ((1.0 - p) * y).sum()

            tv = TP / (TP + alpha * FP + beta * FN + eps)
            losses.append(1.0 - tv)

        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return torch.stack(losses).mean()

    else:
        raise ValueError("reduction must be 'global' or 'per_pair'")


def tversky_loss_per_pair(
    logits,
    labels,
    alpha=0.3,
    beta=0.7,
    ignore_index=-100,
    eps=1e-8
):
    return tversky_loss(
        logits=logits,
        labels=labels,
        alpha=alpha,
        beta=beta,
        ignore_index=ignore_index,
        eps=eps,
        reduction="per_pair"
    )