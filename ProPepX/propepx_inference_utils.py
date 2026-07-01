import sys

sys.path.append("/home/kumail/Transformer Model/ProPepX_complete_code/")


from Import_libraries import *
from Dataset_Preprocessing import H5PairDataset
from pepPI_binding_site_collate import collate_fn
from CoBindingCNN import ProPepX
from compute_loss import compute_loss
from model_load_weight_utils import load_weight
from propepx_metrics import (
    compute_prot_metrics,
    compute_pep_pair_metrics,
    summarize_pep_pair_metrics,compute_joint_mode_metrics,
)


def get_device(gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    emb_dim,
    mode,
    device,
    hidden_dim=512,
    heads=8,
    dropout=0.35,
    num_self_layers=2,
    ff_dim=512,
    max_len_prot=1418,
    max_len_pep=50,
):
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
    )
    return model.to(device)


def make_loader(test_h5, mode, batch_size, num_workers=4, pin_memory=True):
    dataset = H5PairDataset(test_h5, mode=mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataset, loader


@torch.no_grad()
def evaluate_protein(model, loader, device, threshold=0.50):
    model.eval()
    all_true, all_prob = [], []

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="Testing", leave=False):
        prot_emb = prot_emb.to(device)
        pep_emb = pep_emb.to(device)
        prot_label = prot_label.to(device)

        prot_mask = prot_emb.abs().sum(dim=-1) == 0
        pep_mask = pep_emb.abs().sum(dim=-1) == 0

        prot_logits, _ = model(
            prot_emb=prot_emb,
            pep_emb=pep_emb,
            prot_mask=prot_mask,
            pep_mask=pep_mask,
        )

        prob = torch.softmax(prot_logits, dim=-1)[..., 1]
        valid = prot_label != -100

        y_true = prot_label[valid].detach().cpu().numpy()
        y_prob = prob[valid].detach().cpu().numpy()

        if len(y_true) > 0:
            all_true.append(y_true)
            all_prob.append(y_prob)

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)

    metrics = compute_prot_metrics(y_true, y_prob, threshold=threshold)
    metrics["Num_Residues"] = len(y_true)
    metrics["Num_Positive"] = int(y_true.sum())
    metrics["Num_Negative"] = int(len(y_true) - y_true.sum())
    return metrics


@torch.no_grad()
def evaluate_peptide(model, loader, device, threshold=0.50):
    model.eval()
    per_sample_results = []
    sample_index = 0

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="Testing", leave=False):
        prot_emb = prot_emb.to(device)
        pep_emb = pep_emb.to(device)
        pep_label = pep_label.to(device)

        prot_mask = prot_emb.abs().sum(dim=-1) == 0
        pep_mask = pep_emb.abs().sum(dim=-1) == 0

        _, pep_logits = model(
            prot_emb=prot_emb,
            pep_emb=pep_emb,
            prot_mask=prot_mask,
            pep_mask=pep_mask,
        )

        pep_prob = torch.softmax(pep_logits, dim=-1)[..., 1]
        pep_pred = (pep_prob >= threshold).long()

        for i in range(pep_label.size(0)):
            true_i = pep_label[i]
            pred_i = pep_pred[i]

            valid_i = true_i != -100

            y_true = true_i[valid_i].detach().cpu().numpy()
            y_pred = pred_i[valid_i].detach().cpu().numpy()

            if len(y_true) == 0:
                continue

            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            metrics = compute_pep_pair_metrics(y_true, y_pred)

            per_sample_results.append({
                "sample_index": sample_index + 1,
                "peptide_length": len(y_true),
                "num_positive": int(np.sum(y_true)),
                "num_negative": int(len(y_true) - np.sum(y_true)),
                **metrics,
            })

            sample_index += 1

    summary, per_sample_df = summarize_pep_pair_metrics(per_sample_results)
    return summary, per_sample_df


@torch.no_grad()
def evaluate_joint(model, loader, device, mode="mode-GLOBAL", threshold=0.50):
    model.eval()

    total_loss = 0.0
    all_prot_true, all_prot_prob = [], []
    all_pep_true, all_pep_prob = [], []

    for prot_emb, pep_emb, prot_label, pep_label in tqdm(loader, desc="Testing", leave=False):
        prot_emb = prot_emb.to(device)
        pep_emb = pep_emb.to(device)
        prot_label = prot_label.to(device)
        pep_label = pep_label.to(device)

        prot_mask = prot_emb.abs().sum(dim=-1) == 0
        pep_mask = pep_emb.abs().sum(dim=-1) == 0

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

        prot_prob = torch.softmax(prot_logits, dim=-1)[..., 1]
        pep_prob = torch.softmax(pep_logits, dim=-1)[..., 1]

        all_prot_true.append(prot_label.detach().cpu().numpy().reshape(-1))
        all_prot_prob.append(prot_prob.detach().cpu().numpy().reshape(-1))

        all_pep_true.append(pep_label.detach().cpu().numpy().reshape(-1))
        all_pep_prob.append(pep_prob.detach().cpu().numpy().reshape(-1))

    prot_true = np.concatenate(all_prot_true)
    prot_prob = np.concatenate(all_prot_prob)

    pep_true = np.concatenate(all_pep_true)
    pep_prob = np.concatenate(all_pep_prob)

    prot_metrics = compute_joint_mode_metrics(prot_true, prot_prob, threshold)
    pep_metrics = compute_joint_mode_metrics(pep_true, pep_prob, threshold)

    avg_loss = total_loss / len(loader)

    mean_mcc = np.nanmean([prot_metrics["MCC"], pep_metrics["MCC"]])
    mean_aupr = np.nanmean([prot_metrics["AUPR"], pep_metrics["AUPR"]])
    mean_auroc = np.nanmean([prot_metrics["AUROC"], pep_metrics["AUROC"]])

    return avg_loss, prot_metrics, pep_metrics, mean_mcc, mean_aupr, mean_auroc


def mean_sd(df, col):
    return f"{df[col].mean():.4f} ± {df[col].std():.4f}"