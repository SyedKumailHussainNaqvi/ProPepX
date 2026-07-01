from Import_libraries import *
from Dataset_Preprocessing import *
from pepPI_binding_site_collate import *
from padding_mask_to_logits import *
from compute_loss import *
from Dataloader import *
from propepx_metrics import *


def Test_evaluate_model(model, test_loader, device, threshold=0.50):

    model.eval()

    prot_true_all, prot_prob_all = [], []
    pep_true_all, pep_prob_all = [], []

    with torch.no_grad():
        for prot_emb, pep_emb, prot_label, pep_label in tqdm(
            test_loader,
            desc="Evaluating"
        ):

            prot_emb = prot_emb.to(device)
            pep_emb = pep_emb.to(device)
            prot_label = prot_label.to(device)
            pep_label = pep_label.to(device)

            prot_mask = (prot_emb.abs().sum(dim=-1) == 0)
            pep_mask = (pep_emb.abs().sum(dim=-1) == 0)

            prot_logits, pep_logits = model(
                prot_emb,
                pep_emb,
                prot_mask=prot_mask,
                pep_mask=pep_mask
            )

            if prot_logits is not None:
                prot_prob = torch.softmax(prot_logits, dim=-1)[:, :, 1]
                prot_valid = prot_label != -100

                prot_true_all.append(
                    prot_label[prot_valid].detach().cpu().numpy()
                )
                prot_prob_all.append(
                    prot_prob[prot_valid].detach().cpu().numpy()
                )

            if pep_logits is not None:
                pep_prob = torch.softmax(pep_logits, dim=-1)[:, :, 1]
                pep_valid = pep_label != -100

                pep_true_all.append(
                    pep_label[pep_valid].detach().cpu().numpy()
                )
                pep_prob_all.append(
                    pep_prob[pep_valid].detach().cpu().numpy()
                )

    results = {}

    if len(prot_true_all) > 0:
        results["Protein_GLOBAL"] = compute_prot_metrics(
            y_true=np.concatenate(prot_true_all),
            y_prob=np.concatenate(prot_prob_all),
            threshold=threshold
        )

    if len(pep_true_all) > 0:
        results["Peptide_GLOBAL"] = compute_prot_metrics(
            y_true=np.concatenate(pep_true_all),
            y_prob=np.concatenate(pep_prob_all),
            threshold=threshold
        )

    return results


def print_results(results):
    print("\n================ FINAL TEST METRICS ================")

    if "Protein_GLOBAL" in results:
        print("\n=== Protein Binding Residue Prediction ===")
        for k, v in results["Protein_GLOBAL"].items():
            print(f"{k:22s}: {v:.4f}")

    if "Peptide_GLOBAL" in results:
        print("\n=== Peptide Binding Residue Prediction ===")
        for k, v in results["Peptide_GLOBAL"].items():
            print(f"{k:22s}: {v:.4f}")
