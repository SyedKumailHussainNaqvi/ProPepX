from Import_libraries import *


# ============================================================
# PROTEIN MODE
# Global residue-level metrics
# Input:
#   y_true = all true residue labels concatenated
#   y_prob = all predicted binding probabilities concatenated
# ============================================================
def compute_prot_metrics(y_true, y_prob, threshold=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    try:
        aupr = average_precision_score(y_true, y_prob)
    except Exception:
        aupr = np.nan

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception:
        specificity = np.nan

    return {
        "AUC": auc,
        "AUPR": aupr,
        "MCC": mcc,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


# ============================================================
# PEPTIDE MODE
# Per-pair peptide metrics
# Input:
#   y_true = true binary peptide labels for one peptide
#   y_pred = predicted binary peptide labels for one peptide
# Logic:
#   AUC, MCC, Specificity use complement-label correction when
#   true/pred labels contain only one class.
# ============================================================
def compute_pep_pair_metrics(y_true, y_pred):
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    y_true_combined = list(y_true)
    y_pred_combined = list(y_pred)

    if len(set(y_true)) == 1 or len(set(y_pred)) == 1:
        y_true_complement = [1 - int(v) for v in y_true]
        y_pred_complement = [1 - int(v) for v in y_pred]

        y_true_combined.extend(y_true_complement)
        y_pred_combined.extend(y_pred_complement)

    try:
        auc = roc_auc_score(y_true_combined, y_pred_combined)
    except Exception:
        auc = np.nan

    try:
        mcc = matthews_corrcoef(y_true_combined, y_pred_combined)
    except Exception:
        mcc = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(
            y_true_combined,
            y_pred_combined,
            labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception:
        specificity = np.nan

    return {
        "AUC": auc,
        "MCC": mcc,
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


# ============================================================
# PEPTIDE MODE
# Mean of per-pair peptide metrics
# Input:
#   per_pair_results = list of dictionaries from compute_pep_pair_metrics()
# ============================================================
def summarize_pep_pair_metrics(per_pair_results):
    import pandas as pd

    df = pd.DataFrame(per_pair_results)

    return {
        "Mean_AUC": df["AUC"].mean(),
        "Mean_MCC": df["MCC"].mean(),
        "Mean_Specificity": df["Specificity"].mean(),
        "Mean_Precision": df["Precision"].mean(),
        "Mean_Recall": df["Recall"].mean(),
        "Mean_F1": df["F1"].mean(),
        "Mean_Accuracy": df["Accuracy"].mean(),
    }, df
    
def compute_joint_mode_metrics(y_true, y_prob, threshold=0.50):
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    valid = y_true != -100
    y_true = y_true[valid].astype(int)
    y_prob = y_prob[valid]

    y_pred = (y_prob >= threshold).astype(int)

    return {
        "MCC": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUROC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "AUPR": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "Positive": int(np.sum(y_true == 1)),
        "Negative": int(np.sum(y_true == 0)),
        "Total": int(len(y_true)),
    }