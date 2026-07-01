from Import_libraries import *

def collate_fn(batch):
    prot_embs, pep_embs, prot_labels, pep_labels = zip(*batch)

    D = prot_embs[0].size(1)

    Lp = max(p.size(0) for p in prot_embs)
    Lq = max(p.size(0) for p in pep_embs)

    def pad_emb(list_e, L):
        out = torch.zeros(len(list_e), L, D)
        for i, x in enumerate(list_e):
            out[i, :x.size(0)] = x
        return out

    def pad_label(list_l, L):
        out = torch.full((len(list_l), L), -100, dtype=torch.long)
        for i, x in enumerate(list_l):
            if x is not None:
                out[i, :len(x)] = x
        return out

    prot_batch = pad_emb(prot_embs, Lp)
    pep_batch  = pad_emb(pep_embs, Lq)

    prot_lab_b = pad_label(prot_labels, Lp)
    pep_lab_b  = pad_label(pep_labels, Lq)

    return prot_batch, pep_batch, prot_lab_b, pep_lab_b