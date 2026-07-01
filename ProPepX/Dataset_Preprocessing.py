from Import_libraries import *
from pepPI_binding_site_collate import *
from torch.utils.data import Dataset, DataLoader

def seq_hash(seq):
    return hashlib.sha1(seq.encode()).hexdigest() 

class H5PairDataset(Dataset):
    def __init__(self, h5_path, mode="both"):
        self.h5 = h5py.File(h5_path, "r")

        self.prot_grp = self.h5["prot"]
        self.pep_grp  = self.h5["pep"]

        self.mode = mode.lower()
        if self.mode in ["mode-global", "mode-GLOBAL"]:
            self.mode = "both"

        self.has_prot_label = "prot_label" in self.h5
        self.has_pep_label  = "pep_label" in self.h5

        if self.mode == "prot":
            assert self.has_prot_label, "H5 missing protein labels!"

        if self.mode == "pep":
            assert self.has_pep_label, "H5 missing peptide labels!"

        self.keys = list(self.prot_grp.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        prot_emb = torch.tensor(self.prot_grp[key][...], dtype=torch.float)
        pep_emb  = torch.tensor(self.pep_grp[key][...], dtype=torch.float)

        if self.mode in ("prot", "both"):
            prot_label = torch.tensor(self.h5["prot_label"][key][...], dtype=torch.long)
            prot_label = prot_label[:prot_emb.size(0)]
        else:
            prot_label = None

        if self.mode in ("pep", "both"):
            pep_label = torch.tensor(self.h5["pep_label"][key][...], dtype=torch.long)
            pep_label = pep_label[:pep_emb.size(0)]
        else:
            pep_label = None

        return prot_emb, pep_emb, prot_label, pep_label