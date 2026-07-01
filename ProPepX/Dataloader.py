from Import_libraries import *
from Dataset_Preprocessing import *

def build_dataloaders(
    base_dir,
    mode,
    train_batch_size=16,
    val_batch_size=4,
    test_batch_size=16,
    num_workers=4,
    pin_memory=True
):

    paths = {
        "train": f"{base_dir}/Train_Dataset.h5",
        "val":   f"{base_dir}/Val_Dataset.h5",
        "test":  f"{base_dir}/Test_Dataset.h5",
    }

    loaders = {}

    loaders["train"] = DataLoader(
        H5PairDataset(paths["train"], mode=mode),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    loaders["val"] = DataLoader(
        H5PairDataset(paths["val"], mode=mode),
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    loaders["test"] = DataLoader(
        H5PairDataset(paths["test"], mode=mode),
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # -------------------------
    # Logging
    # -------------------------
    print("Dataset summary")
    for split, loader in loaders.items():
        print(f"{split.upper():5s} → {len(loader.dataset):6d} samples | batch={loader.batch_size}")

    return loaders