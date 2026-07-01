"""
model_utils.py

Utility functions for loading ProPepX pretrained and fine-tuned models.

Author: Syed Kumail Hussain Naqvi
"""

from Import_libraries import *


def load_weight(
    model,
    weight_path,
    device,
    strict=False,
    verbose=True,
    resize_positional_embeddings=True,
):
    """
    Load pretrained or fine-tuned ProPepX checkpoint.

    Supports checkpoints saved as:
        - {"model_state": ...}
        - {"model_state_dict": ...}
        - {"state_dict": ...}
        - raw state_dict

    Optionally resizes learnable positional embeddings when max sequence
    length differs between the checkpoint and current model.
    """

    checkpoint = torch.load(
        weight_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if resize_positional_embeddings:
        model_state = model.state_dict()

        for key in ["prot_pos.pos_embed", "pep_pos.pos_embed"]:
            if key in state_dict and key in model_state:
                old = state_dict[key]
                new = model_state[key].clone()

                if old.shape != new.shape:
                    copy_len = min(old.size(1), new.size(1))
                    new[:, :copy_len, :] = old[:, :copy_len, :]
                    state_dict[key] = new

                    if verbose:
                        print(f"Resized {key}: {tuple(old.shape)} -> {tuple(new.shape)}")

    msg = model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print("=" * 70)
        print("Loaded checkpoint:", weight_path)
        print("Missing keys   :", len(msg.missing_keys))
        print("Unexpected keys:", len(msg.unexpected_keys))
        print("=" * 70)

    return model