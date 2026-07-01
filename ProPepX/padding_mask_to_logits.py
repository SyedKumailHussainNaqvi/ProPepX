from Import_libraries import *

def apply_padding_mask_to_logits(logits, mask):
    """
    logits: (B, L, C)
    mask:   (B, L), True = PAD
    """
    if mask is None:
        return logits
    return logits.masked_fill(mask.unsqueeze(-1), 0.0)