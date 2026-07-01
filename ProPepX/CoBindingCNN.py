from Import_libraries import *
from padding_mask_to_logits import *


# Learnable positional encoding

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len=1418, dim=256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        if L > self.pos_embed.size(1):
            raise ValueError(
                f"Input length {L} exceeds max positional length {self.pos_embed.size(1)}"
            )
        return x + self.pos_embed[:, :L, :]

# Input projection: PLM embedding -> hidden_dim

class InputProjection(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Multi-scale CNN block

class MultiScaleCNNBlock(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()

        branch_dim = hidden_dim // 4
        remain_dim = hidden_dim - (branch_dim * 3)

        self.conv_k3 = nn.Conv1d(hidden_dim, branch_dim, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(hidden_dim, branch_dim, kernel_size=5, padding=2)
        self.conv_k9 = nn.Conv1d(hidden_dim, branch_dim, kernel_size=9, padding=4)
        self.conv_dil = nn.Conv1d(hidden_dim, remain_dim, kernel_size=3, padding=2, dilation=2)

        self.bn = nn.BatchNorm1d(hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, H)
        residual = x
        x = x.transpose(1, 2)  # (B, H, L)

        b1 = self.conv_k3(x)
        b2 = self.conv_k5(x)
        b3 = self.conv_k9(x)
        b4 = self.conv_dil(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.out_proj(x)

        x = x.transpose(1, 2)  # (B, L, H)
        x = self.norm(residual + self.dropout(x))
        return x

# Self-attention encoder

class SelfContextEncoder(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, heads=4, ff_dim=512, dropout=0.2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, padding_mask=None):
        # x: (B, L, H)
        # padding_mask: (B, L), True = PAD
        return self.encoder(x, src_key_padding_mask=padding_mask)


# Cross-attention block

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=256, heads=4, dropout=0.3):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, key_padding_mask=None):
        # query:      (B, Lq, H)
        # key_value:  (B, Lk, H)
        # key_padding_mask: (B, Lk), True = PAD
        attn_out, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(query + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_weights

# Gated fusion

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, self_feat, cross_feat, return_gate=False):
        combined = torch.cat([self_feat, cross_feat], dim=-1)  # (B, L, 2H)

        gate = self.gate(combined)    # (B, L, H)
        fused = self.fuse(combined)   # (B, L, H)

        out = gate * fused + (1.0 - gate) * self_feat
        out = self.norm(self_feat + self.dropout(out))

        if return_gate:
            return out, gate
        return out

# Residue classifier

class ResidueClassifier(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        # x: (B, L, H)
        return self.net(x)  # (B, L, 2)


# ============================================================
# Full ProPepX
# IMPORTANT:
# - full architecture is ALWAYS built
# - mode only controls which logits are returned
#   "prot" -> return protein logits only, but uses peptide context
#   "pep"  -> return peptide logits only, but uses protein context
#   "both" -> return both
# ============================================================
class ProPepX(nn.Module):
    def __init__(
        self,
        emb_dim=1024,
        hidden_dim=256,
        heads=4,
        dropout=0.2,
        mode="both",
        num_self_layers=2,
        ff_dim=512,
        max_len_prot=1418,
        max_len_pep=50
    ):
        super().__init__()
        self.mode = mode.lower()
        
        if self.mode == "mode-global":
            self.mode = "both"

        if self.mode not in ["prot", "pep", "both"]:
            raise ValueError("mode must be one of ['prot', 'pep', 'both']")

        print(f"Loading ProPepX architecture in mode={self.mode}")

        self.prot_proj = InputProjection(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.prot_pos = LearnablePositionalEncoding(max_len=max_len_prot, dim=hidden_dim)
        self.prot_cnn = MultiScaleCNNBlock(hidden_dim=hidden_dim, dropout=dropout)
        self.prot_self = SelfContextEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_self_layers,
            heads=heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        self.prot_classifier = ResidueClassifier(hidden_dim=hidden_dim, dropout=dropout)

        self.pep_proj = InputProjection(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.pep_pos = LearnablePositionalEncoding(max_len=max_len_pep, dim=hidden_dim)
        self.pep_cnn = MultiScaleCNNBlock(hidden_dim=hidden_dim, dropout=dropout)
        self.pep_self = SelfContextEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_self_layers,
            heads=heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        self.pep_classifier = ResidueClassifier(hidden_dim=hidden_dim, dropout=dropout)

        self.cross_pep_to_prot = CrossAttentionBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)
        self.cross_prot_to_pep = CrossAttentionBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout)

        self.prot_fusion = GatedFusion(hidden_dim=hidden_dim, dropout=dropout)
        self.pep_fusion = GatedFusion(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, prot_emb, pep_emb, prot_mask=None, pep_mask=None, return_attention=False):
        prot_feat = self.prot_proj(prot_emb)
        prot_feat = self.prot_pos(prot_feat)
        prot_feat = self.prot_cnn(prot_feat)
        prot_self_feat = self.prot_self(prot_feat, padding_mask=prot_mask)

        pep_feat = self.pep_proj(pep_emb)
        pep_feat = self.pep_pos(pep_feat)
        pep_feat = self.pep_cnn(pep_feat)
        pep_self_feat = self.pep_self(pep_feat, padding_mask=pep_mask)

        prot_cross_feat, prot_attn = self.cross_pep_to_prot(
            query=prot_self_feat,
            key_value=pep_self_feat,
            key_padding_mask=pep_mask
        )

        pep_cross_feat, pep_attn = self.cross_prot_to_pep(
            query=pep_self_feat,
            key_value=prot_self_feat,
            key_padding_mask=prot_mask
        )

        prot_fused, prot_gate = self.prot_fusion(prot_self_feat, prot_cross_feat, return_gate=True)
        pep_fused, pep_gate = self.pep_fusion(pep_self_feat, pep_cross_feat, return_gate=True)

        prot_logits = self.prot_classifier(prot_fused)
        pep_logits = self.pep_classifier(pep_fused)

        prot_logits = apply_padding_mask_to_logits(prot_logits, prot_mask)
        pep_logits = apply_padding_mask_to_logits(pep_logits, pep_mask)

        if self.mode == "prot":
            pep_logits = None
        elif self.mode == "pep":
            prot_logits = None

        if return_attention:
            return prot_logits, pep_logits, {
                "prot_attn": prot_attn,   # (B, Lp, Lq)
                "pep_attn": pep_attn,     # (B, Lq, Lp)
                "prot_gate": prot_gate,   # (B, Lp, H)
                "pep_gate": pep_gate      # (B, Lq, H)
            }

        return prot_logits, pep_logits
    
def get_valid_lengths(prot_mask=None, pep_mask=None):
    valid_prot_len = None
    valid_pep_len = None

    if prot_mask is not None:
        valid_prot_len = int((~prot_mask[0]).sum().item())
    if pep_mask is not None:
        valid_pep_len = int((~pep_mask[0]).sum().item())

    return valid_prot_len, valid_pep_len

# Plot cross-attention heatmap

def plot_cross_attention_heatmap(
    attn_map,
    save_path,
    title="Cross-attention heatmap: protein attending to peptide"
):
    plt.figure(figsize=(9, 6))
    plt.imshow(attn_map, aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Peptide residue index")
    plt.ylabel("Protein residue index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Plot gate values as residue-wise line plot

def plot_gate_values(
    gate_tensor,
    save_path,
    title="Gated fusion importance across protein residues"
):
    gate_per_residue = gate_tensor.mean(axis=1)  # (L,)

    plt.figure(figsize=(10, 3.5))
    plt.plot(np.arange(1, len(gate_per_residue) + 1), gate_per_residue)
    plt.ylim(0, 1)
    plt.xlabel("Protein residue index")
    plt.ylabel("Mean gate value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Optional: gate heatmap across residues x hidden channels

def plot_gate_heatmap(
    gate_tensor,
    save_path,
    title="Gate heatmap across residues and hidden channels"
):
    plt.figure(figsize=(10, 5))
    plt.imshow(gate_tensor.T, aspect="auto")
    plt.colorbar(label="Gate value")
    plt.xlabel("Protein residue index")
    plt.ylabel("Hidden channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Optional: predicted protein binding probability

def plot_binding_probability(
    prot_logits,
    save_path,
    title="Predicted protein binding-site probability"
):
    prot_prob = torch.softmax(prot_logits, dim=-1)[0, :, 1].detach().cpu().numpy()

    plt.figure(figsize=(10, 3.5))
    plt.plot(np.arange(1, len(prot_prob) + 1), prot_prob)
    plt.ylim(0, 1)
    plt.xlabel("Protein residue index")
    plt.ylabel("Binding probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")