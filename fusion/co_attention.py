import torch
import torch.nn as nn
import torch.nn.functional as F
from load_embeddings import load_all_embeddings
import numpy as np
import math


class CrossModalAttention(nn.Module):
    """One modality attending over all modalities via Q/K/V cross-modal attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = query.size(0)
        q = query.unsqueeze(1)

        Q = self.W_q(q)
        K = self.W_k(context)
        V = self.W_v(context)

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, context.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)
        attn_out = attn_out.squeeze(1)
        return self.out_proj(attn_out)


class CoAttentionFusion(nn.Module):

    def __init__(
        self,
        input_dims: dict,
        hidden_dim: int = 1024,
        output_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_keys = sorted(input_dims.keys())

        self.projectors = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            for k, dim in input_dims.items()
        })

        self.co_attn = nn.ModuleDict({
            k: CrossModalAttention(hidden_dim, num_heads, dropout)
            for k in self.modality_keys
        })

        self.post_attn_norms = nn.ModuleDict({
            k: nn.LayerNorm(hidden_dim) for k in self.modality_keys
        })

        self.ffn = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for k in self.modality_keys
        })
        self.post_ffn_norms = nn.ModuleDict({
            k: nn.LayerNorm(hidden_dim) for k in self.modality_keys
        })

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def freeze_backbone(self, frozen: bool = True):
        for name in ["co_attn", "ffn", "post_attn_norms", "post_ffn_norms"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = not frozen

    def init_from_concat(self, raw_embeddings: dict):
        print("Initialising projectors from concat baseline...")
        with torch.no_grad():
            for name in self.modality_keys:
                if name not in raw_embeddings:
                    continue
                X = F.normalize(raw_embeddings[name].float(), dim=1)
                linear_layer = self.projectors[name][0]
                D_in = X.shape[1]
                D_out = linear_layer.weight.shape[0]

                if D_in >= D_out:
                    try:
                        U, S, Vh = torch.linalg.svd(X.T @ X)
                        linear_layer.weight.copy_(Vh[:D_out, :])
                    except Exception:
                        pass
                else:
                    W_init = torch.zeros(D_out, D_in)
                    W_init[:D_in, :D_in] = torch.eye(D_in)
                    W_init += torch.randn_like(W_init) * 0.02
                    linear_layer.weight.copy_(W_init)

                if linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)
        print("  Concat-init complete.")

    def forward(self, inputs: dict) -> torch.Tensor:
        present = [k for k in self.modality_keys if k in inputs]

        hidden = {k: self.projectors[k](inputs[k]) for k in present}
        context = torch.stack([hidden[k] for k in present], dim=1)

        attended = {}
        for k in present:
            attn_out = self.co_attn[k](query=hidden[k], context=context)
            h_attn = self.post_attn_norms[k](hidden[k] + attn_out)
            h_ff = self.post_ffn_norms[k](h_attn + self.ffn[k](h_attn))
            attended[k] = h_ff

        stacked = torch.stack([attended[k] for k in present], dim=1)
        pooled = stacked.mean(dim=1)

        fused = self.fusion_head(pooled)
        return F.normalize(fused, p=2, dim=1)


def _smoke_test():
    B = 8
    input_dims = {
        "image_clip": 512, "image_dino": 384,
        "text_clip": 512,  "text_qwen": 1024,
    }
    model = CoAttentionFusion(
        input_dims=input_dims, hidden_dim=256, output_dim=256, num_heads=4
    )
    model.eval()
    dummy = {k: torch.randn(B, d) for k, d in input_dims.items()}
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (B, 256)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)
    print(f"Smoke test passed. Output: {out.shape}, norms≈1: {norms.mean():.6f}")


if __name__ == "__main__":
    import os
    dataset       = os.environ.get("DATASET", "1")
    EMBEDDINGS_DIR      = f"Dataset {dataset}/embeddings"
    FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
    os.makedirs(FUSED_EMBEDDINGS_DIR, exist_ok=True)

    _smoke_test()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = load_all_embeddings(EMBEDDINGS_DIR)
    modality_keys = ["image_clip", "image_dino", "text_clip", "text_qwen"]
    inputs = {k: emb[k].to(device) for k in modality_keys}
    input_dims = {k: inputs[k].shape[1] for k in modality_keys}

    model = CoAttentionFusion(
        input_dims=input_dims,
        hidden_dim=1024,
        output_dim=1024,
        num_heads=8,
        dropout=0.0,
    ).to(device)

    try:
        model.load_state_dict(
            torch.load(f"Dataset {dataset}/models/coattention_trained.pt", map_location=device)
        )
        print(f"Loaded weights from Dataset {dataset}/models/coattention_trained.pt")
    except FileNotFoundError:
        print("Error: Run train_coattention.py first.")
        exit()

    model.eval()
    with torch.no_grad():
        fused = model(inputs)

    fused_np = fused.cpu().numpy().astype(np.float32)
    print("Fused embedding shape:", fused_np.shape)
    np.save(f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_coattention.npy", fused_np)
    print(f"Saved to {FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_coattention.npy")