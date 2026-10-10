import torch
import torch.nn as nn
import torch.nn.functional as F
from load_embeddings import load_all_embeddings
import numpy as np


class ModalityEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query=query, key=context, value=context)
        return self.norm(query + self.drop(attn_out))


class CrossModalFusion(nn.Module):

    VISUAL_KEYS = ["image_clip", "image_dino"]
    TEXT_KEYS   = ["text_clip",  "text_qwen"]

    def __init__(
        self,
        input_dims: dict,
        d_model: int    = 1024,
        output_dim: int = 1024,
        nhead: int      = 8,
        num_layers: int = 2,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.output_dim = output_dim
        self.visual_keys = sorted([k for k in self.VISUAL_KEYS if k in input_dims])
        self.text_keys   = sorted([k for k in self.TEXT_KEYS   if k in input_dims])
        all_modalities   = self.visual_keys + self.text_keys

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            for name, dim in input_dims.items()
        })

        self.visual_mod_emb = ModalityEmbedding(len(self.visual_keys), d_model)
        self.text_mod_emb   = ModalityEmbedding(len(self.text_keys),   d_model)

        def _make_self_attn():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)

        self.visual_self_attn = _make_self_attn()
        self.text_self_attn   = _make_self_attn()

        self.visual_cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.text_cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.final_mod_emb = ModalityEmbedding(len(all_modalities) + 1, d_model)
        self.cls_cross_attn = CrossAttentionBlock(d_model, nhead, dropout)

        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def freeze_backbone(self, frozen: bool = True):
        backbone_modules = [
            self.visual_self_attn,
            self.text_self_attn,
            self.visual_cross_layers,
            self.text_cross_layers,
        ]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = not frozen

    def init_from_concat(self, raw_embeddings: dict):
        print("Initialising cross-attention projections from concat baseline...")
        with torch.no_grad():
            for name in (self.visual_keys + self.text_keys):
                if name not in raw_embeddings:
                    continue

                X = F.normalize(raw_embeddings[name].float(), dim=1)  # (N, D_in)
                linear_layer = self.projections[name][0]               # nn.Linear
                D_in  = X.shape[1]
                D_out = linear_layer.weight.shape[0]                   # d_model

                if D_in >= D_out:
                    try:
                        _, _, Vh = torch.linalg.svd(X.T @ X)
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

    def _project_stream(self, inputs: dict, keys: list) -> torch.Tensor:
        tokens = [self.projections[k](inputs[k]).unsqueeze(1) for k in keys if k in inputs]
        return torch.cat(tokens, dim=1)

    def forward(self, inputs: dict) -> torch.Tensor:
        visual_tokens = self._project_stream(inputs, self.visual_keys)
        text_tokens   = self._project_stream(inputs, self.text_keys)

        visual_tokens = self.visual_mod_emb(visual_tokens)
        text_tokens   = self.text_mod_emb(text_tokens)

        visual_tokens = self.visual_self_attn(visual_tokens)
        text_tokens   = self.text_self_attn(text_tokens)

        for v_layer, t_layer in zip(self.visual_cross_layers, self.text_cross_layers):
            visual_ctx = v_layer(query=visual_tokens, context=text_tokens)
            text_ctx   = t_layer(query=text_tokens,   context=visual_tokens)
            visual_tokens = visual_ctx
            text_tokens   = text_ctx

        all_tokens = torch.cat([visual_tokens, text_tokens], dim=1)

        B   = all_tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        cls_input = torch.cat([cls, all_tokens], dim=1)
        cls_input = self.final_mod_emb(cls_input)
        cls_alone = cls_input[:, :1, :]
        ctx       = cls_input[:, 1:,  :]
        cls_out   = self.cls_cross_attn(query=cls_alone, context=ctx)
        cls_out   = cls_out.squeeze(1)

        out = self.out_head(cls_out)
        return F.normalize(out, p=2, dim=1)


def _smoke_test():
    B = 8
    input_dims = {"image_clip": 512, "image_dino": 384,
                  "text_clip":  512, "text_qwen":  1024}
    model = CrossModalFusion(
        input_dims=input_dims, d_model=256, output_dim=256, nhead=4, num_layers=2
    )
    model.eval()
    dummy = {k: torch.randn(B, d) for k, d in input_dims.items()}
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (B, 256), f"Bad shape: {out.shape}"
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), "Not L2-normalised"
    print(f"Smoke test passed — output: {out.shape}, mean norm: {norms.mean():.6f}")


if __name__ == "__main__":
    import os
    dataset        = os.environ.get("DATASET", "1")
    EMBEDDINGS_DIR       = f"Dataset {dataset}/embeddings"
    FUSED_EMBEDDINGS_DIR = f"Dataset {dataset}/fused_embeddings"
    os.makedirs(FUSED_EMBEDDINGS_DIR, exist_ok=True)

    _smoke_test()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = load_all_embeddings(EMBEDDINGS_DIR)
    modality_keys = ["image_clip", "image_dino", "text_clip", "text_qwen"]
    input_dims = {k: emb[k].shape[1] for k in modality_keys}

    model = CrossModalFusion(
        input_dims=input_dims,
        d_model=1024,
        output_dim=1024,
        nhead=8,
        num_layers=2,
        dropout=0.0,
    ).to(device)

    try:
        model.load_state_dict(
            torch.load(f"Dataset {dataset}/models/crossattention_trained.pt", map_location=device)
        )
        print("Loaded trained weights.")
    except FileNotFoundError:
        print("ERROR: Run train_crossattention.py first!")
        exit()

    model.eval()
    inputs_torch = {k: emb[k].to(device) for k in modality_keys}

    with torch.no_grad():
        fused = model(inputs_torch)

    fused_np = fused.cpu().numpy().astype(np.float32)
    print("Fused shape:", fused_np.shape)
    np.save(f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_crossattention.npy", fused_np)
    print(f"Saved to {FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_crossattention.npy")