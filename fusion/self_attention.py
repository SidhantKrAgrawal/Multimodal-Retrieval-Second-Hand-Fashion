import torch
import torch.nn as nn
import torch.nn.functional as F
from load_embeddings import load_all_embeddings
import numpy as np


class ModalityTypeEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class MultiModalSelfAttentionFusion(nn.Module):

    def __init__(
        self,
        input_dims: dict,
        d_model: int = 1024,
        output_dim: int = 1024,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modality_keys = sorted(input_dims.keys())
        num_modalities = len(self.modality_keys)
        self.d_model = d_model

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            for name, dim in input_dims.items()
        })

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.modality_embeddings = ModalityTypeEmbedding(num_modalities + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def freeze_backbone(self, frozen: bool = True):
        for param in self.transformer.parameters():
            param.requires_grad = not frozen
        for param in self.projections.parameters():
            param.requires_grad = not frozen
        for param in self.modality_embeddings.parameters():
            param.requires_grad = not frozen
        self.cls_token.requires_grad = True

    def init_from_concat(self, raw_embeddings: dict):
        """SVD warm-start: initialise projection weights from each modality's principal components."""
        print("Initialising projections from concat baseline...")
        with torch.no_grad():
            for name in self.modality_keys:
                if name not in raw_embeddings:
                    continue

                X = F.normalize(raw_embeddings[name].float(), dim=1)
                linear_layer = self.projections[name][0]
                D_in  = X.shape[1]
                D_out = linear_layer.weight.shape[0]

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

    def forward(self, inputs: dict) -> torch.Tensor:
        B = next(iter(inputs.values())).size(0)

        tokens = [self.cls_token.expand(B, -1, -1)]

        for name in self.modality_keys:
            if name in inputs and name in self.projections:
                x = self.projections[name](inputs[name])
                tokens.append(x.unsqueeze(1))

        sequence = torch.cat(tokens, dim=1)
        sequence = self.modality_embeddings(sequence)
        transformed = self.transformer(sequence)
        cls_output = transformed[:, 0, :]

        out = self.out_head(cls_output)
        return F.normalize(out, p=2, dim=1)


# ── Smoke test ──────────────────────────────────────────────────────────────
def _smoke_test():
    B = 8
    input_dims = {"image_clip": 512, "image_dino": 384,
                  "text_clip": 512,  "text_qwen": 1024}
    model = MultiModalSelfAttentionFusion(
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

    model = MultiModalSelfAttentionFusion(
        input_dims=input_dims,
        d_model=1024,
        output_dim=1024,
        nhead=8,
        num_layers=2,
        dropout=0.0,
    ).to(device)

    try:
        model.load_state_dict(
            torch.load(f"Dataset {dataset}/models/selfattention_trained.pt", map_location=device)
        )
        print("Loaded trained weights.")
    except FileNotFoundError:
        print("ERROR: Run train_selfattention.py first!")
        exit()

    model.eval()
    inputs_torch = {k: emb[k].to(device) for k in modality_keys}

    with torch.no_grad():
        fused = model(inputs_torch)

    fused_np = fused.cpu().numpy().astype(np.float32)
    print("Fused shape:", fused_np.shape)
    np.save(f"{FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_selfattention.npy", fused_np)
    print(f"Saved to {FUSED_EMBEDDINGS_DIR}/fused_item_embeddings_selfattention.npy")