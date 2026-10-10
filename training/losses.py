import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss supporting fractional pos_mask weights for soft positives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,   # (B, D) L2-normalised
        pos_mask: torch.Tensor,    # (B, B) values in [0,1], diagonal=0
    ) -> torch.Tensor:
        B, device = features.size(0), features.device

        sim = torch.matmul(features, features.T) / self.temperature

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        exp_sim = torch.exp(sim)
        exp_no_self = exp_sim.masked_fill(self_mask, 0.0)
        log_denom = torch.log(exp_no_self.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = sim - log_denom

        weight_sum = pos_mask.sum(dim=1)
        has_pos = weight_sum > 0

        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / weight_sum.clamp(min=1e-8)
        loss = -mean_log_prob[has_pos].mean()
        return loss


class UniformityLoss(nn.Module):
    """Penalises embedding collapse by encouraging uniform distribution on the hypersphere."""

    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sq_dists = torch.pdist(features, p=2).pow(2)
        return sq_dists.mul(-self.t).exp().mean().log()


class CombinedLoss(nn.Module):
    """Contrastive loss with optional uniformity regularisation (set lambda_uniformity=0 to disable)."""

    def __init__(self, temperature: float = 0.07, lambda_uniformity: float = 0.5):
        super().__init__()
        self.supcon = SoftSupervisedContrastiveLoss(temperature)
        self.uniformity = UniformityLoss()
        self.lambda_u = lambda_uniformity

    def forward(
        self,
        features: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        sc_loss = self.supcon(features, pos_mask)
        u_loss = self.uniformity(features) if self.lambda_u > 0 else torch.tensor(0.0)
        total = sc_loss + self.lambda_u * u_loss
        return total, {"supcon": sc_loss.item(), "uniformity": u_loss.item()}