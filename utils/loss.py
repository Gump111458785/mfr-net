"""Uncertainty-Aware Calibration Loss (UAC Loss).
Couples cross-entropy, margin-based label smoothing, and calibration regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = pred.size(1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction="mean")
        return (1.0 - self.smoothing) * nll + (self.smoothing / c) * loss


class UncertaintyCalibrationLoss(nn.Module):
    """Uncertainty-Aware Calibration Loss (UAC Loss).
    
    L_total = L_ce + lambda_smooth * L_smooth + lambda_cal * L_cal
    Where L_cal penalizes overconfident wrong predictions and uncalibrated margins.
    """
    def __init__(
        self,
        smoothing: float = 0.05,
        cal_weight: float = 0.15,
        use_uac: bool = True,
    ):
        super().__init__()
        self.use_uac = use_uac
        self.cal_weight = cal_weight
        self.smoothing = smoothing
        self.ce_criterion = nn.CrossEntropyLoss()
        self.smooth_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.use_uac:
            return self.ce_criterion(logits, targets)

        # Baseline smoothed CE
        loss_smooth = self.smooth_criterion(logits, targets)

        # Confidence calibration regularizer
        probs = F.softmax(logits, dim=1)
        top_probs, _ = torch.topk(probs, k=2, dim=1)
        margin = top_probs[:, 0] - top_probs[:, 1]

        # Penalize excessive confidence (entropy penalty)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        # Calibration regularization encourages calibrated margin
        cal_penalty = torch.relu(margin - 0.90).mean() - 0.05 * entropy

        total_loss = loss_smooth + self.cal_weight * cal_penalty
        return total_loss
