import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroShotCountingLoss(nn.Module):
    def __init__(self, lambda_c=0.1, lambda_d=1.0, lambda_cnt=0.1, margin=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_d = lambda_d
        self.lambda_cnt = lambda_cnt
        self.margin = margin

        self.density_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        self.count_loss_fn = nn.L1Loss(reduction='mean')

    def forward(self, pos_output, neg_output, gt_density):
        B = pos_output.shape[0]

        loss_density = self.density_loss_fn(pos_output, gt_density)
        pred_count = (pos_output.view(B, -1)).sum(dim=1) / 60.0
        gt_count = (gt_density.view(B, -1)).sum(dim=1) / 60.0
        loss_count = self.count_loss_fn(pred_count, gt_count)
        zeros_target = torch.zeros_like(neg_output)
        if neg_output.shape[0] == 0:
            loss_neg = torch.tensor(0.0, device=gt_density.device)
        else:
            loss_neg = self.density_loss_fn(neg_output, zeros_target)

        contrastive_loss = torch.relu(loss_density - loss_neg + self.margin)

        loss = (
            self.lambda_c * contrastive_loss +
            self.lambda_d * loss_density +
            self.lambda_cnt * loss_count
        )

        return loss, {
            "loss_total": loss.item(),
            "loss_contrastive": contrastive_loss.item(),
            "loss_density": loss_density.item(),
            "loss_count": loss_count.item()
        }