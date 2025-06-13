import torch
from torch import nn, Tensor
import torch.nn.functional as F

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _compute_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        loss = self.criterion(pred, label).view(-1)
        valid_loss = loss[loss > self.thresh]
        n_min = label[label != self.ignore_label].numel() // 16
        if valid_loss.numel() < n_min:
            valid_loss, _ = loss.topk(n_min)
        return valid_loss.mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(w * self._compute_loss(p, labels) for p, w in zip(preds, self.aux_weights))
        return self._compute_loss(preds, labels)


class FocalLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _compute_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        ce_loss = self.criterion(preds, labels)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(self._compute_loss(p, labels) for p in preds)
        return self._compute_loss(preds, labels)


class DiceLoss(nn.Module):
    def __init__(self, ignore_label: int = 255) -> None:
        super().__init__()
        self.ignore_label = ignore_label

    def _compute_loss(self, preds: Tensor, labels: Tensor) -> Tensor:
        probs = torch.softmax(preds, dim=1)
        smooth = 1e-5
        inter = (probs[:, 1] * (labels == 1).float()).sum()
        union = probs[:, 1].sum() + (labels == 1).float().sum() + smooth
        return 1 - (2. * inter + smooth) / union

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum(self._compute_loss(p, labels) for p in preds)
        return self._compute_loss(preds, labels)

