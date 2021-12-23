from typing import Sequence

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .metrics import embedding_distance


class SoftMinMSE(nn.Module):
    def forward(
            self,
            predictions: Tensor,
            targets: Sequence[Tensor]
    ) -> Tensor:

        losses = []
        for pred, target in zip(list(predictions), targets):
            mse = torch.mean(torch.abs(pred[None] - target), dim=1)
            soft_min_mse = (mse * torch.exp(-mse)).sum() / (torch.exp(-mse)).sum()
            losses.append(soft_min_mse)
        loss = sum(losses) / len(losses)
        return loss


class EuclideanLoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, target, pred):
        delta = target - pred
        return torch.norm(delta, dim=self.dim).mean()


class EmbeddingClassifierLoss(nn.Module):
    def __init__(self, embedding_weight, kl_weight=1.0):
        super().__init__()
        self.register_buffer('embedding_weight', embedding_weight)
        self.kl_weight = kl_weight

    def forward(self, prediction, target):
        embedding_weight = self.embedding_weight.to(prediction.device)
        target = embedding_distance(target, embedding_weight)
        target = target ** 0.5
        target = F.softmax(-target, dim=1)

        prediction = F.log_softmax(prediction, dim=1)

        kl_div = F.kl_div(prediction, target, reduction='sum')
        N, C = prediction.shape[:2]
        kl_div = kl_div / (N * C)
        loss = kl_div * self.kl_weight

        return loss


class EmbeddingDistributionLoss(nn.Module):
    def __init__(self, embedding_weight, distance_criterion=None, kl_weight=1.0):
        super().__init__()
        self.register_buffer('embedding_weight', embedding_weight)
        self.distance_criterion = distance_criterion
        self.kl_weight = kl_weight

    def forward(self, prediction, target):
        embedding_weight = self.embedding_weight.to(prediction.device)
        prediction = embedding_distance(prediction, embedding_weight)
        target = embedding_distance(target, embedding_weight)

        prediction = prediction ** 0.5
        target = target ** 0.5

        prediction = F.log_softmax(-prediction, dim=1)
        target = F.softmax(-target, dim=1)

        kl_div = F.kl_div(prediction, target, reduction='sum')
        N, C = prediction.shape[:2]
        kl_div = kl_div / (N * C)
        kl_div_loss = kl_div * self.kl_weight

        loss = kl_div_loss
        loss_dict = {'loss': loss, 'kl_div': kl_div, 'kl_div_loss': kl_div_loss}

        if self.distance_criterion:
            distance_loss = self.distance_criterion(prediction, target)
            loss = loss + distance_loss
            loss_dict['distance_loss'] = distance_loss
            loss_dict['loss'] = loss

        loss_dict = {k: round(v.detach().cpu().item(), 3) for k, v in loss_dict.items()}

        return loss, loss_dict


class ProbabalisticCrossEntropyLoss(nn.Module):
    def __init__(self, smoothness=0, eps=1e-5, normalizer=True):
        super().__init__()
        self.smoothness = smoothness
        self.eps = eps
        self.normalizer = normalizer

    def forward(self, prediction, target):
        prediction = F.softmax(prediction, dim=1)

        if self.smoothness > 0:
            target = torch.clamp(prediction, target - self.smoothness, target + self.smoothness)

        one_minus_target = 1. - target
        loss = -target * self.log(prediction) - one_minus_target * self.log(1. - prediction)

        if self.normalizer:
            normalizer = target * self.log(target) + one_minus_target * self.log(one_minus_target)
            loss = loss + normalizer

        return loss.sum(dim=1).mean()

    def log(self, x):
        if self.eps > 0:
            return torch.log((x + self.eps) / (1. + 2. * self.eps))
        else:
            return torch.log(x)


class VariationalLoss(nn.Module):
    def __init__(
            self,
            distance_loss: nn.Module,
            kld_weight: float,
    ):
        super().__init__()
        self.distance_loss = distance_loss
        self.kld_weight = kld_weight

    def forward(self, target, prediction, mu, log_var):
        distance_loss = self.distance_loss(target, prediction)
        kld = ((1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1) * -0.5).mean()
        kld_loss = kld * self.kld_weight

        loss = distance_loss + kld_loss
        loss_dict = {'loss': loss, 'distance_loss': distance_loss, 'kld_loss': kld_loss, 'kld': kld}

        return loss, loss_dict


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = self.cosine_similarity(x, y)
        loss = 1. - loss
        loss = loss * 0.5 + 0.5
        return loss.mean()


class ContrastiveCosineSimilarityLoss(nn.Module):
    def __init__(self, initial_temperature=1.):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, prediction, target):
        prediction = F.normalize(prediction, dim=1)
        target = F.normalize(target, dim=1)

        sim = torch.einsum('i d, j d -> i j', prediction, target) * self.temperature.exp()
        labels = torch.arange(target.shape[0], device=target.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

