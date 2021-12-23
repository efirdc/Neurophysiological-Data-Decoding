from typing import Sequence

import torch
import torchmetrics


def best_r2_score(
        predictions: torch.Tensor,
        targets: Sequence[torch.Tensor],
) -> torch.Tensor:

    best_targets = []
    for prediction, target in zip(list(predictions), targets):
        mse = torch.mean((prediction[None] - target) ** 2, dim=1)
        best_mse_arg = torch.argmin(mse)
        best_targets.append(target[best_mse_arg])
    best_targets = torch.stack(best_targets)

    return torchmetrics.functional.r2_score(predictions, best_targets)


def pearsonr(Y, Y_pred, dim=0):
    Y = Y.to(torch.float64)
    Y_pred = Y_pred.to(torch.float64)

    Y = Y - Y.mean(dim=dim, keepdim=True)
    Y_pred = Y_pred - Y_pred.mean(dim=dim, keepdim=True)

    Y = Y / torch.norm(Y, dim=dim, keepdim=True)
    Y_pred = Y_pred / torch.norm(Y_pred, dim=dim, keepdim=True)

    return (Y * Y_pred).sum(dim=dim).mean().item()


def cosine_similarity(Y, Y_pred, dim=0):
    Y = Y.to(torch.float64)
    Y_pred = Y_pred.to(torch.float64)

    Y = Y / torch.norm(Y, dim=dim, keepdim=True)
    Y_pred = Y_pred / torch.norm(Y_pred, dim=dim, keepdim=True)

    return (Y * Y_pred).sum(dim=dim).mean().item()


def embedding_distance(x, embedding_weight):
    x_squared = (x ** 2).sum(dim=1)[:, None]
    embed_squared = (embedding_weight ** 2).sum(dim=1)[None, :, None, None]
    x_dot_embed = torch.einsum('bchw, ec -> behw', x, embedding_weight)

    # recall (x - e)^2 = x^2 + e^2 - 2xe
    distance_squared = x_squared + embed_squared - 2 * x_dot_embed
    return distance_squared
