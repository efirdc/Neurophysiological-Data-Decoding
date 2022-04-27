from typing import Sequence, Callable

import torch
import torchmetrics
import numpy as np

from pipeline.utils import nested_insert


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


def r2_score(Y, Y_pred, dim=0, cast_dtype=torch.float64, reduction: str = 'mean'):
    in_dtype = Y.dtype
    if cast_dtype:
        Y = Y.to(cast_dtype)
        Y_pred = Y_pred.to(cast_dtype)

    ss_res = ((Y - Y_pred) ** 2).sum(dim=dim)
    ss_tot = ((Y - Y.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)

    r2 = 1 - ss_res / ss_tot
    if reduction == 'mean':
        r2 = r2.mean()
    if cast_dtype:
        r2 = r2.to(in_dtype)
    return r2


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


def squared_euclidean_distance(Y1, Y2):
    Y1_squared = (Y1 ** 2).sum(dim=-1)
    Y2_squared = (Y2 ** 2).sum(dim=-1)
    Y1_dot_Y2 = torch.einsum('... i, ... i -> ...', Y1, Y2)

    # recall (y1 - y2)^2 = y1^2 + y2^2 - 2y1*y2
    squared_distance = Y1_squared + Y2_squared - 2 * Y1_dot_Y2
    return squared_distance


def mean_squared_distance(Y1, Y2):
    return squared_euclidean_distance(Y1, Y2) / Y1.shape[1]


def cosine_distance(Y1, Y2):
    Y1 = Y1 / Y1.norm(dim=-1, keepdim=True)
    Y2 = Y2 / Y2.norm(dim=-1, keepdim=True)
    return 1. - torch.einsum('... i, ... i -> ...', Y1, Y2)


def smooth_euclidean_distance(Y1, Y2, beta=1):
    '''
    https://www.desmos.com/calculator/o0ei4cd3ee
    '''
    sqr_distance = squared_euclidean_distance(Y1, Y2)
    distance = torch.sqrt(sqr_distance)
    abs_distance = torch.abs(distance)

    smooth_distance = sqr_distance / (2 * beta)
    mask = abs_distance >= beta
    smooth_distance[mask] = abs_distance[mask] - 0.5 * beta
    return smooth_distance


def contrastive_score(distances, eps=1e-7):
    N = distances.shape[0]
    distances = distances - torch.eye(N) * eps

    column_score = ((distances > distances.diag()).sum(dim=0) / (N - 1)).mean()
    row_score = ((distances > distances.diag()[:, None]).sum(dim=1) / (N - 1)).mean()
    score = (column_score + row_score) * 0.5

    return score


def two_versus_two(distances, eps=1e-7):
    different = distances + distances.T

    distances_diag = torch.diag(distances)
    same = distances_diag[None, :] + distances_diag[:, None] - eps

    comparison = same < different
    return comparison[np.triu_indices(distances.shape[0], k=1)].float().mean()


def two_versus_two_slow(Y1, Y2, distance_measure):
    N = Y1.shape[0]
    results = []
    for i in range(N):
        for j in range(i + 1, N):
            s1 = distance_measure(Y1[i], Y2[i])
            s2 = distance_measure(Y1[j], Y2[j])
            d1 = distance_measure(Y1[i], Y2[j])
            d2 = distance_measure(Y1[j], Y2[i])
            results.append((s1 + s2) < (d1 + d2))
    return torch.tensor(results).float().mean()


def evaluate_decoding(
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        fold_name: str,
        evaluation_metrics: Sequence[Callable] = (r2_score, pearsonr),
        distance_metrics: Sequence[Callable] = (cosine_distance, mean_squared_distance),
        distance_classification_measures: Sequence[Callable] = (two_versus_two, contrastive_score),
):
    evaluation_dict = {}
    Y = Y.flatten(start_dim=1)
    Y_pred = Y_pred.flatten(start_dim=1)
    for evaluation_metric in evaluation_metrics:
        key = (evaluation_metric.__name__, fold_name)
        result = evaluation_metric(Y, Y_pred)
        if isinstance(result, torch.Tensor):
            result = result.item()
        nested_insert(evaluation_dict, key, result)

    for distance_metric in distance_metrics:
        distances = distance_metric(Y[None, :], Y_pred[:, None])
        for measure in distance_classification_measures:
            distance_measure = measure(distances).item()
            key = (distance_metric.__name__, measure.__name__, fold_name)
            nested_insert(evaluation_dict, key, distance_measure)
        nested_insert(evaluation_dict, (distance_metric.__name__, 'mean', fold_name), distances.diag().mean().item())
        nested_insert(evaluation_dict, (distance_metric.__name__, 'std', fold_name), distances.diag().std().item())

    return evaluation_dict

