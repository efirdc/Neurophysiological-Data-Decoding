import numpy as np
import torch
from sklearn.model_selection import KFold
from einops import rearrange
from fracridge import fracridge


def pearsonr(X, Y, dim=0, cast_dtype=torch.float64):
    in_dtype = X.dtype
    X = X.to(cast_dtype)
    Y = Y.to(cast_dtype)

    X = X - X.mean(dim=dim, keepdim=True)
    Y = Y - Y.mean(dim=dim, keepdim=True)

    X = X / torch.norm(X, dim=dim, keepdim=True)
    Y = Y / torch.norm(Y, dim=dim, keepdim=True)

    r = torch.tensordot(X, Y, dims=dim).to(in_dtype)
    return r


def rsquared(Y, Y_pred, dim=0, cast_dtype=torch.float64):
    in_dtype = Y.dtype
    Y = Y.to(cast_dtype)
    Y_pred = Y_pred.to(cast_dtype)

    ss_res = (Y ** 2).sum(dim=dim) + (Y_pred ** 2).sum(dim=dim) - 2 * torch.einsum('...ij,...ij->...j', Y, Y_pred)
    #ss_res = ((Y - Y_pred) ** 2).sum(dim=dim)
    ss_tot = ((Y - Y.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)

    r2 = 1 - ss_res / ss_tot
    return r2.to(in_dtype)


def frac_ridge_regression(X, Y, fractions=None, tol=1e-6, cast_dtype=torch.float64):
    BIG_BIAS = 10e3
    SMALL_BIAS = 10e-3
    BIAS_STEP = 0.2

    print = lambda x: x

    if fractions is None:
        fractions = torch.arange(.1, 1.1, .1)
    if cast_dtype:
        in_dtype = X.dtype
        X = X.to(cast_dtype)
        Y = Y.to(cast_dtype)
        fractions = fractions.to(cast_dtype)

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    Y_new = U.transpose(-1, -2) @ Y
    ols_coef = (Y_new.transpose(-1, -2) / S[..., None, :]).transpose(-1, -2)

    S_small = torch.broadcast_to(S < tol, ols_coef.shape[:-1])
    ols_coef[S_small, ...] = 0.

    val1 = BIG_BIAS * S[..., 0] ** 2
    val2 = SMALL_BIAS * S[..., -1] ** 2
    val2 = torch.max(val2, torch.tensor(1e-8).to(val2.device))
    print(f'{val1=}, {val2=}')

    grid_low = torch.floor(torch.log10(val2))
    grid_high = torch.ceil(torch.log10(val1))
    print(f'{grid_low=}, {grid_high=},')
    steps = int(torch.max(grid_high - grid_low).item() / BIAS_STEP)
    alphagrid = 10 ** torch.stack([
        (i / steps) * grid_high + (1 - i / steps) * grid_low
        for i in range(steps)
    ])
    alphagrid = torch.cat([torch.zeros_like(grid_low)[None], alphagrid])
    print(f'{alphagrid.shape=}, {alphagrid=}')

    S_squared = S ** 2
    scaling = S_squared / (S_squared + alphagrid[..., None])
    scaling_squared = scaling ** 2
    print(f'{scaling_squared.shape=}, {scaling_squared.isnan().sum()=}')

    newlen = torch.sqrt(torch.einsum('g ... p, ... p b -> g ... b', scaling_squared, ols_coef ** 2))
    newlen = (newlen / newlen[0])
    print(f'{newlen.shape=}, {newlen.T=}')

    while len(fractions.shape) < len(newlen.shape):
        fractions = fractions[:, None]

    threshold = fractions[None, :] < newlen[:, None]
    threshold = threshold.sum(dim=0)
    #threshold = (threshold[1:] != threshold[:-1]).int()
    #threshold = threshold.argmax(dim=0)
    print(f'{threshold.shape=}, {threshold.T=}')
    #print(f'{threshold2.shape=}, {threshold2=}')

    newlen_high = torch.gather(newlen, 0, torch.clamp(threshold - 1, min=0))
    newlen_low = torch.gather(newlen, 0, threshold)
    print(f'{newlen_high.shape=}, {newlen_high.T=}')
    print(f'{newlen_low.shape=}, {newlen_low.T=}')

    t = (newlen_high - fractions) / (newlen_high - newlen_low)
    t[t.isnan()] = 0.
    print(f'{t.shape=}, {t.T=}')
    log_alphagrid = torch.log(1 + alphagrid)
    log_alphagrid = torch.broadcast_to(log_alphagrid[..., None], newlen.shape)
    print(f'{log_alphagrid.shape=}, {log_alphagrid.T=}')

    alpha_high = torch.gather(log_alphagrid, 0, torch.clamp(threshold - 1, min=0))
    alpha_low = torch.gather(log_alphagrid, 0, threshold)
    print(f'{alpha_high.shape=}, {alpha_high.T=}')
    print(f'{alpha_low.shape=}, {alpha_low.T=}')
    alpha = (1. - t) * alpha_high + t * alpha_low
    alpha = torch.exp(alpha) - 1.
    print(f'{alpha.shape=}, {alpha.T=}')

    sc = S_squared / (S_squared + rearrange(alpha, 'f ... b -> f b ... 1'))
    coef = sc * rearrange(ols_coef, '... p b -> 1 b ... p')

    coef = torch.einsum('... p i, f b ... p -> ... f i b', Vt, coef)
    alpha = rearrange(alpha, 'f ... b -> ... f b')

    print(coef.norm(dim=-2) / ols_coef.norm(dim=-2))
    if cast_dtype:
        coef, alpha = [tensor.to(in_dtype) for tensor in (coef, alpha)]

    return coef, alpha


def predict(X, coef):
    return torch.einsum('... n p, ... f p b -> ... f n b', X, coef)


def frac_ridge_regression_cv(X, Y, fractions=None, cv=5, tol=1e-6):
    if fractions is None:
        fractions = torch.arange(.1, 1.1, .1, device=X.device)

    kf = KFold(n_splits=cv, shuffle=True)
    r2 = []
    for train_ids, val_ids in kf.split(np.arange(X.shape[-2])):
        X_train = X[..., train_ids, :]
        Y_train = Y[..., train_ids, :]
        X_val = X[..., val_ids, :]
        Y_val = Y[..., val_ids, :]

        coef, alpha = frac_ridge_regression(X_train, Y_train, fractions)
        Y_val_pred = predict(X_val, coef)
        r2.append(rsquared(Y_val[..., None, :, :], Y_val_pred, dim=-2))
    r2 = torch.stack(r2).mean(dim=0)
    best_r2, best_fraction_ids = r2.max(dim=-2)
    best_fractions = fractions[best_fraction_ids]

    best_coefs, best_alpha = frac_ridge_regression(X, Y, best_fractions[None])
    return best_coefs[..., 0, :, :], best_alpha[..., 0, :], best_r2, best_fractions


def ridge_regression(X, Y, alpha=None,):
    lhs = X.transpose(-2, -1) @ X
    rhs = X.transpose(-2, -1) @ Y
    if alpha is None:
        return torch.linalg.lstsq(lhs, rhs).solution
    else:
        ridge = alpha * torch.eye(lhs.shape[-2], device=X.device)
        return torch.linalg.lstsq(lhs + ridge, rhs).solution


def ridge_regression_cv(X, Y, alpha=None, cv=5, tol=1e-6):
    if alpha is None:
        alpha = 10 ** torch.linspace(1, 5, 20)

    kf = KFold(n_splits=cv, shuffle=True)
    r2 = []
    for train_ids, val_ids in kf.split(np.arange(X.shape[-2])):
        X_train = X[..., train_ids, :]
        Y_train = Y[..., train_ids, :]
        X_val = X[..., val_ids, :]
        Y_val = Y[..., val_ids, :]

        # Add a new dimension for alphas (try every alpha vs every target)
        coefs = ridge_regression(X_train[None], Y_train[None], alpha[:, None, None])
        Y_val_pred = X_val[None] @ coefs

        r2.append(rsquared(Y_val[None], Y_val_pred, dim=-2))

    r2 = torch.stack(r2).mean(dim=0)
    best_r2, best_alpha_ids = r2.max(dim=-2)
    best_alpha = alpha[best_alpha_ids]

    best_coefs = ridge_regression(X[None], Y.transpose(-2, -1)[..., None], best_alpha[:, None, None])
    best_coefs = best_coefs[..., 0].transpose(-1, -2)
    return best_coefs, best_alpha, best_r2
