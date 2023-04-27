import numpy as np
import torch
from sklearn.model_selection import KFold
from einops import rearrange
from fracridge import fracridge


def predict(X, coef):
    return torch.einsum('... n p, ... p f b -> ... f n b', X, coef)


def rsquared(Y, Y_pred, axis=0):
    ss_res = ((Y - Y_pred) ** 2).sum(axis=axis)
    ss_tot = ((Y - Y.mean(axis=axis, keepdims=True)) ** 2).sum(axis=axis)
    r2 = 1 - ss_res / ss_tot
    return r2


def fracridge_cv(X, Y, fractions=None, cv=5):
    if fractions is None:
        fractions = np.arange(.1, 1.1, .1)

    kf = KFold(n_splits=cv, shuffle=True)
    r2 = []
    for train_ids, val_ids in kf.split(np.arange(X.shape[-2])):
        X_train = X[train_ids, :]
        Y_train = Y[train_ids, :]
        X_val = X[val_ids, :]
        Y_val = Y[val_ids, :]

        coef, alpha = fracridge(X_train, Y_train, fractions)
        Y_val_pred = predict(X_val, coef)
        r2.append(rsquared(Y_val[..., None, :, :], Y_val_pred, dim=-2))
    r2 = torch.stack(r2).mean(dim=0)
    best_r2, best_fraction_ids = r2.max(dim=-2)
    best_fractions = fractions[best_fraction_ids]

    best_coefs, best_alpha = frac_ridge_regression(X, Y, best_fractions[None])
    return best_coefs[..., 0, :, :], best_alpha[..., 0, :], best_r2, best_fractions