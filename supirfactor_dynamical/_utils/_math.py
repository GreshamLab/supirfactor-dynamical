import torch
import numpy as np


def _calculate_erv(rss_full, rss_reduced):

    erv = rss_full[:, None] / rss_reduced

    try:
        erv = erv.numpy()
    except AttributeError:
        pass

    # Check for precision-related differences in rss
    # and fix any NaNs
    _eps = np.finfo(erv.dtype).eps * 2

    _no_contrib_mask = np.abs(rss_full[:, None] - rss_reduced)
    _no_contrib_mask = _no_contrib_mask <= _eps

    erv[np.isnan(erv)] = 1.0
    erv[_no_contrib_mask.numpy()] = 1.0

    erv = 1 - erv

    return erv


def _calculate_tss(
    data,
    ybar=True
):

    _last_axis = data.ndim - 1

    if ybar:
        _comparison = data.mean(
                axis=list(range(_last_axis))
            ).reshape(
                *[1 if i != _last_axis else -1 for i in range(data.ndim)]
            ).expand(
                data.shape
            )
    else:
        _comparison = torch.zeros_like(data)

    tss = torch.nn.MSELoss(reduction='none')(
        data,
        _comparison
    ).sum(
        axis=list(range(_last_axis))
    )

    return tss


def _calculate_rss(x, y):

    _last_axis = x.ndim - 1

    return torch.nn.MSELoss(reduction='none')(
        x,
        y
    ).sum(
        axis=list(range(_last_axis))
    )


def _calculate_r2(rss, tss):

    valid_ss = tss != 0.

    # Special case where all tss are zeros
    # shouldn't ever actually happen outside testing
    if not torch.any(valid_ss):
        return torch.zeros_like(tss)

    r2 = torch.full_like(tss, np.nan)
    r2[valid_ss] = rss[valid_ss] / tss[valid_ss]
    r2 *= -1
    r2 += 1

    return r2


def _true_positive(x, target):

    return torch.logical_and(
        x,
        target
    ).sum(axis=tuple(range(x.ndim - 1)))


def _false_positive(x, target):

    return torch.logical_and(
        x,
        torch.logical_not(target)
    ).sum(axis=tuple(range(x.ndim - 1)))


def _true_negative(x, target):

    return torch.logical_and(
        torch.logical_not(x),
        torch.logical_not(target)
    ).sum(axis=tuple(range(x.ndim - 1)))


def _false_negative(x, target):

    return torch.logical_and(
        torch.logical_not(x),
        target
    ).sum(axis=tuple(range(x.ndim - 1)))


def _f1_score(tp, fp, fn):

    tp = 2 * tp

    return tp / (tp + fp + fn)
