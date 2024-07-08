import torch

from supirfactor_dynamical._utils._math import (
    _calculate_rss,
    _calculate_tss,
    _calculate_r2,
    _true_positive,
    _false_negative,
    _false_positive,
    _f1_score
)
from supirfactor_dynamical._utils.misc import (
    argmax_last_dim,
    to
)
from supirfactor_dynamical.datasets import stack_dataloaders


def r2_score(
    dataloader,
    model,
    target_data_idx=None,
    input_data_idx=None,
    multioutput='uniform_average',
    exclude_low_variance=None
):

    if dataloader is None:
        return None

    model.eval()

    _rss = 0
    _tss = 0
    _n = 0

    with torch.no_grad():
        for data in stack_dataloaders(dataloader):

            input_data, target_data = _extract_data(
                data,
                model,
                input_data_idx,
                target_data_idx
            )

            _rss += _calculate_rss(
                target_data,
                model._slice_data_and_forward(input_data),
            )

            _tss += _calculate_tss(
                target_data
            )

            _n = _n + target_data.shape[0]

            # Exclude very low variance features from
            # R2 calculation
            # Useful if averaging across features
            # when some features are not informative
            if exclude_low_variance is not None:
                _var_mask = (_tss / (_n - 1)) < exclude_low_variance
                _tss[_var_mask] = 0
                _rss[_var_mask] = 0

        if multioutput == 'raw_values':
            return _calculate_r2(_rss, _tss)
        elif multioutput == 'uniform_average':
            return torch.nanmean(_calculate_r2(_rss, _tss))
        elif multioutput == 'uniform_truncated_average':
            rsq = _calculate_r2(_rss, _tss)
            rsq[rsq < 0] = 0
            return torch.nanmean(rsq)
        elif multioutput == 'variance_weighted':
            return 1 - torch.nansum(_rss) / torch.nansum(_tss)
        else:
            raise ValueError(
                f"Invalid multioutput = {multioutput}"
            )


def f1_score(
    dataloader,
    model,
    target_data_idx=None,
    input_data_idx=None,
    multioutput='micro',
    targets_one_hot_encoded=True
):

    if dataloader is None:
        return None

    model.eval()

    _tp = 0
    _fp = 0
    _fn = 0

    with torch.no_grad():
        for data in dataloader:

            input_data, target_data = _extract_data(
                data,
                model,
                input_data_idx,
                target_data_idx
            )

            if not targets_one_hot_encoded:
                target_data = torch.nn.functional.one_hot(
                    target_data
                )

            predicts = torch.nn.functional.one_hot(
                argmax_last_dim(
                    model._slice_data_and_forward(input_data)
                ),
                num_classes=target_data.shape[-1]
            )

            _fp += _false_positive(
                predicts,
                target_data
            )

            _fn += _false_negative(
                predicts,
                target_data
            )

            _tp += _true_positive(
                predicts,
                target_data
            )

        if multioutput == 'micro':
            return _f1_score(_tp.sum(), _fp.sum(), _fn.sum())

        elif multioutput == 'macro':
            scores = _f1_score(_tp, _fp, _fn)
            scores = scores[~torch.isnan(scores)]
            return scores.mean()

        elif multioutput == 'weighted':
            scores = _f1_score(_tp, _fp, _fn)
            occurs = target_data.sum(axis=tuple(range(target_data.ndim - 1)))
            occurs = occurs / occurs.sum()
            _mask = ~torch.isnan(scores)
            scores = scores[_mask] * occurs[_mask]
            return scores.sum()

        elif multioutput is None:
            return _f1_score(_tp, _fp, _fn)

        else:
            raise ValueError(
                f"Invalid multioutput = {multioutput}"
            )


def _extract_data(
    data,
    model,
    input_data_idx,
    target_data_idx
):

    device = model._model_device

    if target_data_idx is None:
        target_data = to(data, device)
    else:
        target_data = to(data[target_data_idx], device)

    if input_data_idx is None:
        input_data = to(data, device)
    else:
        input_data = to(data[input_data_idx], device)

    output_data = model.output_data(target_data)

    return input_data, output_data
