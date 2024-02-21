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


def r2_score(
    dataloader,
    model,
    target_data_idx=None,
    input_data_idx=None,
    multioutput='uniform_average'
):

    if dataloader is None:
        return None

    model.eval()

    _rss = 0
    _tss = 0

    with torch.no_grad():
        for data in dataloader:

            if target_data_idx is None:
                target_data = data
            else:
                target_data = data[target_data_idx]

            if input_data_idx is None:
                input_data = data
            else:
                input_data = data[input_data_idx]

            output_data = model.output_data(target_data)

            _rss += _calculate_rss(
                output_data,
                model._slice_data_and_forward(input_data),
            )

            _tss += _calculate_tss(
                output_data
            )

        if multioutput == 'raw_values':
            return _calculate_r2(_rss, _tss)
        elif multioutput == 'uniform_average':
            return _calculate_r2(_rss, _tss).mean()
        elif multioutput == 'uniform_truncated_average':
            rsq = _calculate_r2(_rss, _tss)
            rsq[rsq < 0] = 0
            return rsq.mean()
        elif multioutput == 'variance_weighted':
            return 1 - _rss.sum() / _tss.sum()


def f1_score(
    dataloader,
    model,
    target_data_idx=None,
    input_data_idx=None,
    multioutput='uniform_average'
):

    if dataloader is None:
        return None

    model.eval()

    _tp = 0
    _fp = 0
    _fn = 0

    with torch.no_grad():
        for data in dataloader:

            if target_data_idx is None:
                target_data = data
            else:
                target_data = data[target_data_idx]

            if input_data_idx is None:
                input_data = data
            else:
                input_data = data[input_data_idx]

            target_data = model.output_data(target_data)
            predicts = model._slice_data_and_forward(input_data)

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