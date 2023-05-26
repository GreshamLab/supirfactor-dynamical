import torch

import numpy as np
import pandas as pd

from inferelator.postprocessing import ResultsProcessor


def _process_weights_to_tensor(
    prior_network,
    transpose=True
):

    if isinstance(prior_network, pd.DataFrame):
        labels = (prior_network.index, prior_network.columns)
        data = torch.tensor(
            prior_network.values,
            dtype=torch.float32
        )

    elif isinstance(prior_network, np.ndarray):
        labels = (None, None)
        data = torch.tensor(
            prior_network,
            dtype=torch.float32
        )

    else:
        labels = (None, None)
        data = prior_network

    if transpose:
        data = torch.transpose(
            data,
            0,
            1
        )

    return data, labels


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


def evaluate_results(
    model_weights,
    model_erv,
    prior_network,
    gold_standard_network
):

    return ResultsProcessor(
        [model_weights],
        [model_erv],
        metric="combined"
    ).summarize_network(
        None,
        gold_standard_network,
        prior_network,
        full_model_betas=model_weights
    )
