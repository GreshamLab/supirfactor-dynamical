import torch

import numpy as np
import pandas as pd


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
