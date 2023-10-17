import torch

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


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

    elif isinstance(prior_network, sparse.csr_matrix):
        labels = (None, None)
        data = torch.sparse_csr_tensor(
            prior_network.indptr,
            prior_network.indices,
            prior_network.data.astype(np.float32)
        ).to_dense()

    elif isinstance(prior_network, ad.AnnData):
        labels = (prior_network.obs_names, prior_network.var_names)
        data = _process_weights_to_tensor(
            prior_network.X
        )[0]

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
