import gc
import anndata as ad
import numpy as np
import pandas as pd
import torch
import scanpy as sc

from supirfactor_dynamical._utils._trunc_robust_scaler import (
    TruncRobustScaler
)


def load_standard_data(
    data_file=None,
    file_type='h5ad',
    prior_file=None,
    gold_standard_file=None,
    data_layer='X',
    depth_normalize_data=True,
    normalization_depth=None,
    scale_data=True,
    genes=None,
    **kwargs
):
    """
    Load data into tensors and dataframes

    :param data_file: Data file, defaults to None
    :type data_file: str, optional
    :param file_type: Data file type, 'h5ad' and 'tsv' are options,
        defaults to 'h5ad'
    :type file_type: str, optional
    :param prior_file: Prior TSV file, defaults to None
    :type prior_file: str, optional
    :param gold_standard_file: Gold standard TSV file, defaults to None
    :type gold_standard_file: _type_, optional
    :param data_layer: Data layer from an h5ad file,
        defaults to 'X'
    :type data_layer: str, optional
    :param scale_data: Scale data with TruncRobustScaler,
        defaults to True
    :type scale_data: bool, optional
    :return: Data, scaler, prior, gold standard
    :rtype: torch.Tensor, TruncRobustScaler, pd.DataFrame, pd.DataFrame
    """

    if data_file is not None and file_type == 'h5ad':

        print(f"Loading and processing data from {data_file}")
        adata = ad.read_h5ad(data_file, **kwargs)

        if genes is not None:
            adata = adata[:, genes]

        var_names = adata.var_names
        count_data = _get_data_from_ad(adata, data_layer)

        del adata
        gc.collect()

    elif data_file is not None and file_type == 'tsv':

        print(f"Loading and processing data from {data_file}")
        count_data = pd.read_csv(
            data_file,
            sep="\t",
            **kwargs
        )

        if genes is not None:
            count_data = count_data[:, genes]

        var_names = count_data.columns
        count_data = count_data.values

    elif data_file is None:
        var_names = genes
        count_data = None
    else:
        raise ValueError(
            f"Unknown file_type {file_type}"
        )

    if depth_normalize_data and count_data is not None:
        count_data = sc.pp.normalize_total(
            ad.AnnData(count_data),
            target_sum=normalization_depth
        ).X

    if count_data is not None and scale_data:
        count_scaling = TruncRobustScaler(with_centering=False)
        count_data = count_scaling.fit_transform(count_data)
    else:
        count_scaling = None

    if prior_file is not None:
        print(f"Loading and processing priors from {prior_file}")
        prior = pd.read_csv(
            prior_file,
            sep="\t",
            index_col=0
        ).reindex(
            var_names,
            axis=0
        ).fillna(
            0
        ).astype(int)

        prior = prior.loc[:, prior.sum(axis=0) > 0].copy()
    else:
        prior = None

    if gold_standard_file is not None:
        print(f"Loading gold standard from {gold_standard_file}")
        gs = pd.read_csv(
            gold_standard_file,
            sep="\t",
            index_col=0
        )
    else:
        gs = None

    if count_data is not None:
        count_data = torch.Tensor(count_data)

        print(
            f"Loaded data {count_data.shape} from {data_file}"
        )

    return count_data, count_scaling, prior, gs


def _get_data_from_ad(
    adata,
    layers,
    agg_func=np.add,
    densify=False,
    **kwargs
):

    if isinstance(layers, (tuple, list)):
        _output = _get_data_from_ad(adata, layers[0], densify=densify).copy()
        for layer in layers[1:]:
            agg_func(
                _output,
                _get_data_from_ad(adata, layer, densify=densify),
                out=_output,
                **kwargs
            )

    elif layers == 'X':
        _output = adata.X

    else:
        _output = adata.layers[layers]

    if densify:
        try:
            _output = _output.toarray()
        except AttributeError:
            pass

    return _output
