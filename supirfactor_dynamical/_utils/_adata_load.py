import gc
import anndata as ad
import numpy as np
import pandas as pd
import torch

from pandas.api.types import is_float_dtype

from sklearn.preprocessing import StandardScaler
from supirfactor_dynamical import TruncRobustScaler

from inferelator.preprocessing import ManagePriors
from inferelator.preprocessing.simulate_data import (
    _sim_ints,
    _sim_float
)

_TIME_LOOKUP = {
    'rapa': [-10, 60, [-10, 0]],
    'cc': [0, 88, None]
}

_SHUFFLE_TIMES = {
    'rapa': [-10, 88],
    'cc': [0, 88]
}


def load_standard_data(
    data_file=None,
    file_type='h5ad',
    prior_file=None,
    gold_standard_file=None,
    data_layer='X',
    scale_data=True,
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
        adata = ad.read(data_file, **kwargs)
        var_names = adata.var_names
        count_data = _get_data(adata, data_layer)

        del adata
        gc.collect()

    elif data_file is not None and file_type == 'tsv':

        print(f"Loading and processing data from {data_file}")
        count_data = pd.read_csv(
            data_file,
            sep="\t",
            **kwargs
        )
        var_names = count_data.columns
        count_data = count_data.values

    elif data_file is None:
        count_data = None
    else:
        raise ValueError(
            f"Unknown file_type {file_type}"
        )

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

    return count_data, count_scaling, prior, gs


def load_data_files_jtb_2023(
    adata_file,
    prior_file=None,
    gold_standard_file=None,
    counts=True,
    velocity=False,
    decay_velocity=False,
    shuffle_time=False,
    shuffle_data=False,
    untreated_only=False,
    counts_layer='X',
    velocity_layers=('rapamycin_velocity', 'cell_cycle_velocity'),
    decay_velocity_layers=('decay_constants', 'denoised')
):
    """
    Helper to load the data used in the associated manuscript:
    https://www.biorxiv.org/content/10.1101/2023.09.21.558277v1

    :param adata_file: Prepared adata file with counts in counts_layer
        saved as an '.h5ad' file.
    :type adata_file: str
    :param prior_file: Priors TSV file, defaults to None
    :type prior_file: str, optional
    :param gold_standard_file: Gold standard TSV file, defaults to None
    :type gold_standard_file: str, optional
    :param counts: Return array contains counts,
        defaults to True
    :type counts: bool, optional
    :param velocity: Return array contains preprocessed velocities,
        defaults to False
    :type velocity: bool, optional
    :param decay_velocity: Return array contains preprocessed
        decay-specific velocities, calculated from decay_constants and
        denoised expression, defaults to False
    :type decay_velocity: bool, optional
    :param shuffle_time: Shuffle time metadata, defaults to False
    :type shuffle_time: bool, optional
    :param shuffle_data: Turn array data into white noise,
        defaults to False
    :type shuffle_data: bool, optional
    :param untreated_only: Return only untreated cells,
        in pools 1 & 2, defaults to False
    :type untreated_only: bool, optional
    :param counts_layer: Layer with count data,
        defaults to 'X'
    :type counts_layer: str, optional
    :param velocity_layers: Layer with preprocessed velocity data,
        defaults to ('rapamycin_velocity', 'cell_cycle_velocity')
    :type velocity_layers: tuple, optional
    :param decay_velocity_layers: Layers with preprocessed decay rates,
        and expression data, defaults to ('decay_constants', 'denoised')
    :type decay_velocity_layers: tuple, optional
    :return:
        Array data,
        Time lookup dict,
        Prior dataframe,
        Gold standard dataframe,
        Count scaler,
        Velocity scaler
    :rtype: _type_
    """
    count_scaling = TruncRobustScaler(with_centering=False)
    velo_scaling = TruncRobustScaler(with_centering=False)

    print(f"Loading and processing data from {adata_file}")
    adata = ad.read(adata_file)

    if untreated_only:
        adata = adata[adata.obs['Pool'].isin([1, 2]), :].copy()

    if counts:

        if shuffle_data:
            count_data = _shuffle_data(
                adata.layers['counts']
            ).astype(float)
        else:
            count_data = _get_data(adata, counts_layer)

        data = [count_scaling.fit_transform(count_data).A]

    else:
        data = []

    if velocity:

        velocity_data = _get_data(adata, velocity_layers)

        if shuffle_data:
            velocity_data = _shuffle_data(
                velocity_data,
                sim_counts=False,
                sim_velo=True
            )

        data.append(
            velo_scaling.fit_transform(
                velocity_data
            )
        )
    elif decay_velocity:
        velo_scaling.fit(
            _get_data(adata, velocity_layers)
        )

    if decay_velocity:

        velocity_data = _get_data(adata, decay_velocity_layers, np.multiply)
        velocity_data *= -1

        if shuffle_data:
            velocity_data = _shuffle_data(
                velocity_data,
                sim_counts=False,
                sim_velo=True
            )

            velocity_data = np.minimum(
                velocity_data, 0
            )

        data.append(
            velo_scaling.transform(
                velocity_data
            )
        )

    if len(data) > 1:
        data = np.stack(data, axis=-1)
    else:
        data = data[0]

    time_lookup = {
        'rapa': [adata.obs['program_rapa_time'].values] + _TIME_LOOKUP['rapa'],
        'cc': [adata.obs['program_cc_time'].values] + _TIME_LOOKUP['cc']
    }

    if shuffle_time:
        for k in _SHUFFLE_TIMES.keys():
            time_lookup[k][-1] = _SHUFFLE_TIMES[k]

    _, _, prior, gs = load_standard_data(
        data_file=None,
        prior_file=prior_file,
        gold_standard_file=gold_standard_file
    )

    return (
        data,
        time_lookup,
        prior,
        gs,
        count_scaling,
        velo_scaling
    )


def _get_data(
    adata,
    layers,
    agg_func=np.add,
    densify=False,
    **kwargs
):

    if isinstance(layers, (tuple, list)):
        _output = _get_data(adata, layers[0]).copy()
        for layer in layers[1:]:
            agg_func(
                _output,
                _get_data(adata, layer),
                out=_output,
                **kwargs
            )

    elif layers == 'X':
        _output = adata.X

    else:
        _output = adata.layers[layers]

    if densify:
        try:
            _output = _output.A
        except AttributeError:
            pass

    return _output


def _shuffle_prior(prior, seed=100):

    return ManagePriors.shuffle_priors(
        prior,
        -1,
        seed
    )


def _shuffle_data(
    x,
    sim_counts=True,
    sim_velo=False
):

    if sim_counts:
        pvec = x.sum(axis=0)

        try:
            pvec = pvec.A1
        except AttributeError:
            pass

        return _sim_ints(
            pvec / pvec.sum(),
            np.full(pvec.shape, 3099, dtype=int),
            sparse=hasattr(x, 'A')
        )

    elif sim_velo:
        ss = StandardScaler().fit(x)

        return _sim_float(
            ss.mean_,
            np.sqrt(ss.var_),
            x.shape[0]
        )


def _write(
    df,
    filename,
    _header,
    leader_columns
):
    if df is None:
        return

    # Float precision
    for col in df.columns[~df.columns.isin(leader_columns)]:
        if is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.6f}")

    df.to_csv(
        filename,
        sep="\t",
        index=False,
        header=_header,
        mode="a"
    )
