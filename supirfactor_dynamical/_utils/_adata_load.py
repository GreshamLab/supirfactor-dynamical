import anndata as ad
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype

from supirfactor_dynamical import TruncRobustScaler


def load_data_files_jtb_2023(
    adata_file,
    prior_file=None,
    gold_standard_file=None,
    counts=True,
    velocity=False,
    decay_velocity=False
):
    count_scaling = TruncRobustScaler(with_centering=False)
    velo_scaling = TruncRobustScaler(with_centering=False)

    print(f"Loading and processing data from {adata_file}")
    adata = ad.read(adata_file)

    if counts:
        data = [
                count_scaling.fit_transform(
                    adata.X
                ).A
        ]
    else:
        data = []

    if velocity:
        data.append(
            velo_scaling.fit_transform(
                adata.layers['rapamycin_velocity'] +
                adata.layers['cell_cycle_velocity']
            )
        )
    elif decay_velocity:
        velo_scaling.fit(
            adata.layers['rapamycin_velocity'] +
            adata.layers['cell_cycle_velocity']
        )

    if decay_velocity:
        data.append(
            velo_scaling.transform(
                np.multiply(
                    adata.layers['decay_constants'] * -1,
                    adata.layers['denoised']
                )
            )
        )

    if len(data) > 1:
        data = np.stack(data, axis=-1)
    else:
        data = data[0]

    time_lookup = {
        'rapa': (adata.obs['program_rapa_time'].values, -10, 60, [-10, 0]),
        'cc': (adata.obs['program_cc_time'].values, 0, 88, None)
    }

    if prior_file is not None:
        print(f"Loading and processing priors from {prior_file}")
        prior = pd.read_csv(
            prior_file,
            sep="\t",
            index_col=0
        ).reindex(
            adata.var_names,
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

    return (
        data,
        time_lookup,
        prior,
        gs,
        count_scaling,
        velo_scaling
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
