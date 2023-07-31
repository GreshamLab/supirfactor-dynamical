import argparse
import sys
import itertools
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from pandas.api.types import is_float_dtype
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    TruncRobustScaler
)

from supirfactor_dynamical.models.decay_model import DecayModule


DEFAULT_PATH = "/mnt/ceph/users/cjackson/inferelator/data/RAPA/"
DEFAULT_PRIOR = os.path.join(
    DEFAULT_PATH,
    "JOINT_PRIOR_20230701.tsv.gz"
)
DEFAULT_GS = os.path.join(
    DEFAULT_PATH,
    "gold_standard.tsv.gz"
)
DEFAULT_DATA = os.path.join(
    DEFAULT_PATH,
    "2021_INFERELATOR_DATA.h5ad"
)

SLURM_ID = os.environ.get('SLURM_ARRAY_TASK_ID', None)
SLURM_N = os.environ.get('SLURM_ARRAY_TASK_COUNT', None)

print("Supirfactor-rnn Grid Search")
print(" ".join(sys.argv))

ap = argparse.ArgumentParser(description="SUPIRFACTOR-RNN Parameter Search")

ap.add_argument(
    "-o",
    "-O",
    dest="outfile",
    help="Output TSV file prefix",
    metavar="FILE",
    required=True
)

ap.add_argument(
    "-f",
    dest="datafile",
    help="Data AnnData file",
    metavar="FILE",
    default=DEFAULT_DATA
)

ap.add_argument(
    "--lr",
    dest="lr",
    help="Search Learning Rates",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--wd",
    dest="wd",
    help="Search Weight Decays",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--epochs",
    dest="epochs",
    help="NUM Epochs",
    metavar="NUM",
    type=int,
    default=200
)

args = ap.parse_args()

data_file = args.datafile
_outfile = args.outfile

n_epochs = args.epochs

if SLURM_ID is None:
    outfile_loss = _outfile + "_LOSSES.tsv"
    outfile_results = _outfile + "_RESULTS.tsv"
    outfile_time_loss = _outfile + "_FINAL_LOSSES_OVER_TIME.tsv"
else:
    outfile_loss = _outfile + f"_LOSSES_{SLURM_ID}.tsv"
    outfile_results = _outfile + f"_RESULTS_{SLURM_ID}.tsv"
    outfile_time_loss = _outfile + f"_FINAL_LOSSES_OVER_TIME_{SLURM_ID}.tsv"

if args.lr:
    learning_rates = [
        1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6
    ][::-1]
else:
    learning_rates = [
        5e-5
    ]

if args.wd:
    weight_decays = [
        1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7
    ][::-1]
else:
    weight_decays = [
        1e-7
    ]

batch_sizes = (250, 20)
sequence_lengths = [
    20
]

seeds = list(range(111, 121))

validation_size = 0.25

count_scaling = TruncRobustScaler(with_centering=False)
velo_scaling = TruncRobustScaler(with_centering=False)

print(f"Loading and processing data from {data_file}")
adata = ad.read(data_file)

adata.X = adata.X.astype(np.float32)
sc.pp.normalize_per_cell(adata, min_counts=0)
data = np.stack(
    (
        count_scaling.fit_transform(
            adata.X
        ).A,
        velo_scaling.fit_transform(
            adata.layers['rapamycin_velocity'] +
            adata.layers['cell_cycle_velocity']
        ),
        adata.layers['decay_constants']
    ),
    axis=-1
)

_shuffle_times = ([-10, 0], None)

time_lookup = {
    'rapa': (
        adata.obs['program_rapa_time'].values,
        -10, 60, _shuffle_times[0]
    ),
    'cc': (
        adata.obs['program_cc_time'].values,
        0, 88, _shuffle_times[1]
    )
}

print(f"Splitting {validation_size} of data into validation")
train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=100
)

both_cols = [
    "Learning_Rate",
    "Weight_Decay",
    "Seed",
    "Epochs",
    "Model_Type",
    "Time_Axis"
]

df_cols = both_cols + [
    "R2_training", "R2_validation"
]
loss_cols = both_cols + ["Loss_Type"] + list(range(1, n_epochs + 1))


def prep_loaders(
    random_seed,
    dynamic_sequence_length,
    time_type='rapa'
):

    time_vector, tmin, tmax, shuffle_times = time_lookup[time_type]

    _train = data[train_idx, :]
    _test = data[test_idx, :]

    try:
        _train = _train.A
        _test = _test.A
    except AttributeError:
        pass

    dyn_tdl = DataLoader(
        TimeDataset(
            _train,
            time_vector[train_idx],
            tmin,
            tmax,
            1,
            sequence_length=dynamic_sequence_length,
            shuffle_time_vector=shuffle_times,
            random_seed=random_seed + 200
        ),
        batch_size=batch_sizes[1],
        drop_last=True
    )

    dyn_vdl = DataLoader(
        TimeDataset(
            _test,
            time_vector[test_idx],
            tmin,
            tmax,
            1,
            shuffle_time_vector=shuffle_times,
            sequence_length=dynamic_sequence_length,
            random_seed=random_seed + 300
        ),
        batch_size=batch_sizes[1],
        drop_last=True
    )

    return dyn_tdl, dyn_vdl


def _results(
    result_leader,
    obj,
    model_name,
    tt
):

    _t_lead = [model_name, tt, "training"]
    _v_lead = [model_name, tt, "validation"]

    result_line = [model_name, tt] + [
        obj.training_r2,
        obj.validation_r2
    ]

    training_loss = _t_lead + obj.training_loss
    validation_loss = _v_lead + obj.validation_loss

    loss_lines = [
        result_leader + training_loss,
        result_leader + validation_loss
    ]

    results = [result_leader + result_line]

    try:
        if obj is not None and hasattr(obj, "training_r2_over_time"):

            _n = len(obj.training_r2_over_time)
            _cols = both_cols + ["Loss_Type"] + list(range(0, _n))

            time_dependent_loss = pd.DataFrame(
                [
                    result_leader + _t_lead + obj.training_r2_over_time,
                    result_leader + _v_lead + obj.validation_r2_over_time
                ],
                columns=_cols
            )

        else:
            time_dependent_loss = None

    except TypeError:
        time_dependent_loss = None

    return results, loss_lines, time_dependent_loss


def _train_cv(
    lr, wd, seed, slen
):

    print(
        f"Training model (epochs: {n_epochs}, lr: {lr}, seed: {seed}, "
        f"weight_decay: {wd}, sequence_length: {slen})"
    )

    result_leader = [
        lr,
        wd,
        seed,
        n_epochs
    ]

    results = []
    loss_lines = []
    time_loss_lines = []

    for tt in ['rapa', 'cc']:

        dyn_tdl, dyn_vdl = prep_loaders(
            seed,
            slen,
            time_type=tt
        )

        torch.manual_seed(seed)

        model_obj = DecayModule(
            data.shape[1]
        )

        model_obj.set_scaling(
            count_scaling=count_scaling.scale_,
            velocity_scaling=velo_scaling.scale_,
        )

        model_obj.train_model(
            dyn_tdl,
            n_epochs,
            validation_dataloader=dyn_vdl,
            optimizer={'lr': lr, 'weight_decay': wd},
        )

        r, ll, time_loss = _results(
            result_leader,
            model_obj,
            'decay',
            tt
        )

        time_loss_lines.append(time_loss)
        results.extend(r)
        loss_lines.extend(ll)

    results = pd.DataFrame(
        results,
        columns=df_cols
    )

    losses = pd.DataFrame(
        loss_lines,
        columns=loss_cols
    )

    try:
        time_loss_lines = pd.concat(time_loss_lines)
    except ValueError:
        time_loss_lines = None

    return results, losses, time_loss


def _write(df, filename, _header):
    if df is None:
        return

    # Float precision
    for col in df.columns[~df.columns.isin(both_cols)]:
        if is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.6f}")

    df.to_csv(
        filename,
        sep="\t",
        index=False,
        header=_header,
        mode="a"
    )


_header = True

for j, (wd, lr, i, slen) in enumerate(
    itertools.product(
        weight_decays,
        learning_rates,
        seeds,
        sequence_lengths
    )
):

    if SLURM_N is not None:
        _j = j % SLURM_N
        if _j != SLURM_ID:
            continue

    results, losses, time_loss = _train_cv(
        lr,
        wd,
        i,
        slen
    )

    _write(results, outfile_results, _header)
    _write(losses, outfile_loss, _header)
    _write(time_loss, outfile_time_loss, _header)

    _header = False
