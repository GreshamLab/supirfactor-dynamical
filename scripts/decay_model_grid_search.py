import argparse
import sys
import itertools
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    process_results_to_dataframes
)

from supirfactor_dynamical.models.decay_model import DecayModule

from supirfactor_dynamical._utils._adata_load import (
    load_data_files_jtb_2023 as load_data,
    _write
)

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
    "--model_width",
    dest="mw",
    help="Search Hidden Layer Widths",
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

if args.mw:
    model_widths = [
        100, 50, 25, 20, 15, 10, 5, 4, 3, 2, 1
    ]
else:
    model_widths = [50]

seeds = list(range(111, 121))

validation_size = 0.25

data, time_lookup, _, _, count_scaling, velo_scaling = load_data(
    data_file,
    None,
    None,
    counts=True,
    decay_velocity=True
)

print(f"Splitting {validation_size} of data into validation")
train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=100
)


def prep_loaders(
    random_seed,
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
            sequence_length=10,
            shuffle_time_vector=shuffle_times,
            random_seed=random_seed + 200
        ),
        batch_size=20,
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
            sequence_length=10,
            random_seed=random_seed + 300
        ),
        batch_size=20,
        drop_last=True
    )

    return dyn_tdl, dyn_vdl


both_cols = [
    "Learning_Rate",
    "Weight_Decay",
    "Decay_Model_Width",
    "Seed",
    "Epochs",
    "Time_Axis"
]


def _train_cv(
    lr, wd, seed, mw
):

    print(
        f"Training model (epochs: {n_epochs}, lr: {lr}, seed: {seed}, "
        f"weight_decay: {wd}, model_width: {mw})"
    )

    result_leader = [
        lr,
        wd,
        mw,
        seed,
        n_epochs,
        None
    ]

    results = []
    loss_lines = []
    time_loss_lines = []

    for tt in ['rapa', 'cc']:
        result_leader[-1] = tt

        dyn_tdl, dyn_vdl = prep_loaders(
            seed,
            time_type=tt
        )

        torch.manual_seed(seed)

        model_obj = DecayModule(
            data.shape[1],
            k=mw,
            input_dropout_rate=0.5,
            hidden_dropout_rate=0
        )

        model_obj.set_scaling(
            count_scaling=count_scaling.scale_,
            velocity_scaling=velo_scaling.scale_
        )

        model_obj.train_model(
            dyn_tdl,
            n_epochs,
            validation_dataloader=dyn_vdl,
            optimizer={'lr': lr, 'weight_decay': wd}
        )

        r, ll, time_loss = process_results_to_dataframes(
            model_obj,
            None,
            model_type='decay',
            leader_columns=both_cols,
            leader_values=result_leader
        )

        results.append(r)
        loss_lines.append(ll)
        time_loss_lines.append(time_loss)

    results = pd.concat(
        results
    )

    losses = pd.concat(
        loss_lines
    )

    try:
        time_loss_lines = pd.concat(time_loss_lines)
    except ValueError:
        time_loss_lines = None

    return results, losses, time_loss_lines


_header = True

for j, (wd, lr, i, mw) in enumerate(
    itertools.product(
        weight_decays,
        learning_rates,
        seeds,
        model_widths
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
        mw
    )

    _write(results, outfile_results, _header, both_cols)
    _write(losses, outfile_loss, _header, both_cols)
    _write(time_loss, outfile_time_loss, _header, both_cols)

    _header = False
