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
    model_training,
    process_results_to_dataframes,
    process_combined_results
)

from supirfactor_dynamical._utils._adata_load import (
    load_data_files_jtb_2023 as load_data,
    _write
)

from inferelator.preprocessing import ManagePriors

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

print("Predictive Supirfactor-rnn Grid Search")
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
    "-p",
    dest="priorfile",
    help="Prior Network TSV file",
    metavar="FILE",
    default=DEFAULT_PRIOR
)

ap.add_argument(
    "-g",
    dest="gsfile",
    help="Gold Standard Network TSV file",
    metavar="FILE",
    default=DEFAULT_GS
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

ap.add_argument(
    "--shuffle",
    dest="shuffle",
    help="Shuffle prior labels",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--shuffle_data",
    dest="shuffle_data",
    help="Shuffle counts to noise data",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--shuffle_times",
    dest="shuffle_time",
    help="Shuffle time labels on cells",
    action='store_const',
    const=True,
    default=False
)


args = ap.parse_args()

data_file = args.datafile
prior_file = args.priorfile
gs_file = args.gsfile

n_epochs = args.epochs

if args.shuffle:
    shuffle = 'Prior'
else:
    shuffle = 'False'

outfile_loss = args.outfile + "_LOSSES.tsv"
outfile_results = args.outfile + "_RESULTS.tsv"
outfile_time_loss = args.outfile + "_FINAL_LOSSES_OVER_TIME.tsv"

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

offsets = [
    10
]

seeds = list(range(111, 121))


data, time_lookup, prior, gs, count_scaling, velo_scaling = load_data(
    data_file,
    prior_file,
    gs_file,
    counts=True,
    velocity=True
)

validation_size = 0.25
print(f"Splitting {validation_size} of data into validation")
train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=100
)


def prep_loaders(random_seed, time_type):

    time_vector, tmin, tmax, shuffle_times = time_lookup[time_type]

    _train = data[train_idx, ...]
    _test = data[test_idx, ...]

    batch_size = 25

    slen = 20

    dynamic_tdl = DataLoader(
        TimeDataset(
            _train,
            time_vector[train_idx],
            tmin,
            tmax,
            1,
            sequence_length=slen,
            shuffle_time_vector=shuffle_times,
            random_seed=random_seed + 200
        ),
        batch_size=batch_size,
        drop_last=True
    )

    dynamic_vdl = DataLoader(
        TimeDataset(
            _test,
            time_vector[test_idx],
            tmin,
            tmax,
            1,
            shuffle_time_vector=shuffle_times,
            sequence_length=slen,
            random_seed=random_seed + 300
        ),
        batch_size=batch_size,
        drop_last=True
    )

    return dynamic_tdl, dynamic_vdl, slen


both_cols = [
    "Shuffle",
    "Decay_Model",
    "Learning_Rate",
    "Weight_Decay",
    "Seed",
    "Sequence_Length",
    "Output_Layer_Time_Offset",
    "Epochs",
    "Time_Axis"
]


def _train_cv(lr, wd, offset, seed, prior_cv, gs_cv):

    print(
        f"Training model (epochs: {n_epochs}, lr: {lr}, "
        f"seed: {seed}, weight_decay: {wd}, "
        f"prediction_offset: {offset})"
    )

    torch.manual_seed(seed)

    result_leader = [
        shuffle,
        False,
        lr,
        wd,
        seed,
        20,
        offset,
        n_epochs,
        None
    ]

    if shuffle == 'Prior':
        prior_cv = ManagePriors.shuffle_priors(
            prior_cv,
            -1,
            seed
        )

    results = []
    loss_lines = []
    time_loss_lines = []

    inf_results = []

    for tt in ['rapa', 'cc']:
        result_leader[-1] = tt
        tdl, vdl, _tlen = prep_loaders(seed, tt)

        dyn_obj, dynamic_results, _erv = model_training(
            tdl,
            prior_cv,
            n_epochs,
            validation_dataloader=vdl,
            optimizer_params={'lr': lr, 'weight_decay': wd},
            gold_standard=gs_cv,
            prediction_length=_tlen - offset,
            prediction_loss_offset=offset,
            model_type='biophysical',
            decay_model=False,
            count_scaling=count_scaling.scale_,
            velocity_scaling=velo_scaling.scale_,
            return_erv=True
        )

        inf_results.append((dynamic_results, _erv))

        r, ll, time_loss = process_results_to_dataframes(
            dyn_obj,
            dynamic_results,
            model_type='rnn',
            leader_columns=both_cols,
            leader_values=result_leader
        )

        results.append(r)
        loss_lines.append(ll)
        time_loss_lines.append(time_loss)

    result_leader[-1] = 'combined'
    results.append(
        process_combined_results(
            inf_results,
            gold_standard=gs_cv,
            prior_network=prior_cv,
            model_type='rnn',
            leader_columns=both_cols,
            leader_values=result_leader
        )
    )

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

for j, params in enumerate(
    itertools.product(
        offsets,
        weight_decays,
        learning_rates,
        seeds
    )
):

    offset, wd, lr, i = params

    p_cv, gs_cv = ManagePriors.cross_validate_gold_standard(
        prior,
        gs,
        0,
        0.25,
        i
    )

    p_cv = p_cv.reindex(
        prior.index,
        axis=0
    ).fillna(
        0
    ).astype(int)

    results, losses, time_loss = _train_cv(
        lr,
        wd,
        offset,
        i,
        p_cv,
        gs_cv,
    )

    _write(results, outfile_results, _header, both_cols)
    _write(losses, outfile_loss, _header, both_cols)
    _write(time_loss, outfile_time_loss, _header, both_cols)

    _header = False
