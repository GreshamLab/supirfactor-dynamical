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
    SupirFactorBiophysical,
    process_results_to_dataframes,
    process_combined_results
)

from inferelator.preprocessing import ManagePriors

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
TRAINED_DECAY = os.path.join(
    "/mnt/ceph/users/cjackson/supirfactor_trs_decay",
    "RAPA_DECAY_MODEL.h5"
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
    "--pretrain_decay",
    dest="pretrained_decay",
    help="Use pretrained decay",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--tune_decay",
    dest="tuned_decay",
    help="Tune decay model",
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
prior_file = args.priorfile
gs_file = args.gsfile
_outfile = args.outfile

n_epochs = args.epochs

static_meta = True

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

batch_sizes = [
    (250, 20)
]

dropouts = [
    (0.5, 0.0)
]

seqlens = [
    20
]

tuned_delay = 100 if args.tuned_decay else n_epochs

seeds = list(range(111, 121))

validation_size = 0.25

data, time_lookup, prior, gs, count_scaling, velo_scaling = load_data(
    data_file,
    prior_file,
    gs_file,
    counts=True,
    velocity=True,
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

    _train = data[train_idx, ...]
    _test = data[test_idx, ...]

    seq_len = 11
    batch_size = 25

    dyn_tdl = DataLoader(
        TimeDataset(
            _train,
            time_vector[train_idx],
            tmin,
            tmax,
            1,
            sequence_length=seq_len,
            shuffle_time_vector=shuffle_times,
            random_seed=random_seed + 200
        ),
        batch_size=batch_size,
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
            sequence_length=seq_len,
            random_seed=random_seed + 300
        ),
        batch_size=batch_size,
        drop_last=True
    )

    return dyn_tdl, dyn_vdl


model_name = "Joint"
if args.pretrained_decay:
    model_name = model_name + "_Pretrained"

if args.tuned_decay:
    model_name = model_name + "_Tuned"


both_cols = [
    "Decay_Model",
    "Learning_Rate",
    "Weight_Decay",
    "Seed",
    "Static_Batch_Size",
    "Dynamic_Batch_Size",
    "Sequence_Length",
    "Input_Dropout",
    "Hidden_Layer_Dropout",
    "Output_Layer_Time_Offset",
    "Epochs",
    "Time_Axis"
]


def _train_cv(
    lr, wd, sb, db, in_drop, hl_drop,
    seed, prior_cv, gs_cv, slen
):

    print(
        f"Training Velocity model (epochs: {n_epochs}, lr: {lr}, "
        f"seed: {seed}, weight_decay: {wd}, input_dropout: {in_drop}, "
        f"hidden_dropout: {hl_drop}, static_batch_size: {s_batch}, "
        f"dynamic_batch_size: {d_batch}, sequence length {slen})"
    )

    result_leader = [
        model_name,
        lr,
        wd,
        seed,
        sb,
        db,
        slen,
        in_drop,
        hl_drop,
        0,
        n_epochs,
        None
    ]

    results = []
    loss_lines = []
    time_loss_lines = []

    inf_results = []

    for tt in ['rapa', 'cc']:

        result_leader[-1] = tt

        dyn_tdl, dyn_vdl = prep_loaders(
            seed,
            time_type=tt
        )

        torch.manual_seed(seed)

        dyn_obj = SupirFactorBiophysical(
            prior_cv,
            decay_model=TRAINED_DECAY if args.pretrained_decay else None,
            input_dropout_rate=in_drop,
            hidden_dropout_rate=hl_drop,
            decay_epoch_delay=tuned_delay if args.pretrained_decay else 0,
            output_activation='softplus'
        )

        dyn_obj, dynamic_results, _erv = model_training(
            dyn_tdl,
            prior_cv,
            n_epochs,
            validation_dataloader=dyn_vdl,
            optimizer_params={'lr': lr, 'weight_decay': wd},
            gold_standard=gs_cv,
            model_type=dyn_obj,
            return_erv=True,
            count_scaling=count_scaling.scale_,
            velocity_scaling=velo_scaling.scale_
        )

        inf_results.append((dynamic_results, _erv))

        r, ll, time_loss = process_results_to_dataframes(
            dyn_obj,
            dynamic_results,
            model_type='biophysical',
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
            model_type='biophysical',
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
        weight_decays,
        learning_rates,
        batch_sizes,
        dropouts,
        seqlens,
        seeds
    )
):

    if SLURM_N is not None:
        _j = j % SLURM_N
        if _j != SLURM_ID:
            continue

    wd, lr, (s_batch, d_batch), (in_drop, hl_drop), slen, i = params

    prior_cv, gs_cv = ManagePriors.cross_validate_gold_standard(
        prior,
        gs,
        0,
        0.25,
        i
    )

    prior_cv = prior_cv.reindex(
        prior.index,
        axis=0
    ).fillna(
        0
    ).astype(int)

    results, losses, time_loss = _train_cv(
        lr,
        wd,
        s_batch,
        d_batch,
        in_drop,
        hl_drop,
        i,
        prior_cv,
        gs_cv,
        slen
    )

    _write(results, outfile_results, _header, both_cols)
    _write(losses, outfile_loss, _header, both_cols)
    _write(time_loss, outfile_time_loss, _header, both_cols)

    _header = False
