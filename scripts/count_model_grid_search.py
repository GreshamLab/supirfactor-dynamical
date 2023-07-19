import argparse
import gc
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
    model_training,
    TruncRobustScaler
)

from inferelator.preprocessing import ManagePriors
from inferelator.postprocessing.model_metrics import MetricHandler
from inferelator.postprocessing import ResultsProcessor

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
    "--layer",
    dest="layer",
    help="Data AnnData file",
    metavar="FILE",
    default="X"
)

ap.add_argument(
    "--skip_static",
    dest="skip_static",
    help="Skip static model training",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--static_only",
    dest="static_only",
    help="Only static model training",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--shuffle",
    dest="shuffle",
    help="Shuffle prior labels",
    action='store_const',
    const=True,
    default=False
)

args = ap.parse_args()

data_file = args.datafile
prior_file = args.priorfile
gs_file = args.gsfile
_outfile = args.outfile

n_epochs = args.epochs
layer = args.layer
shuffle = args.shuffle

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

dropouts = 0.0

offsets = [
    None
]

models = [
    'rnn'
]

seqlens = [
    20
]

if args.skip_static:
    do_static = False
else:
    do_static = True

if args.static_only:
    static_only = True
else:
    static_only = False

seeds = list(range(111, 121))

validation_size = 0.25

print(f"Loading and processing data from {data_file}")
adata = ad.read(data_file)

if layer == "X":
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_per_cell(adata, min_counts=0)
    data = TruncRobustScaler(with_centering=False).fit_transform(adata.X)
else:
    data = TruncRobustScaler(with_centering=False).fit_transform(
        adata.layers[layer]
    ).astype(np.float32)

time_lookup = {
    'rapa': (adata.obs['program_rapa_time'].values, -10, 60, [-10, 0]),
    'cc': (adata.obs['program_cc_time'].values, 0, 88, None)
}

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

del adata
gc.collect()

prior = prior.loc[:, prior.sum(axis=0) > 0].copy()
g, k = prior.shape

print(f"Loading and processing gold standard from {gs_file}")
gs = pd.read_csv(
    gs_file,
    sep="\t",
    index_col=0
)

print(f"Splitting {validation_size} of data into validation")
train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=100
)

both_cols = [
    "Shuffle",
    "Layer",
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
    "Model_Type",
    "Time_Axis"
]

df_cols = both_cols + MetricHandler.get_metric('combined').all_names() + [
    "R2_training", "R2_validation"
]
loss_cols = both_cols + ["Loss_Type"] + list(range(1, n_epochs + 1))


def prep_loaders(
    static_size,
    dynamic_size,
    random_seed,
    dynamic_sequence_length,
    static_offset=None,
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

    if not do_static:
        static_tdl = None
        static_vdl = None

    elif static_offset is None or static_offset == 0:
        static_tdl = DataLoader(
            TimeDataset(
                _train,
                time_vector[train_idx],
                tmin,
                tmax,
                random_seed=random_seed
            ),
            batch_size=static_size,
            drop_last=True
        )

        static_vdl = DataLoader(
            TimeDataset(
                _test,
                time_vector[test_idx],
                tmin,
                tmax,
                random_seed=random_seed + 100
            ),
            batch_size=static_size,
            drop_last=True
        )

    else:
        static_tdl = DataLoader(
            TimeDataset(
                _train,
                time_vector[train_idx],
                tmin,
                tmax,
                1,
                sequence_length=1 + static_offset,
                random_seed=random_seed
            ),
            batch_size=static_size,
            drop_last=True
        )

        static_vdl = DataLoader(
            TimeDataset(
                _test,
                time_vector[test_idx],
                tmin,
                tmax,
                1,
                sequence_length=1 + static_offset,
                random_seed=random_seed + 100
            ),
            batch_size=static_size,
            drop_last=True
        )

    if static_only:
        dyn_tdl = None
        dyn_vdl = None

    else:
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
            batch_size=dynamic_size,
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
            batch_size=dynamic_size,
            drop_last=True
        )

    return static_tdl, static_vdl, dyn_tdl, dyn_vdl


def _results(
    result_leader,
    results,
    obj,
    model_name,
    tt
):

    _t_lead = [model_name, tt, "training"]
    _v_lead = [model_name, tt, "validation"]

    if obj is not None:

        result_line = [model_name, tt] + [
            results.all_scores[n]
            for n in results.all_names
        ] + [
            obj.training_r2,
            obj.validation_r2
        ]

        training_loss = _t_lead + obj.training_loss
        validation_loss = _v_lead + obj.validation_loss

        loss_lines = [
            result_leader + training_loss,
            result_leader + validation_loss
        ]

    else:
        result_line = [model_name, tt] + [
            results.all_scores[n]
            for n in results.all_names
        ] + [
            None,
            None
        ]

        loss_lines = None

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


def _combine_weights(*args):

    weights = args[0].copy()

    for a in args:
        weights += a

    weights /= len(args)

    return weights


def _process_combined(
    result_leader,
    inf_results,
    gs,
    pr,
    model_name
):

    _combined_weights = _combine_weights(
        inf_results['rapa'][0].betas[0],
        inf_results['cc'][0].betas[0]
    )

    r, _, _ = _results(
        result_leader,
        ResultsProcessor(
            [_combined_weights],
            [np.maximum(
                inf_results['rapa'][1],
                inf_results['cc'][1]
            )],
            metric="combined"
        ).summarize_network(
            None,
            gs,
            pr,
            full_model_betas=None
        ),
        None,
        model_name,
        'combined'
    )

    return r


def _train_cv(
    lr, wd, sb, db, in_drop, hl_drop, offset,
    seed, prior_cv, gs_cv, model, slen
):

    s_model = "static" if not static_meta else "static_meta"

    print(
        f"Training model (epochs: {n_epochs}, lr: {lr}, seed: {seed}, "
        f"weight_decay: {wd}, input_dropout: {in_drop}, "
        f"hidden_dropout: {hl_drop}, static_batch_size: {s_batch}, "
        f"dynamic_batch_size: {d_batch}, sequence length {slen}, "
        f"prediction_offset: {offset}, dynamic model type: {m}, "
        f"static model type: {s_model})"
    )

    result_leader = [
        shuffle,
        layer,
        lr,
        wd,
        seed,
        sb,
        db,
        slen,
        in_drop,
        hl_drop,
        offset,
        n_epochs
    ]

    if shuffle:
        prior_cv = ManagePriors.shuffle_priors(
            prior_cv,
            -1,
            seed
        )

    results = []
    loss_lines = []
    time_loss_lines = []

    inf_results = {k: {} for k in ['static', 'dynamic']}

    for tt in ['rapa', 'cc']:

        static_tdl, static_vdl, dyn_tdl, dyn_vdl = prep_loaders(
            sb,
            db,
            seed,
            slen,
            static_offset=offset,
            time_type=tt
        )

        torch.manual_seed(seed)

        if tt == 'rapa' and (static_only or do_static):
            static_obj, static_results, _erv = model_training(
                static_tdl,
                prior_cv,
                n_epochs,
                validation_dataloader=static_vdl,
                optimizer_params={'lr': lr, 'weight_decay': wd},
                gold_standard=gs_cv,
                input_dropout_rate=in_drop,
                hidden_dropout_rate=hl_drop,
                model_type=s_model,
                return_erv=True,
                output_relu=True
            )

            r, ll, _ = _results(
                result_leader,
                static_results,
                static_obj,
                s_model,
                tt
            )

            results.extend(r)
            loss_lines.extend(ll)
            time_loss_lines.append(None)

        if not static_only:
            dyn_obj, dynamic_results, _erv = model_training(
                dyn_tdl,
                prior_cv,
                n_epochs,
                validation_dataloader=dyn_vdl,
                optimizer_params={'lr': lr, 'weight_decay': wd},
                gold_standard=gs_cv,
                input_dropout_rate=in_drop,
                hidden_dropout_rate=hl_drop,
                prediction_length=False,
                model_type=model,
                return_erv=True,
                output_relu=True,
            )

            r, ll, time_loss = _results(
                result_leader,
                dynamic_results,
                dyn_obj,
                model,
                tt
            )

            inf_results['dynamic'][tt] = (dynamic_results, _erv)

            time_loss_lines.append(time_loss)
            results.extend(r)
            loss_lines.extend(ll)

        else:
            time_loss = None

    if not static_only:
        results.extend(
            _process_combined(
                result_leader,
                inf_results['dynamic'],
                gs_cv,
                prior_cv,
                model
            )
        )

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

for j, params in enumerate(
    itertools.product(
        offsets,
        weight_decays,
        learning_rates,
        batch_sizes,
        models,
        seqlens,
        seeds
    )
):

    if SLURM_N is not None:
        _j = j % SLURM_N
        if _j != SLURM_ID:
            continue

    in_drop = 0.5
    hl_drop = dropouts
    offset, wd, lr, (s_batch, d_batch), m, slen, i = params

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
        offset,
        i,
        prior_cv,
        gs_cv,
        m,
        slen
    )

    _write(results, outfile_results, _header)
    _write(losses, outfile_loss, _header)
    _write(time_loss, outfile_time_loss, _header)

    _header = False
