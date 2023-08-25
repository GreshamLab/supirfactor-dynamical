import argparse
import gc
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    TruncRobustScaler,
    model_training
)

from supirfactor_dynamical.train import pretrain_and_tune_dynamic_model

DEFAULT_PATH = "/mnt/ceph/users/cjackson/inferelator/data/RAPA/"
DEFAULT_PRIOR = os.path.join(
    DEFAULT_PATH,
    "JOINT_PRIOR_20230701.tsv.gz"
)
DEFAULT_DATA = os.path.join(
    DEFAULT_PATH,
    "2021_INFERELATOR_DATA.h5ad"
)

print("Supirfactor-rnn Full Training")
print(" ".join(sys.argv))

ap = argparse.ArgumentParser(description="SUPIRFACTOR-RNN Full Build")

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
    default=DEFAULT_PRIOR
)

ap.add_argument(
    "--epochs",
    dest="epochs",
    help="NUM Epochs",
    metavar="NUM",
    type=int,
    default=2000
)

args = ap.parse_args()

data_file = args.datafile
prior_file = args.priorfile
gs_file = args.gsfile

static_outfile = args.outfile + "_STATIC_MODEL.h5"
dynamic_outfile = args.outfile + "_RNN_MODEL.h5"

static_result_dir = args.outfile + "_STATIC"
dynamic_result_dir = args.outfile + "_RNN"

n_epochs = args.epochs

validation_size = 0.25
random_seed = 1800

print(f"Loading and processing data from {data_file}")
adata = ad.read(data_file)

count_scaling = TruncRobustScaler(with_centering=False)

data = count_scaling.fit_transform(
    adata.X
).A

time_vector = adata.obs['program_rapa_time'].values

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

print(f"Loading and processing gold standard from {gs_file}")
gs = pd.read_csv(
    gs_file,
    sep="\t",
    index_col=0
)

del adata
gc.collect()

train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=random_seed
)

_train = data[train_idx, :]
_test = data[test_idx, :]

print(f"Training on {_train.shape} and validating on {_test.shape}")

static_tdl = DataLoader(
    TimeDataset(
        _train,
        time_vector[train_idx],
        -10,
        60,
        1,
        sequence_length=2,
        random_seed=random_seed,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=100,
    drop_last=True
)

static_vdl = DataLoader(
    TimeDataset(
        _test,
        time_vector[test_idx],
        -10,
        60,
        1,
        sequence_length=2,
        random_seed=random_seed + 100,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=100,
    drop_last=True
)

dyn_tdl = DataLoader(
    TimeDataset(
        _train,
        time_vector[train_idx],
        -10,
        60,
        1,
        sequence_length=20,
        random_seed=random_seed + 200,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=20,
    drop_last=True
)

dyn_vdl = DataLoader(
    TimeDataset(
        _test,
        time_vector[test_idx],
        -10,
        60,
        1,
        sequence_length=20,
        random_seed=random_seed + 300,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=20,
    drop_last=True
)

dynamic_pretrain = DataLoader(
    TimeDataset(
        _train,
        time_vector[train_idx],
        -10,
        60,
        1,
        sequence_length=11,
        shuffle_time_vector=[-10, 0],
        random_seed=random_seed + 200
    ),
    batch_size=20,
    drop_last=True
)

dynamic_preval = DataLoader(
    TimeDataset(
        _test,
        time_vector[test_idx],
        -10,
        60,
        1,
        shuffle_time_vector=[-10, 0],
        sequence_length=11,
        random_seed=random_seed + 300
    ),
    batch_size=20,
    drop_last=True
)

static_obj, static_results = model_training(
    static_tdl,
    prior,
    n_epochs,
    validation_dataloader=static_vdl,
    optimizer_params={'lr': 5e-5, 'weight_decay': 1e-7},
    gold_standard=gs,
    input_dropout_rate=0.5,
    hidden_dropout_rate=0.5,
    prediction_length=0,
    model_type="static_meta"
)

static_obj.save(static_outfile)

os.makedirs(static_result_dir, exist_ok=True)
static_results.write_result_files(
    static_result_dir
)

dyn_obj, _, post_results = pretrain_and_tune_dynamic_model(
    dynamic_pretrain,
    dyn_tdl,
    prior,
    n_epochs,
    pretraining_validation_dataloader=dynamic_preval,
    prediction_tuning_validation_dataloader=dyn_vdl,
    optimizer_params={'lr': 5e-5, 'weight_decay': 1e-7},
    gold_standard=gs,
    input_dropout_rate=0.5,
    hidden_dropout_rate=(0.0, 0.5),
    prediction_length=10,
    prediction_loss_offset=9,
    model_type="rnn"
)

dyn_obj.save(dynamic_outfile)

os.makedirs(dynamic_result_dir, exist_ok=True)
post_results.write_result_files(
    dynamic_result_dir
)
