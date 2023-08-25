import argparse
import os
import sys

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    model_training,
    get_model
)

from supirfactor_dynamical.train import pretrain_and_tune_dynamic_model

from supirfactor_dynamical._utils._adata_load import (
    load_data_files_jtb_2023 as load_data
)


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

hidden_dropout = (0.0, 0.5)

validation_size = 0.25
random_seed = 1800

data, time_lookup, prior, gs, count_scaling, velo_scaling = load_data(
    data_file,
    prior_file,
    gs_file,
    counts=True,
    velocity=True
)

train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=random_seed
)

time_vector = time_lookup['rapa'][0]

_train = data[train_idx, ...]
_test = data[test_idx, ...]

print(f"Training on {_train.shape} and validating on {_test.shape}")

static_tdl = DataLoader(
    TimeDataset(
        _train,
        time_vector[train_idx],
        -10,
        60,
        random_seed=random_seed
    ),
    batch_size=200,
    drop_last=True
)

static_vdl = DataLoader(
    TimeDataset(
        _test,
        time_vector[test_idx],
        -10,
        60,
        random_seed=random_seed + 100
    ),
    batch_size=200,
    drop_last=True
)

pre_tdl = DataLoader(
    TimeDataset(
        _train,
        time_vector[train_idx],
        -10,
        60,
        1,
        sequence_length=11,
        random_seed=random_seed + 200,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=20,
    drop_last=True
)

pre_vdl = DataLoader(
    TimeDataset(
        _test,
        time_vector[test_idx],
        -10,
        60,
        1,
        sequence_length=11,
        random_seed=random_seed + 300,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=20,
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

dyn_obj, pre_res, post_results = pretrain_and_tune_dynamic_model(
    pre_tdl,
    dyn_tdl,
    prior,
    n_epochs,
    pretraining_validation_dataloader=pre_vdl,
    prediction_tuning_validation_dataloader=dyn_vdl,
    optimizer_params={'lr': 5e-5, 'weight_decay': 1e-7},
    gold_standard=gs,
    input_dropout_rate=0.5,
    hidden_dropout_rate=(0.0, 0.5),
    prediction_length=10,
    prediction_loss_offset=9,
    count_scaling=count_scaling.scale_,
    velocity_scaling=velo_scaling.scale_,
    model_type='biophysical',
    decay_model=False
)

dyn_obj.save(dynamic_outfile)

os.makedirs(dynamic_result_dir, exist_ok=True)

post_results.write_result_files(
    dynamic_result_dir
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
    model_type=get_model(
        'static_meta',
        velocity=True
    )
)

static_obj.save(static_outfile)

os.makedirs(static_result_dir, exist_ok=True)
static_results.write_result_files(
    static_result_dir
)
