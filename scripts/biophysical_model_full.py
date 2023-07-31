import argparse
import gc
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    TruncRobustScaler
)

from supirfactor_dynamical import pretrain_and_tune_dynamic_model

DEFAULT_PATH = "/mnt/ceph/users/cjackson/inferelator/data/RAPA/"
DEFAULT_PRIOR = os.path.join(
    DEFAULT_PATH,
    "JOINT_PRIOR_20230701.tsv.gz"
)
DEFAULT_DATA = os.path.join(
    DEFAULT_PATH,
    "2021_INFERELATOR_DATA.h5ad"
)
TRAINED_DECAY = os.path.join(
    "/mnt/ceph/users/cjackson/supirfactor_trs_decay",
    "RAPA_FULL_DECAY_MODEL.h5"
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

ap.add_argument(
    "--freeze",
    dest="freeze",
    help="Freeze Decay",
    action='store_const',
    const=True,
    default=False
)

args = ap.parse_args()

data_file = args.datafile
prior_file = args.priorfile
gs_file = args.gsfile

dynamic_outfile = args.outfile + "_BIOPHYSICAL_MODEL.h5"
dynamic_result_dir = args.outfile

n_epochs = args.epochs

hidden_dropout = (0.0, 0.5)

validation_size = 0.25
random_seed = 1800

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
        )
    ),
    axis=-1
)

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

_train = data[train_idx, ...]
_test = data[test_idx, ...]

print(f"Training on {_train.shape} and validating on {_test.shape}")

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

bio_model, bio_result, _ = pretrain_and_tune_dynamic_model(
    pre_tdl,
    dyn_tdl,
    prior,
    n_epochs,
    pretraining_validation_dataloader=pre_vdl,
    prediction_tuning_validation_dataloader=dyn_vdl,
    optimizer_params={'lr': 5e-5, 'weight_decay': 1e-7},
    gold_standard=gs,
    input_dropout_rate=0.5,
    hidden_dropout_rate=hidden_dropout,
    prediction_length=10,
    prediction_loss_offset=10,
    count_scaling=count_scaling.scale_,
    velocity_scaling=velo_scaling.scale_,
    model_type='biophysical',
    decay_model=TRAINED_DECAY,
    freeze_decay_model=args.freeze
)

bio_model.save(dynamic_outfile)

os.makedirs(dynamic_result_dir, exist_ok=True)

bio_result.write_result_files(
    dynamic_result_dir
)
