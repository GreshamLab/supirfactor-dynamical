import argparse
import sys
import os

import anndata as ad
import numpy as np
import scanpy as sc

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

print("Supirfactor-rnn Decay Model Build")
print(" ".join(sys.argv))

ap = argparse.ArgumentParser(description="SUPIRFACTOR-RNN Decay Model")

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
    "--epochs",
    dest="epochs",
    help="NUM Epochs",
    metavar="NUM",
    type=int,
    default=2000
)

args = ap.parse_args()

data_file = args.datafile
model_outfile = args.outfile + "_DECAY_MODEL.h5"

n_epochs = args.epochs

random_seed = 1800
validation_size = 0.25

print(f"Loading and processing data from {data_file}")
adata = ad.read(data_file)

adata.X = adata.X.astype(np.float32)
sc.pp.normalize_per_cell(adata, min_counts=0)
data = np.stack(
    (
        TruncRobustScaler(with_centering=False).fit_transform(
            adata.X
        ).A,
        adata.layers['decay_constants']
    ),
    axis=-1
)

time_vector = adata.obs['program_rapa_time'].values

train_idx, test_idx = train_test_split(
    np.arange(data.shape[0]),
    test_size=validation_size,
    random_state=random_seed
)

_train = data[train_idx, ...]
_test = data[test_idx, ...]

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

torch.manual_seed(random_seed)

model_obj = DecayModule(
    data.shape[1]
)

model_obj.train_model(
    dyn_tdl,
    n_epochs,
    validation_dataloader=dyn_vdl,
    optimizer={'lr': 5e-5, 'weight_decay': 1e-7}
)

model_obj.save(model_outfile)
