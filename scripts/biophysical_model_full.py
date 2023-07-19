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
    TruncRobustScaler
)

from supirfactor_dynamical import model_training

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

ap.add_argument(
    "--dropout",
    dest="dropouts",
    help="Search Dropout Rates",
    action='store_const',
    const=True,
    default=False
)

args = ap.parse_args()

data_file = args.datafile
prior_file = args.priorfile
gs_file = args.gsfile

dynamic_outfile = args.outfile + "_{decay}_MODEL.h5"
dynamic_result_dir = args.outfile + "_{decay}"

n_epochs = args.epochs

if args.dropouts:
    hidden_dropout = (0.0, 0.5)
else:
    hidden_dropout = (0.0, 0.5)

validation_size = 0.25
random_seed = 1800

print(f"Loading and processing data from {data_file}")
adata = ad.read(data_file)

data = np.stack(
    (
        TruncRobustScaler(with_centering=False).fit_transform(
            adata.X
        ).A,
        TruncRobustScaler(with_centering=False).fit_transform(
            adata.layers['rapamycin_velocity']
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

dyn_tdl = DataLoader(
    TimeDataset(
        _train,
        time_vector[train_idx],
        -10,
        60,
        1,
        sequence_length=70,
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
        sequence_length=70,
        random_seed=random_seed + 300,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=20,
    drop_last=True
)

for d in ['NO_DECAY', 'CONSTANT_DECAY', 'DYNAMIC_DECAY']:

    bio_model, bio_result, _ = model_training(
        dyn_tdl,
        prior,
        n_epochs,
        None,
        validation_dataloader=dyn_vdl,
        optimizer_params={'lr': 5e-5, 'weight_decay': 1e-7},
        gold_standard=gs,
        hidden_dropout_rate=0.5,
        prediction_length=10,
        prediction_loss_offset=0,
        decay_model=False if d == "NO_DECAY" else None,
        time_dependent_decay=True if d != 'CONSTANT_DECAY' else False,
        model_type='biophysical'
    )

    bio_model.save(dynamic_outfile.format(decay=d))

    os.makedirs(dynamic_result_dir.format(decay=d), exist_ok=True)
    bio_result.write_result_files(
        dynamic_result_dir.format(decay=d)
    )
