import argparse
import os
import sys
import torch

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    SupirFactorBiophysical
)

from supirfactor_dynamical._utils._adata_load import (
    load_data_files_jtb_2023 as load_data,
)

from supirfactor_dynamical import (
    model_training
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
TRAINED_DECAY = os.path.join(
    "/mnt/ceph/users/cjackson/supirfactor_trs_decay",
    "RAPA_DECAY_MODEL.h5"
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
    default=500
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
    "--constrain_decay",
    dest="constrain_decay",
    help="Constrain Decay with Real Data",
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
tuned_delay = 100 if args.tuned_decay else n_epochs

validation_size = 0.25
random_seed = 1800

torch.manual_seed(random_seed)

data, time_lookup, prior, gs, count_scaling, velo_scaling = load_data(
    data_file,
    prior_file,
    gs_file,
    counts=True,
    velocity=True,
    decay_velocity=True
)

time_vector = time_lookup['rapa'][0]

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
        sequence_length=20,
        random_seed=random_seed + 200,
        shuffle_time_vector=[-10, 0]
    ),
    batch_size=25,
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
    batch_size=25,
    drop_last=True
)


dyn_obj = SupirFactorBiophysical(
    prior,
    decay_model=TRAINED_DECAY if args.pretrained_decay else None,
    input_dropout_rate=0.5,
    hidden_dropout_rate=0.5,
    decay_epoch_delay=tuned_delay if args.pretrained_decay else 0,
    output_activation='softplus',
    separately_optimize_decay_model=args.constrain_decay
)

dyn_obj.set_time_parameters(
    n_additional_predictions=10
)

bio_model, bio_result, _erv = model_training(
    dyn_tdl,
    prior,
    n_epochs,
    validation_dataloader=dyn_vdl,
    optimizer_params={'lr': 5e-5, 'weight_decay': 1e-7},
    gold_standard=gs,
    model_type=dyn_obj,
    return_erv=True,
    count_scaling=count_scaling.scale_,
    velocity_scaling=velo_scaling.scale_
)

bio_model.save(dynamic_outfile)

os.makedirs(dynamic_result_dir, exist_ok=True)

bio_result.write_result_files(
    dynamic_result_dir
)
