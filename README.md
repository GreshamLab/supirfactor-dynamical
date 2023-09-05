# supirfactor-dynamical

[![PyPI version](https://badge.fury.io/py/supirfactor-dynamical.svg)](https://badge.fury.io/py/supirfactor-dynamical)
[![CI](https://github.com/GreshamLab/supirfactor-dynamical/actions/workflows/python-package.yml/badge.svg)](https://github.com/GreshamLab/supirfactor-dynamical/actions/workflows/python-package.yml/)
[![codecov](https://codecov.io/gh/GreshamLab/supirfactor-dynamical/branch/main/graph/badge.svg)](https://codecov.io/gh/GreshamLab/supirfactor-dynamical)

This is a PyTorch model package for creating dynamical, biophysical models of
transcriptional output and regulation.

### Installation

Install this package using the standard python package manager `python -m pip install supirfactor_dynamical`.
It depends on [PyTorch](https://pytorch.org/get-started/locally/) and the standard python scientific computing
packages (e.g. scipy, numpy, pandas).

### Usage

```
from supirfactor_dynamical import (
    SupirFactorBiophysical
)

# Construct model object
model = SupirFactorBiophysical(
    prior_network,                  # Prior knowledge connectivity network [Genes x TFs]
    output_activation='softplus'    # Use softplus activation for transcriptional model output
)

# Set prediction parameter
model.set_time_parameters(
    n_additional_predictions=10     # Make forward predictions in time during training
)

# Train model
model.train_model(
    training_dataloader,            # Training data in a torch DataLoader
    500                             # Epochs
)

# Save model
model.save("supirfactor_dynamical.h5")
```

Examples containing data loading, hyperparameter searching, and result testing are located in `./scripts/`
