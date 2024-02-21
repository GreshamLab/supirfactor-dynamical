import torch
import tqdm
import numpy as np

from supirfactor_dynamical.models._model_mixins.training_mixin import (
    _shuffle_time_data
)
from supirfactor_dynamical.training._utils import _set_submodels
from supirfactor_dynamical._utils import to


def train_embedding_submodels(
    model,
    training_dataloader,
    epochs,
    training_encoder=None,
    training_intermediate=None,
    training_data_map=None,
    frozen_encoder='default_encoder',
    frozen_intermediate='default_intermediate',
    validation_dataloader=None,
    loss_function=torch.nn.MSELoss(),
    optimizer=None
):
    """
    Train model encoder/intermediate modules to produce a specific
    latent embedding, produced by the frozen_encoder and
    frozen_intermediate modubles

    :param model: Torch model with the multisubmodel mixin property
    :type model: torch.nn.Module
    :param training_dataloader: Training data in a torch DataLoader
    :type training_dataloader: torch.utils.data.DataLoader
    :param epochs: Number of training epochs
    :type epochs: int
    :param training_encoder: _description_, defaults to None
    :type training_encoder: _type_, optional
    :param training_intermediate: _description_, defaults to None
    :type training_intermediate: _type_, optional
    :param frozen_encoder: Encoder submodel module name to use to build the
        "correct" embeddings, defaults to 'default_encoder'
    :type frozen_encoder: str, optional
    :param frozen_intermediate: Intermediate submodel module name to use to
        build the "correct" embeddings, defaults to 'default_intermediate'
    :type frozen_intermediate: str, optional
    :param validation_dataloader: Validation data in a torch DataLoader,
        for calculating validation loss only. Defaults to None
    :type validation_dataloader: torch.utils.data.DataLoader, optional
    :param loss_function: Torch loss function object for training,
        defaults to torch.nn.MSELoss()
    :type loss_function: torch.nn.Loss(), optional
    :param optimizer: Torch optimizer for training. Can be an
        Optimizer instance, a dict of kwargs for torch.optim.Adam, or
        if None, defaults to Adam(lr=1e-3, weight_decay=1e-10)
    :type optimizer: torch.optim.object(), dict(), optional
    """

    if (
        not hasattr(model, "_multisubmodel_model") or
        not model._multisubmodel_model
    ):
        raise RuntimeError(
            "This training loop requires a model with multiple subunits"
        )

    to(model, device=model.device)
    optimizer = model.process_optimizer(optimizer)

    # Set training time and create loss lists
    model.set_training_time()
    model.training_loss
    model.training_n

    try:
        model.add_submodel(
            'NULL',
            torch.nn.Sequential()
        )
    except ValueError:
        pass

    model.select_submodel(
        'NULL',
        model_type='decoder'
    )

    for epoch_num in tqdm.trange(epochs):

        model.train()

        _batch_losses = []
        _batch_n = 0
        for train_x in training_dataloader:

            to(train_x, model.device)
            _embed_x, _train_x = _get_data(train_x)
            _batch_n = _batch_n + train_x.shape[0]

            frozen_embedding = _get_embedding(
                model,
                _embed_x,
                frozen_encoder,
                frozen_intermediate
            )

            _set_submodels(
                model,
                encoder=training_encoder,
                intermediate=training_intermediate
            )

            mse = model._training_step(
                epoch_num,
                _train_x,
                optimizer,
                loss_function,
                target_x=frozen_embedding
            )
            del _train_x

            _batch_losses.append(mse)

        model._training_loss.append(
            np.mean(np.array(_batch_losses), axis=0)
        )
        model._training_n.append(_batch_n)

        # Get validation losses during training
        # if validation data was provided
        if validation_dataloader is not None:
            model.validation_loss
            model.validation_n

            with torch.no_grad():

                _val_loss = []
                _val_n = 0
                for val_x in validation_dataloader:

                    to(val_x, model.device)

                    _embed_val_x, _val_x = _get_data(val_x)
                    _embed = _get_embedding(model, _embed_val_x)
                    _val_n = _val_n + _val_x.shape[0]

                    _set_submodels(
                        model,
                        encoder=training_encoder,
                        intermediate=training_intermediate
                    )

                    _val_loss.append(
                        model._calculate_all_losses(
                            _val_x,
                            loss_function,
                            target_data=_embed
                        )[0]
                    )

            model._validation_n.append(_val_n)
            model._validation_loss.append(
                np.mean(np.array(_val_loss), axis=0)
            )

        # Shuffle stratified time data
        # is a noop unless the underlying DataSet is a TimeDataset
        _shuffle_time_data(training_dataloader)
        _shuffle_time_data(validation_dataloader)

    return model


def _get_data(
    x
):
    if isinstance(x, (list, tuple)):
        _embed_x, _train_x = x
    else:
        _embed_x, _train_x = x, x

    return _embed_x, _train_x


def _get_embedding(
    model,
    x,
    encoder='default_encoder',
    intermediate='default_intermediate'
):

    _set_submodels(
        model,
        encoder=encoder,
        intermediate=intermediate
    )

    with torch.no_grad():
        return model.latent_embedding(x)
