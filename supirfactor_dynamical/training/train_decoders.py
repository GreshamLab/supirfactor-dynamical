import torch
import tqdm
import numpy as np

from supirfactor_dynamical.models._model_mixins.training_mixin import (
    _shuffle_time_data
)
from supirfactor_dynamical._utils import to


def train_decoder_submodels(
    model,
    training_dataloader,
    epochs,
    encoder_data_index=0,
    decoder_models=('default_decoder', ),
    freeze_embeddings=False,
    validation_dataloader=None,
    loss_function=torch.nn.MSELoss(),
    optimizer=None
):
    """
    Train this model

    :param training_dataloader: Training data in a torch DataLoader
    :type training_dataloader: torch.utils.data.DataLoader
    :param epochs: Number of training epochs
    :type epochs: int
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
    :return: Training losses and validation losses
        (if validation_dataloader is provided)
    :rtype: np.ndarray, np.ndarray
    """

    if (
        not hasattr(model, "_multisubmodel_model") or
        not model._multisubmodel_model
    ):
        raise RuntimeError(
            "This training loop requires a model with multiple subunits"
        )

    to(model, device=model.device)

    # Set training time and create loss lists
    model.set_training_time()
    model.training_loss
    model.training_n

    [model._check_label(x) for x in decoder_models]

    if freeze_embeddings:
        model.freeze_submodel('encoder')
        model.freeze_submodel('intermediate')
        optimizers = {
            x: model.process_optimizer(
                optimizer,
                params=model.module_bag[x].parameters()
            )
            for x in decoder_models
        }
    else:
        optimizers = {}
        for x in decoder_models:
            model.select_submodel(x, 'decoder')
            optimizers[x] = model.process_optimizer(
                optimizer
            )

    if not isinstance(loss_function, (tuple, list)):
        loss_function = [loss_function] * len(optimizers)

    for epoch_num in tqdm.trange(epochs):

        model.train()

        _batch_losses = []
        _batch_n = 0
        for train_x in training_dataloader:

            to(train_x, model.device)
            if not isinstance(train_x, (tuple, list)):
                train_x = [train_x] * len(decoder_models)

            _decoder_losses = []
            _batch_n = _batch_n + train_x[0].shape[0]
            for x, lf, _target_x in zip(
                decoder_models,
                loss_function,
                train_x
            ):
                model.select_submodel(x, 'decoder')

                _decoder_losses.append(
                    model._training_step(
                        epoch_num,
                        train_x[encoder_data_index],
                        optimizers[x],
                        lf,
                        target_x=_target_x
                    )
                )

            _batch_losses.append(_decoder_losses)

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

                    if not isinstance(val_x, (tuple, list)):
                        val_x = [val_x] * len(decoder_models)

                    _batch_loss = []
                    _val_n = _val_n + train_x[0].shape[0]
                    for x, lf, _target_x in zip(
                        decoder_models,
                        loss_function,
                        val_x
                    ):
                        model.select_submodel(x, 'decoder')

                        _batch_loss.append(
                            model._calculate_all_losses(
                                val_x[encoder_data_index],
                                lf,
                                target_data=_target_x
                            )
                        )

                    _val_loss.append(tuple(_batch_loss))

            model._validation_n.append(_val_n)
            model._validation_loss.append(
                np.mean(np.array(_val_loss), axis=0)
            )

        # Shuffle stratified time data
        # is a noop unless the underlying DataSet is a TimeDataset
        _shuffle_time_data(training_dataloader)
        _shuffle_time_data(validation_dataloader)

    return model
