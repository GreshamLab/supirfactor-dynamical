import torch
import numpy as np
import tqdm
import warnings

from supirfactor_dynamical._utils import (
    to,
    _nobs
)
from supirfactor_dynamical.datasets import stack_dataloaders


def train_simple_multidecoder(
    model,
    training_dataloader,
    epochs,
    device='cpu',
    validation_dataloader=None,
    decoder_models=('default_decoder', ),
    skip_decoder_training_models=tuple(),
    decoder_models_validation=None,
    freeze_embeddings=tuple(),
    loss_function=torch.nn.MSELoss(),
    optimizer=None,
    post_epoch_hook=None,
    training_loss_weights=None,
    loss_index=None
):
    """
    Train this model with multiple decoders

    :param model: Torch model to be trained.
    :type model: torch.nn.Module
    :param training_dataloader: Training data in a torch DataLoader
    :type training_dataloader: torch.utils.data.DataLoader
    :param epochs: Number of training epochs
    :type epochs: int
    :param device: Move and train on this device. Defaults to 'cpu'.
    :type device: str, optional
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
    :param post_epoch_hook: Callable function that will be executed as
        post_epoch_hook(model) after each epoch
    :type post_epoch_hook: Callable, optional
    :return: Model reference
    :rtype: torch.nn.Module
    """

    if hasattr(model, 'module'):
        model_ref = model.module
    else:
        model_ref = model

    if (
        not hasattr(model_ref, "_multisubmodel_model") or
        not model_ref._multisubmodel_model
    ):
        warnings.warn(
            "This training loop requires a model with multiple subunits"
        )

    to(model, device=model.device)

    # Set training time
    model_ref.set_training_time()

    [model_ref._check_label(x) for x in decoder_models]

    if freeze_embeddings is True:
        freeze_embeddings = decoder_models

    if decoder_models_validation is None:
        decoder_models_validation = decoder_models

    optimizer = model_ref.process_optimizer(
        optimizer
    )

    if not isinstance(loss_function, (tuple, list)):
        loss_function = [loss_function] * len(decoder_models)

    if not isinstance(training_loss_weights, (tuple, list)):
        training_loss_weights = [training_loss_weights] * len(decoder_models)

    for epoch_num in (
        pbar := tqdm.trange(model_ref.current_epoch + 1, epochs)
    ):

        model_ref.train()

        _batch_losses = []
        _batch_n = []
        for train_x in stack_dataloaders(training_dataloader):

            train_x = to(train_x, device)

            # Run through each decoder in order
            _decoder_losses = []
            for x, lf, _target_x, loss_weight in zip(
                decoder_models,
                loss_function,
                train_x,
                training_loss_weights
            ):

                if x in skip_decoder_training_models:
                    _decoder_losses.append(0.)
                    continue

                model_ref.select_submodel(x, 'decoder')

                if x in freeze_embeddings:
                    model_ref.freeze_submodel('encoder')
                    model_ref.freeze_submodel('intermediate')

                predict_x = model_ref(
                    model_ref.input_data(train_x[0])
                )

                loss = lf(
                    predict_x,
                    model_ref.output_data(_target_x)
                )

                if loss_weight is not None:
                    loss *= loss_weight

                loss.backward()
                _decoder_losses.append(loss.item())

                if x in freeze_embeddings:
                    model_ref.freeze_submodel('encoder', unfreeze=True)
                    model_ref.freeze_submodel('intermediate', unfreeze=True)

            optimizer.step()
            optimizer.zero_grad()

            _batch_losses.append(_decoder_losses)
            _batch_n.append(_nobs(train_x))

        if validation_dataloader is not None:

            _validation_loss = []
            _validation_n = []

            with torch.no_grad():

                for val_x in stack_dataloaders(training_dataloader):

                    val_x = to(val_x, device)

                    _decoder_losses = []
                    for x, lf, _target_x in zip(
                        decoder_models,
                        loss_function,
                        val_x
                    ):

                        model_ref.select_submodel(x, 'decoder')

                        predict_x = model_ref(
                            model_ref.input_data(val_x[0])
                        )

                        loss = lf(
                            predict_x,
                            model_ref.output_data(_target_x)
                        )

                        _decoder_losses.append(loss.item())

                    _validation_loss.append(_decoder_losses)
                    _validation_n.append(_nobs(val_x))

        else:
            _validation_loss = None
            _validation_n = None

        model_ref.append_loss(
            training_loss=np.average(
                np.array(_batch_losses),
                axis=0,
                weights=np.array(_batch_n)
            ),
            training_n=np.sum(_batch_n),
            validation_loss=np.average(
                np.array(_validation_loss),
                axis=0,
                weights=np.array(_validation_n)
            ),
            validation_n=np.sum(_validation_n),
            training_loss_idx=loss_index,
            validation_loss_idx=loss_index
        )

        model_ref.current_epoch = epoch_num
        pbar.set_description(f"[{epoch_num} n={np.sum(_batch_n)}]")

        if post_epoch_hook is not None:
            post_epoch_hook(model_ref)

    return model
