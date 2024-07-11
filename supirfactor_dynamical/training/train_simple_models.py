import torch
import numpy as np
import tqdm

from supirfactor_dynamical._utils import (
    to,
    _nobs
)
from supirfactor_dynamical.datasets import stack_dataloaders


def train_simple_model(
    model,
    training_dataloader,
    epochs,
    device='cpu',
    validation_dataloader=None,
    loss_function=torch.nn.MSELoss(),
    optimizer=None,
    post_epoch_hook=None,
    loss_index=None
):
    """
    Train this model

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

    to(model_ref, device)

    optimizer = model_ref.process_optimizer(
        optimizer
    )

    # Set training time
    model_ref.set_training_time()

    for epoch_num in (
        pbar := tqdm.trange(model_ref.current_epoch + 1, epochs)
    ):

        model_ref.train()

        _batch_losses = []
        _batch_n = []
        for train_x in stack_dataloaders(training_dataloader):

            train_x = to(train_x, device)

            predict_x = model_ref(
                model_ref.input_data(train_x)
            )

            loss = loss_function(
                predict_x,
                model_ref.output_data(train_x)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _batch_losses.append(loss.item())
            _batch_n.append(_nobs(train_x))

        if validation_dataloader is not None:

            with torch.no_grad():
                _validation_loss = []
                _validation_n = []
                for val_x in stack_dataloaders(validation_dataloader):

                    val_x = to(val_x, device)

                    predict_x = model_ref(
                        model_ref.input_data(val_x)
                    )

                    loss = loss_function(
                        predict_x,
                        model_ref.output_data(val_x)
                    )

                    _validation_loss.append(loss.item())
                    _validation_n.append(_nobs(val_x))

                _validation_loss = np.average(
                    np.array(_validation_loss),
                    axis=0,
                    weights=np.array(_validation_n)
                )
                _validation_n = np.sum(_validation_n)

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
            validation_loss=_validation_loss,
            validation_n=_validation_n,
            training_loss_idx=loss_index,
            validation_loss_idx=loss_index
        )

        model_ref.current_epoch = epoch_num
        pbar.set_description(f"[{epoch_num} n={np.sum(_batch_n)}]")

        if post_epoch_hook is not None:

            # If the hook returns True
            # it's early stopping time
            if post_epoch_hook(model_ref) is True:
                break

    return model
