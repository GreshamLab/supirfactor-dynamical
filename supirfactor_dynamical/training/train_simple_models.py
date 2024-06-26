import torch
import numpy as np
import tqdm

from supirfactor_dynamical._utils import (
    to,
    _nobs
)


def train_simple_model(
    model,
    training_dataloader,
    epochs,
    device='cpu',
    validation_dataloader=None,
    loss_function=torch.nn.MSELoss(),
    optimizer=None,
    post_epoch_hook=None,
    separate_output_data=False,
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

    def _input_data(x):
        if separate_output_data:
            return x[0]
        else:
            return x

    def _output_data(x):
        if separate_output_data:
            return x[1]
        else:
            return x

    for epoch_num in (
        pbar := tqdm.trange(model_ref.current_epoch + 1, epochs)
    ):

        model_ref.train()

        _batch_losses = []
        _batch_n = []
        for train_x in training_dataloader:

            to(train_x, device)

            predict_x = model_ref(
                _input_data(train_x)
            )

            loss = loss_function(
                predict_x,
                _output_data(train_x)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _batch_losses.append(loss.item())
            _batch_n.append(_nobs(train_x))

        if validation_dataloader is not None:

            _validation_loss = []
            _validation_n = []
            for val_x in training_dataloader:

                to(val_x, device)

                predict_x = model_ref(
                    _input_data(val_x)
                )

                loss = loss_function(
                    predict_x,
                    _output_data(val_x)
                )

                _validation_loss.append(loss.item())
                _validation_n.append(_nobs(train_x))

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

        )

        model_ref.current_epoch = epoch_num
        pbar.set_description(f"[{epoch_num} n={np.sum(_batch_n)}]")

        if post_epoch_hook is not None:
            post_epoch_hook(model_ref)

    return model
