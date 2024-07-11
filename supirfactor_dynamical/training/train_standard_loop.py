import torch
import numpy as np
import tqdm

from supirfactor_dynamical._utils import (
    to,
    _nobs
)
from supirfactor_dynamical.datasets import (
    stack_dataloaders,
    _shuffle_time_data
)


def train_model(
    model,
    training_dataloader,
    epochs,
    validation_dataloader=None,
    loss_function=torch.nn.MSELoss(),
    optimizer=None,
    post_epoch_hook=None,
    input_data_index=None,
    output_data_index=None,
    final_device=None,
    loss_index=None
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

    if hasattr(model, 'module'):
        model_ref = model.module
    else:
        model_ref = model

    to(model_ref, model_ref.device)

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

            if output_data_index is not None:
                target_x = to(
                    train_x[output_data_index],
                    model_ref.device
                )
            else:
                target_x = None

            if input_data_index is not None:
                train_x = train_x[input_data_index]

            train_x = to(train_x, model_ref.device)

            mse = model_ref._training_step(
                epoch_num,
                train_x,
                optimizer,
                loss_function,
                target_x=target_x
            )

            _batch_losses.append(mse)
            _batch_n.append(_nobs(train_x))

        model_ref.append_loss(
            training_loss=np.average(
                np.array(_batch_losses),
                axis=0,
                weights=np.array(_batch_n)
            ),
            training_n=np.sum(_batch_n),
            training_loss_idx=loss_index
        )

        # Get validation losses during training
        # if validation data was provided
        if validation_dataloader is not None:

            _vloss, _vn = model_ref._calculate_validation_loss(
                stack_dataloaders(validation_dataloader),
                loss_function,
                input_data_index=input_data_index,
                output_data_index=output_data_index
            )

            model_ref.append_loss(
                validation_loss=_vloss,
                validation_n=_vn,
                validation_loss_idx=loss_index
            )

        # Shuffle stratified time data
        # is a noop unless the underlying DataSet is a TimeDataset
        _shuffle_time_data(training_dataloader)
        _shuffle_time_data(validation_dataloader)

        model_ref.current_epoch = epoch_num
        pbar.set_description(f"[{epoch_num} n={np.sum(_batch_n)}]")

        if post_epoch_hook is not None:

            # If the hook returns True
            # it's early stopping time
            if post_epoch_hook(model_ref) is True:
                break

    model_ref.eval()

    model_ref.r2(
        training_dataloader,
        validation_dataloader,
        input_data_index=input_data_index,
        target_data_index=output_data_index
    )

    if final_device is not None:
        to(model_ref, final_device)

    return model
