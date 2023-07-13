import torch
import numpy as np
import tqdm

from .._utils import (
    _calculate_rss,
    _calculate_tss,
    _calculate_r2,
    _aggregate_r2
)


DEFAULT_OPTIMIZER_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 1e-10
}


class _TrainingMixin:

    training_loss = None
    validation_loss = None

    training_r2 = None
    validation_r2 = None

    output_t_plus_one = False
    n_additional_predictions = 0
    loss_offset = 0

    input_dropout_rate = 0.5
    hidden_dropout_rate = 0.0

    def train_model(
        self,
        training_dataloader,
        epochs,
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

        optimizer = self.process_optimizer(
            optimizer
        )

        # Create lists for loss per epoch if they do not exist
        # otherwise append to the existing list
        if self.training_loss is None:
            losses = []
        else:
            losses = self.training_loss

        if self.validation_loss is None and validation_dataloader is not None:
            validation_losses = []
        elif validation_dataloader is None:
            validation_losses = None
        else:
            validation_losses = self.validation_loss

        for _ in tqdm.trange(epochs):

            self.train()

            _batch_losses = []
            for train_x in training_dataloader:

                mse = loss_function(
                    self._slice_data_and_forward(train_x),
                    self.output_data(train_x)
                )

                mse.backward()
                optimizer.step()
                optimizer.zero_grad()

                _batch_losses.append(mse.item())

            losses.append(
                np.mean(_batch_losses)
            )

            # Get validation losses during training
            # if validation data was provided
            if validation_dataloader is not None:

                _validation_batch_losses = []

                with torch.no_grad():
                    for val_x in validation_dataloader:

                        _validation_batch_losses.append(
                            loss_function(
                                self._slice_data_and_forward(val_x),
                                self.output_data(val_x)
                            ).item()
                        )

                validation_losses.append(
                    np.mean(_validation_batch_losses)
                )

            # Shuffle stratified time data
            # is a noop unless the underlying DataSet is a TimeDataset
            _shuffle_time_data(training_dataloader)
            _shuffle_time_data(validation_dataloader)

        self.training_loss = losses
        self.validation_loss = validation_losses

        self.eval()
        self.r2(
            training_dataloader,
            validation_dataloader
        )

        return losses, validation_losses

    def set_dropouts(
        self,
        input_dropout_rate,
        hidden_dropout_rate
    ):

        self.input_dropout = torch.nn.Dropout(
            p=input_dropout_rate
        )

        self.hidden_dropout = torch.nn.Dropout(
            p=hidden_dropout_rate
        )

        self.input_dropout_rate = input_dropout_rate
        self.hidden_dropout_rate = hidden_dropout_rate

    def process_optimizer(
        self,
        optimizer
    ):

        # If it's None, create a default Adam optimizer
        if optimizer is None:
            return torch.optim.Adam(
                self.parameters(),
                **DEFAULT_OPTIMIZER_PARAMS
            )

        # If it's an existing optimizer, return it
        elif isinstance(optimizer, torch.optim.Optimizer):
            return optimizer

        # Otherwise assume it's a dict of Adam kwargs
        else:
            return torch.optim.Adam(
                self.parameters(),
                **optimizer
            )

    def set_time_parameters(
        self,
        output_t_plus_one=None,
        n_additional_predictions=None,
        loss_offset=None
    ):
        """
        Set parameters for dynamic prediction

        :param output_t_plus_one: Shift data so that models are using t for
            input and t+1 for output.
        :type output_t_plus_one: bool, optional
        :param n_additional_predictions: Number of time units to do forward
            prediction, defaults to None
        :type n_additional_predictions: int, optional
        :param loss_offset: How much of the input data to exclude from
            the loss function, increasing the importance of prediction,
            defaults to None
        :type loss_offset: int, optional
        """

        if n_additional_predictions is not None:
            self.n_additional_predictions = n_additional_predictions

        if loss_offset is not None:
            self.loss_offset = loss_offset

        if output_t_plus_one is not None:
            self.output_t_plus_one = output_t_plus_one

        if self.n_additional_predictions > 0:
            self.output_t_plus_one = True

        return self

    @torch.inference_mode()
    def r2(
        self,
        training_dataloader,
        validation_dataloader=None
    ):
        """
        Calculate unweighted-average R2 score and store in the model object

        :param training_dataloader: Training data in a torch DataLoader
        :type training_dataloader: torch.utils.data.DataLoader
        :param validation_dataloader: Validation data in a torch DataLoader,
            for calculating validation loss only. Defaults to None
        :type validation_dataloader: torch.utils.data.DataLoader, optional
        :returns: R2 for training and validation data (if provided)
        :rtype: float, float
        """

        self.eval()

        self.training_r2 = _aggregate_r2(
            self._calculate_r2_score(
                training_dataloader
            )
        )

        if validation_dataloader is not None:
            self.validation_r2 = _aggregate_r2(
                self._calculate_r2_score(
                    validation_dataloader
                )
            )

        return self.training_r2, self.validation_r2

    @torch.inference_mode()
    def _calculate_r2_score(
        self,
        dataloader
    ):

        if dataloader is None:
            return None

        _rss = 0
        _ss = 0

        with torch.no_grad():
            for data in dataloader:

                output_data = self.output_data(data)

                _rss += _calculate_rss(
                    output_data,
                    self._slice_data_and_forward(data),
                )

                _ss += _calculate_tss(
                    output_data
                )

        return _calculate_r2(_rss, _ss)

    def _slice_data_and_forward(self, train_x):

        forward = self(
            self.input_data(train_x),
            n_time_steps=self.n_additional_predictions
        )

        return self.output_data(
            forward,
            offset_only=True,
            keep_all_dims=True
        )

    def output_data(
        self,
        x,
        offset_only=False,
        truncate=True,
        no_loss_offset=False,
        keep_all_dims=False
    ):
        """
        Process data from DataLoader for output nodes in training.
        If output_t_plus_one is not None or zero, return the offset data in the
        sequence length axis.

        :param x: Data
        :type x: torch.Tensor
        :return: Output node values
        :rtype: torch.Tensor
        """

        # No need to do predictive offsets
        if not self.output_t_plus_one:
            return x

        # Don't shift for prediction if offset_only
        elif offset_only and self.loss_offset == 0:
            return x

        elif no_loss_offset and offset_only and not truncate:
            return x

        # Shift and truncate
        else:

            if not offset_only and not no_loss_offset:
                _, loss_offset = self._get_data_offsets(x)
                loss_offset += 1
            elif not offset_only and no_loss_offset:
                loss_offset = 1
            elif offset_only and no_loss_offset:
                loss_offset = 0
            else:
                _, loss_offset = self._get_data_offsets(x, check=False)

            if truncate:
                end_offset = self.n_additional_predictions + 2
                return x[:, loss_offset:end_offset, :]

            else:
                return x[:, loss_offset:, :]

    def input_data(self, x):
        """
        Process data from DataLoader for input nodes in training.
        If output_t_plus_one is set, return the first data point in the
        sequence length axis.

        :param x: Data
        :type x: torch.Tensor
        :return: Input node values
        :rtype: torch.Tensor
        """
        if self.output_t_plus_one:
            return x[:, [0], ...]
        else:
            return x

    def _get_data_offsets(self, x, check=True):

        if not self.output_t_plus_one:
            return None, None

        if self.output_t_plus_one and x.ndim < 3:
            raise ValueError(
                "3D data (N, L, H) must be provided when "
                "predicting time-dependent data"
            )

        L = x.shape[1]

        input_offset = L - self.n_additional_predictions - 1
        output_offset = 1 + self.loss_offset

        if check and ((input_offset < 1) or (output_offset >= L)):
            raise ValueError(
                f"Cannot train on {L} sequence length with "
                f"{self.n_additional_predictions} additional predictions and "
                f"{self.loss_offset} values excluded from loss"
            )

        return input_offset, self.loss_offset


def _shuffle_time_data(dl):
    try:
        dl.dataset.shuffle()
    except AttributeError:
        pass