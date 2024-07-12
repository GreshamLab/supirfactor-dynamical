import torch
import numpy as np
import pandas as pd
import time

from supirfactor_dynamical._utils import (
    to,
    _cat,
    _nobs,
    _to_tensor
)

from supirfactor_dynamical._io._writer import write
from supirfactor_dynamical._utils import _check_data_offsets
from supirfactor_dynamical.postprocessing.eval import r2_score
from supirfactor_dynamical.datasets import stack_dataloaders

from torch.utils.data import DataLoader


DEFAULT_OPTIMIZER_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 1e-10
}


class _TrainingMixin:

    device = 'cpu'

    training_time = None
    current_epoch = -1

    _training_loss = None
    _validation_loss = None
    _training_n = None
    _validation_n = None
    _loss_type_names = None

    training_r2 = None
    validation_r2 = None

    output_t_plus_one = False
    n_additional_predictions = 0
    loss_offset = 0

    input_dropout_rate = 0.5
    hidden_dropout_rate = 0.0

    @property
    def _model_device(self):
        device = next(self.parameters()).device
        if device == -1:
            device = 'cpu'
        return device

    @property
    def _offset_data(self):
        return (
            self.output_t_plus_one or
            (self.n_additional_predictions != 0) or
            (self.loss_offset != 0)
        )

    @property
    def training_loss(self):
        if self._training_loss is None:
            return np.array([])

        return self._training_loss

    @property
    def validation_loss(self):
        if self._validation_loss is None:
            return np.array([])

        return self._validation_loss

    @property
    def training_loss_df(self):
        return self._loss_df(self.training_loss)

    @property
    def validation_loss_df(self):
        return self._loss_df(self.validation_loss)

    @property
    def training_n(self):
        # Check dims for backward compatibility
        if self._training_n is None:
            pass
        elif self._training_n.ndim < 2:
            self._training_n = self._training_n.reshape(-1, 1)

        return self._training_n

    @property
    def validation_n(self):
        # Check dims for backward compatibility
        if self._validation_n is None:
            pass
        elif self._validation_n.ndim < 2:
            self._validation_n = self._validation_n.reshape(-1, 1)

        return self._validation_n

    def train_model(
        self,
        training_dataloader,
        epochs,
        validation_dataloader=None,
        loss_function=torch.nn.MSELoss(),
        optimizer=None,
        post_epoch_hook=None,
        **kwargs
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

        from supirfactor_dynamical.training.train_standard_loop import (
            train_model
        )

        return train_model(
            self,
            training_dataloader,
            epochs,
            validation_dataloader=validation_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            post_epoch_hook=post_epoch_hook,
            **kwargs
        )

    def _training_step(
        self,
        epoch_num,
        train_x,
        optimizer,
        loss_function,
        input_x=None,
        target_x=None,
        loss_weight=None,
        optimizer_step=True,
        **kwargs
    ):

        if input_x is None:
            input_x = self._slice_data_and_forward(train_x, **kwargs)

        if target_x is None:
            target_x = self.output_data(train_x)

        mse = loss_function(
            input_x,
            target_x
        )

        if loss_weight is not None:
            mse = mse * loss_weight

        mse.backward()

        if optimizer_step:
            optimizer.step()
            optimizer.zero_grad()

        return mse.item()

    def _calculate_all_losses(
        self,
        x,
        loss_function,
        target_data=None,
        **kwargs
    ):

        if target_data is None:
            target_data = self.output_data(x)

        loss = loss_function(
            self._slice_data_and_forward(x, **kwargs),
            target_data
        ).item()

        return (loss, )

    def _calculate_validation_loss(
        self,
        validation_dataloader,
        loss_function,
        input_data_index=None,
        output_data_index=None
    ):
        # Get validation losses during training
        # if validation data was provided
        if validation_dataloader is not None:

            _validation_batch_losses = []
            _validation_n = []

            with torch.no_grad():
                device = self._model_device

                for val_x in validation_dataloader:

                    if output_data_index is not None:
                        val_target_x = val_x[output_data_index]
                        val_target_x = to(val_target_x, device)
                    else:
                        val_target_x = None

                    if input_data_index is not None:
                        val_x = val_x[input_data_index]

                    val_x = to(val_x, device)

                    _validation_batch_losses.append(
                        self._calculate_all_losses(
                            val_x,
                            loss_function,
                            target_data=val_target_x
                        )
                    )

                    _validation_n.append(_nobs(val_x))

            return (
                np.average(
                    np.array(_validation_batch_losses),
                    axis=0,
                    weights=np.array(_validation_n)
                ),
                np.sum(_validation_n)
            )

    def set_dropouts(
        self,
        input_dropout_rate,
        hidden_dropout_rate,
        intermediate_dropout_rate=None
    ):

        self.input_dropout = torch.nn.Dropout(
            p=input_dropout_rate
        )

        self.hidden_dropout = torch.nn.Dropout(
            p=hidden_dropout_rate
        )

        self.input_dropout_rate = input_dropout_rate
        self.hidden_dropout_rate = hidden_dropout_rate

        if intermediate_dropout_rate is not None:
            self.intermediate_dropout = torch.nn.Dropout(
                p=intermediate_dropout_rate
            )
            self.intermediate_dropout_rate = intermediate_dropout_rate

        return self

    def process_optimizer(
        self,
        optimizer,
        params=None
    ):

        if params is None:
            params = self.parameters()

        # If it's None, create a default Adam optimizer
        if optimizer is None:
            return torch.optim.Adam(
                params,
                **DEFAULT_OPTIMIZER_PARAMS
            )

        # If it's False, pass it back through
        elif optimizer is False:
            return False

        # If it's a tuple, process the individual tuple elements
        # separately for optimizer
        elif isinstance(optimizer, (tuple, list)):
            return tuple(
                self.process_optimizer(opt)
                for opt in optimizer
            )

        # If it's an existing optimizer, return it
        elif isinstance(optimizer, torch.optim.Optimizer):
            return optimizer

        # Otherwise assume it's a dict of Adam kwargs
        else:
            return torch.optim.Adam(
                params,
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

        return self

    @torch.inference_mode()
    def r2(
        self,
        training_dataloader,
        validation_dataloader=None,
        multioutput='uniform_truncated_average',
        input_data_index=None,
        target_data_index=None
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

        self.training_r2 = r2_score(
            training_dataloader,
            self,
            multioutput=multioutput,
            input_data_idx=input_data_index,
            target_data_idx=target_data_index
        )

        self.validation_r2 = r2_score(
            validation_dataloader,
            self,
            multioutput=multioutput,
            input_data_idx=input_data_index,
            target_data_idx=target_data_index
        )

        return to(self.training_r2, 'cpu'), to(self.validation_r2, 'cpu')

    def _slice_data_and_forward(
        self,
        train_x,
        **kwargs
    ):

        return self.output_model(
            self(
                self.input_data(train_x),
                n_time_steps=self.n_additional_predictions,
                **kwargs
            )
        )

    def output_data(
        self,
        x,
        output_t_plus_one=None,
        loss_offset_only=False,
        no_loss_offset=False,
        **kwargs
    ):
        """
        Process data from DataLoader for output nodes in training.

        This function trims data in the time axis:

        output_t_plus_one means that t will be offset by one
        so that the model will minimze MSE for output of t+1 from intputs
        at time t

        loss_offset means that early times t will be sliced out of outputs
        the data from these earlier times will be provided as input but the
        model will not minimize MSE for them

        :param x: Training data X with ndim >= 3
        :type x: torch.Tensor
        :param output_t_plus_one: _description_, defaults to None
        :type output_t_plus_one: _type_, optional
        :param loss_offset_only: Only process loss_offset,
            defaults to False
        :type loss_offset_only: bool, optional
        :param no_loss_offset: _description_, defaults to False
        :type no_loss_offset: bool, optional
        :return: _description_
        :rtype: _type_
        """

        if x.ndim < 3:
            raise RuntimeError(
                f"Training data must have at least 3 dims; {x.ndim} provided"
            )

        if loss_offset_only:
            _t_plus_one = False
        elif output_t_plus_one is None:
            _t_plus_one = self.output_t_plus_one
        else:
            _t_plus_one = output_t_plus_one

        if no_loss_offset:
            loss_offset = 0
        else:
            loss_offset = self.loss_offset

        _, output_offset = _check_data_offsets(
            x.shape[1],
            _t_plus_one,
            self.n_additional_predictions,
            loss_offset
        )

        # No need to do predictive offsets
        if output_offset == 0:
            return x

        return x[:, output_offset:, ...]

    def output_model(
        self,
        x
    ):
        """
        Process model results so they align with the output_data
        result correctly

        :param x: Model output
        :type x: torch.Tensor
        :return: Output node values for loss function
        :rtype: torch.Tensor
        """

        if isinstance(x, (tuple, list)):
            return tuple(
                self.output_model(_x) for _x in x
            )

        elif x is None:
            return None

        else:
            return self.output_data(
                x,
                output_t_plus_one=False,
                keep_all_dims=True
            )

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

        return x

    def save(
        self,
        file_name,
        **kwargs
    ):
        """
        Save this model to a file

        :param file_name: File name
        :type file_name: str
        """

        _current_device = self._model_device

        to(self, 'cpu')
        write(self, file_name, **kwargs)
        to(self, _current_device)

    def set_training_time(
        self,
        reset=False
    ):
        if reset or self.training_time is None:
            self.training_time = time.time()

    @staticmethod
    def _process_loss(
        existing_loss,
        loss,
        loss_idx=None
    ):
        if loss is None:
            return existing_loss

        loss = np.asanyarray(loss).reshape(1, -1)

        if loss_idx is not None:
            loss_idx = np.asanyarray(loss_idx).reshape(-1)

            _n_losses = max(
                np.max(loss_idx) + 1,
                len(loss_idx),
                loss.shape[1],
                existing_loss.shape[1] if existing_loss is not None else 0
            )

            _loss = np.zeros((1, _n_losses))
            _loss[:, loss_idx] = loss
            loss = _loss

        if existing_loss is None:
            return loss
        else:
            return np.append(
                existing_loss,
                loss,
                axis=0
            )

    def append_loss(
        self,
        training_loss=None,
        training_n=None,
        training_loss_idx=None,
        validation_loss=None,
        validation_n=None,
        validation_loss_idx=None
    ):

        self._training_loss = self._process_loss(
            self._training_loss,
            training_loss,
            training_loss_idx
        )

        self._training_n = self._process_loss(
            self._training_n,
            training_n
        )

        self._validation_loss = self._process_loss(
            self._validation_loss,
            validation_loss,
            validation_loss_idx
        )

        self._validation_n = self._process_loss(
            self._validation_n,
            validation_n
        )

    def _loss_df(self, loss_array):

        if loss_array.size == 0:
            return None
        elif loss_array.ndim == 1:
            loss_array = loss_array.reshape(-1, 1)

        _loss = pd.DataFrame(loss_array).T

        if self._loss_type_names is not None:
            _loss.insert(0, 'loss_model', self._loss_type_names)
        else:
            _loss.insert(0, 'loss_model', self.type_name)

        return _loss

    @staticmethod
    def to_tensor(x):

        return _to_tensor(x)

    @torch.inference_mode()
    def predict(
        self,
        x,
        **kwargs
    ):
        """
        Wraps forward so that it will take a
        Tensor, numpy-like array, or a DataLoader

        :param x: Input data
        :type x: torch.Tensor, np.ndarray, torch.DataLoader
        :return: Model outputs, concatenated on the first axis
            if necessary
        :rtype: torch.Tensor, tuple(torch.Tensor)
        """

        self.eval()
        device = self._model_device

        # Recursive call if x is a DataLoader
        if isinstance(x, DataLoader):
            results = [
                self(
                    to(batch_x, device),
                    **kwargs
                )
                for batch_x in x
            ]

            if len(results) == 1:
                return results[0]

            if isinstance(results[0], torch.Tensor):
                return to(_cat(results, 0), 'cpu')

            else:
                return tuple(
                    to(
                        _cat(
                            [results[i][j] for i in range(len(results))],
                            0
                        ),
                        'cpu'
                    )
                    for j in range(len(results[0]))
                )

        elif not torch.is_tensor(x):
            x = torch.Tensor(x)

        return to(
            self(
                to(x, device),
                **kwargs
            ),
            'cpu'
        )

    @torch.inference_mode()
    def score(
        self,
        dataloader,
        loss_function=torch.nn.MSELoss(),
        reduction='sum',
        **kwargs
    ):

        if dataloader is None:
            return None

        _score = []
        _count = []

        with torch.no_grad():
            for data in stack_dataloaders(dataloader):
                _score.append(
                    loss_function(
                        self._slice_data_and_forward(data),
                        self.output_data(data),
                        **kwargs
                    )
                )
                _count.append(
                    self.input_data(data).shape[0]
                )

        _score = torch.Tensor(_score)
        _count = torch.Tensor(_count)

        if reduction == 'mean':
            _score = torch.sum(
                torch.mul(_score, _count) / torch.sum(_count)
            )
        elif reduction == 'sum':
            _score = torch.sum(_score)
        elif reduction is None:
            pass
        else:
            raise ValueError(
                f'reduction must be `mean`, `sum` or None; {reduction} passed'
            )

        return _score
