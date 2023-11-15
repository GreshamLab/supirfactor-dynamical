import torch

from .recurrent_models import TFRNNDecoder
from ._base_model import _TFMixin
from ._model_mixins import (
    _VelocityMixin,
    _TrainingMixin,
    _TimeOffsetMixinRecurrent
)
from .decay_model import DecayModule
from .._utils.misc import (_cat, _add)
from .._utils._math import _calculate_rss


class SupirFactorBiophysical(
    torch.nn.Module,
    _VelocityMixin,
    _TimeOffsetMixinRecurrent,
    _TFMixin,
    _TrainingMixin
):

    type_name = 'biophysical'

    _loss_type_names = [
        'biophysical_velocity',
        'biophysical_decay_rate'
    ]

    separately_optimize_decay_model = False
    decay_epoch_delay = 0
    decay_k = 20

    @property
    def has_decay(self):
        return self._decay_model is not None

    def __init__(
        self,
        prior_network=None,
        decay_model=None,
        decay_epoch_delay=None,
        decay_k=20,
        separately_optimize_decay_model=False,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        transcription_model=None,
        activation='relu',
        output_activation='relu'
    ):
        """
        Biophysical deep learning model for transcriptional regulatory
        network inference.

        :param prior_network: Prior knowledge network to constrain affine
            transformation into the TFA layer. Genes x TFs.
        :type prior_network: pd.DataFrame
        :param decay_model: A pretrained decay model which will be frozen
            for transcriptional model. None will create a new decay model
            that will be trained with the transcriptional model.
            False will disable decay model training. Defaults to None
        :type decay_model: torch.nn.Module, False, or None
        :param decay_epoch_delay: Number of epochs to freeze decay model
            during training. Allows transcription model to fit first.
            None disables. Defaults to None.
        :type decay_epoch_delay: int, optional
        :param decay_k: Width of hidden layers in decay model. Ignored if
            a model is provided to `decay_model`. Defaults to 20.
        :type decay_k: int, optional
        :param use_prior_weights: Use prior weights for the transcriptional
            embedding into TFA, defaults to False
        :type use_prior_weights: bool, optional
        :param input_dropout_rate: Input dropout for each model,
            defaults to 0.5
        :type input_dropout_rate: float, optional
        :param hidden_dropout_rate: Hiddenl layer dropout for each model,
            defaults to 0.0
        :type hidden_dropout_rate: float, optional
        :param transcription_model: Model to use for transcriptional
            output, None uses the standard RNN, defaults to None
        :type transcription_model: torch.nn.Module, optional
        :param time_dependent_decay: Fit a time-dependent decay constant
            instead of a single decay constant per gene, defaults to True
        :type time_dependent_decay: bool, optional
        """
        super().__init__()

        self.prior_network = self.process_prior(prior_network)

        self.activation = activation
        self.output_activation = output_activation

        self.decay_epoch_delay = decay_epoch_delay
        self.separately_optimize_decay_model = separately_optimize_decay_model

        if decay_model is False:
            output_activation = None

        if transcription_model is None:
            transcription_model = TFRNNDecoder

        self._transcription_model = transcription_model(
            prior_network=prior_network,
            use_prior_weights=use_prior_weights,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            activation=activation,
            output_activation=output_activation
        )

        if decay_model is False:

            self._decay_model = None

        elif decay_model is not None:

            if isinstance(decay_model, str):
                from .._io._loader import read
                decay_model = read(decay_model)

            self._decay_model = decay_model
            self.decay_k = self._decay_model.hidden_layer_width

        else:

            self._decay_model = DecayModule(
                n_genes=self.g,
                hidden_layer_width=decay_k,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
            )

            self.decay_k = decay_k

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def set_dropouts(
        self,
        input_dropout_rate,
        hidden_dropout_rate
    ):

        self._transcription_model.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        return super().set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def set_time_parameters(
            self,
            output_t_plus_one=None,
            n_additional_predictions=None,
            loss_offset=None
    ):

        if output_t_plus_one is not None and output_t_plus_one:
            raise ValueError(
                "Biophysical model does not support offset training; "
                "this model uses velocity for forward predictions"
            )

        return super().set_time_parameters(
            n_additional_predictions=n_additional_predictions,
            loss_offset=loss_offset
        )

    def set_scaling(
            self,
            count_scaling=False,
            velocity_scaling=False
    ):

        if self.has_decay:
            self._decay_model.set_scaling(
                count_scaling,
                velocity_scaling
            )

        return super().set_scaling(
            count_scaling,
            velocity_scaling
        )

    def forward(
        self,
        x,
        n_time_steps=0,
        return_submodels=False,
        unmodified_counts=False,
        hidden_state=False
    ):
        """
        Run model on data x

        :param x: Input data of (N, L, G) where N is batch, L is
            sequence length, and G is gene features
        :type x: torch.Tensor
        :param n_time_steps: Number of time steps to predict from, starting
            at the end of the input data, defaults to 0
        :type n_time_steps: int, optional
        :param return_submodels: Return velocity positive & negative submodels
            as a tuple of velocities, defaults to False
        :type return_submodels: bool, optional
        :param unmodified_counts: Return unmodified counts from input data x,
            instead of the next time step prediction within the input sequence,
            defaults to False
        :type unmodified_counts: bool, optional
        :param hidden_state: Use the existing hidden state of the model
            instead of reinitializing it, defaults to False
        :type hidden_state: bool, optional
        :return: Velocity, Count, and Decay Rate predictions, depending on
            the return flags provided
        :rtype: torch.Tensor or tuple(torch.Tensor)
        """

        # Calculate model predictions for the data provided
        counts, v, d, tfa = self.forward_time_step(
            x,
            return_submodels=return_submodels,
            hidden_state=hidden_state,
            return_unmodified_counts=unmodified_counts
        )

        _output_velo = [v]
        _output_count = [counts]
        _output_decay = [d]
        _output_tfa = [tfa]

        # Do forward predictions starting from the last value in the
        # data provided for n_time_steps time units
        _x = counts[:, [-1], :]

        # Iterate through the number of steps for prediction
        for _ in range(n_time_steps):

            # Call the model on a single time point to get velocity
            # and update the counts by adding velocity
            _x, _v, _d, _tfa = self.forward_time_step(
                _x,
                hidden_state=True,
                return_submodels=return_submodels,
                return_unmodified_counts=False
            )

            _output_velo.append(_v)
            _output_count.append(_x)
            _output_decay.append(_d)
            _output_tfa.append(_tfa)

        return tuple(
            _cat(output, output_dim) if n_time_steps > 0 else output[0]
            for output, output_dim in (
                (_output_velo, 1),
                (_output_count, 1),
                (_output_decay, 1),
                (_output_tfa, 1)
            )
        )

    def forward_time_step(
        self,
        x,
        hidden_state=True,
        return_submodels=False,
        return_unmodified_counts=False
    ):

        _v, _d, _tfa = self.forward_model(
            x,
            hidden_state=hidden_state,
            return_submodels=return_submodels
        )

        if not return_unmodified_counts:
            x = self.next_count_from_velocity(
                x,
                _v
            )

        return x, _v, _d, _tfa

    def forward_model(
        self,
        x,
        return_submodels=False,
        hidden_state=False
    ):
        """
        Velocity from Count Data

        :param x: Count data tensor
        :type x: torch.Tensor
        :param n_time_steps: Time steps to predict forward, defaults to 0
        :type n_time_steps: int, optional
        :param return_submodels: Return positive and negative components of
            prediction instead of adding them, defaults to False
        :type return_submodels: bool, optional
        :return: Predicted velocity tensor
        :rtype: torch.Tensor
        """

        # Run the transcriptional model
        x_positive, tfa_positive = self.forward_transcription_model(
            x,
            hidden_state
        )

        # Run the decay model
        x_negative, x_decay_rate = self.forward_decay_model(
            x,
            hidden_state=hidden_state
        )

        if return_submodels:
            return (x_positive, x_negative), x_decay_rate, tfa_positive

        elif x_negative is None:
            return x_positive, x_decay_rate, tfa_positive

        else:
            return _add(x_positive, x_negative), x_decay_rate, tfa_positive

    def forward_transcription_model(
        self,
        x,
        hidden_state=False
    ):

        if hidden_state:
            _hidden = self._transcription_model.hidden_final
        else:
            _hidden = None

        return self._transcription_model(
            x,
            hidden_state=_hidden,
            return_tfa=True
        )

    def forward_decay_model(
        self,
        x,
        hidden_state=False,
    ):

        if not self.has_decay:
            return None, None

        return self._decay_model(
            x,
            hidden_state=hidden_state,
            return_decay_constants=True
        )

    def train_model(
        self,
        training_dataloader,
        epochs,
        validation_dataloader=None,
        loss_function=torch.nn.MSELoss(),
        optimizer=None
    ):

        # Create separate optimizers for the decay and transcription
        # models and pass them as tuple
        optimizer = (
            self._transcription_model.process_optimizer(
                optimizer
            ),
            self._decay_model.process_optimizer(
                optimizer
            ) if self.has_decay else False,
            self.process_optimizer(
                optimizer
            )
        )

        return super().train_model(
            training_dataloader,
            epochs,
            validation_dataloader,
            loss_function,
            optimizer
        )

    def _training_step(
        self,
        epoch_num,
        train_x,
        optimizer,
        loss_function
    ):
        """
        Do a training step for the transcription model
        and for the decay model if conditions for that
        are satisfied

        :param epoch_num: Epoch number
        :type epoch_num: int
        :param train_x: Training data (N, L, G, ...)
        :type train_x: torch.Tensor
        :param optimizer: Tuple of optimizers where
            optimizer[0] is for transcription model and
            optimizer[1] is for decay model and
            optimizer[2] is for both combined
        :type optimizer: torch.optim, torch.optim
        :param loss_function: Loss function
        :type loss_function: torch.nn.Loss
        :return: Returns transcription model loss,
            decay loss, and decay tuning loss
        :rtype: float, float, float
        """

        if self.separately_optimize_decay_model:

            decay_loss = self._training_step_decay(
                epoch_num,
                train_x,
                optimizer[1],
                loss_function,
                compare_decay_data=True
            )

        else:

            decay_loss = 0.

        loss = self._training_step_joint(
                epoch_num,
                train_x,
                optimizer[2] if self._decay_optimize(epoch_num) else
                optimizer[0],
                loss_function
            )

        return loss, decay_loss

    def _training_step_joint(
        self,
        epoch_num,
        train_x,
        optimizer,
        loss_function
    ):

        return super()._training_step(
            epoch_num,
            train_x,
            optimizer,
            loss_function,
            input_x=self._slice_data_and_forward(train_x)
        )

    def _training_step_decay(
        self,
        epoch_num,
        train_x,
        optimizer,
        loss_function,
        compare_decay_data=False
    ):

        if not self._decay_optimize(epoch_num):
            return 0

        # Get model output for training data
        pos, neg = self._slice_data_and_forward(
            train_x,
            return_submodels=True
        )

        if compare_decay_data:
            _compare_x = self.output_data(
                train_x,
                decay=True
            )
        else:
            _compare_x = torch.sub(
                self.output_data(train_x),
                pos.detach()
            )

        return super()._training_step(
            epoch_num,
            train_x,
            optimizer,
            loss_function,
            input_x=neg,
            target_x=_compare_x
        )

    def _calculate_all_losses(
        self,
        x,
        loss_function,
    ):

        # Get model output for training data
        (pos, neg) = self._slice_data_and_forward(
            x,
            return_submodels=True
        )

        loss = loss_function(
            _add(pos, neg),
            self.output_data(x)
        ).item()

        if (
            self.has_decay and
            self.separately_optimize_decay_model
        ):
            decay_rate_loss = loss_function(
                neg,
                self.output_data(x, decay=True)
            ).item()
        else:
            decay_rate_loss = 0

        return loss, decay_rate_loss

    def _calculate_error(
        self,
        input_data,
        output_data,
        n_additional_predictions
    ):

        if self._decay_model is None:
            return self._transcription_model._calculate_error(
                input_data,
                output_data,
                n_additional_predictions
            )

        self.set_drop_tfs(None)

        # Get TFA
        with torch.no_grad():
            predict_velo, _, decay_rates, _ = self(
                input_data,
                n_time_steps=n_additional_predictions
            )

        full_rss = _calculate_rss(
            predict_velo,
            output_data
        )

        del predict_velo

        rss = torch.zeros((self.g, self.k))
        # For each node in the latent layer,
        # zero all values in the data and then
        # decode to full expression data
        for ik in range(self.k):

            self.set_drop_tfs(self.prior_network_labels[1][ik])

            with torch.no_grad():
                latent_dropout = self._perturbed_model_forward(
                    input_data,
                    decay_rates,
                    n_time_steps=self.n_additional_predictions
                )[0]

            rss[:, ik] = _calculate_rss(
                latent_dropout,
                output_data
            )

        self.set_drop_tfs(None)

        return full_rss, rss

    def _perturbed_model_forward(
        self,
        data,
        decay_rates,
        n_time_steps=0,
        unmodified_counts=False,
        return_submodels=False
    ):

        _L = data.shape[1]

        # Perturbed transcription
        _v, _tfa = self._perturbed_velocities(
            data,
            decay_rates[:, 0:_L, :]
        )

        if unmodified_counts:
            counts = data
        else:
            counts = self.next_count_from_velocity(
                data,
                _v
            )

        # Do forward predictions
        _output_velo = [_v]
        _output_count = [counts]
        _output_tfa = [_tfa]

        _x = counts[:, [-1], :]

        # Iterate through the number of steps for prediction
        for i in range(n_time_steps):

            _offset = _L + i

            _v, _tfa = self._perturbed_velocities(
                _x,
                decay_rates[:, _offset:_offset + 1, :],
                hidden_state=True
            )

            _x = self.next_count_from_velocity(
                _x,
                _v
            )

            _output_velo.append(_v)
            _output_count.append(_x)
            _output_tfa.append(_tfa)

        self.set_drop_tfs(None)

        if n_time_steps > 0:
            _output_velo = _cat(_output_velo, 1)
            _output_count = _cat(_output_count, 1)
            _output_tfa = _cat(_output_tfa, 1)
        else:
            _output_velo = _output_velo[0]
            _output_count = _output_count[0]
            _output_tfa = _output_tfa[0]

        if not return_submodels:
            _output_velo = _add(
                _output_velo[0],
                _output_velo[1]
            )

        return (
            _output_velo,
            _output_count,
            decay_rates,
            _output_tfa
        )

    def _perturbed_velocities(
        self,
        data,
        decay_rates,
        return_submodels=True,
        **kwargs
    ):

        # Perturbed transcription
        (x_fwd, _), _, _, fwd_tfa = self(
            data,
            return_submodels=True,
            **kwargs
        )

        x_rev = self.rescale_velocity(
            torch.multiply(
                data,
                decay_rates
            )
        )

        if return_submodels:
            return (x_fwd, x_rev), fwd_tfa
        else:
            return _add(x_fwd, x_rev), fwd_tfa

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
            )[0]
        )

    def output_weights(self, *args, **kwargs):
        return self._transcription_model.output_weights(*args, **kwargs)

    def set_drop_tfs(self, *args, **kwargs):
        self._transcription_model.set_drop_tfs(*args, **kwargs)
        return self

    def next_count_from_velocity(self, x, v):
        if isinstance(v, tuple):
            v = _add(v[0], v[1])

        return super().next_count_from_velocity(x, v)

    def _decay_optimize(self, n):
        if not self.has_decay:
            return False
        elif n is True:
            return True
        elif self.decay_epoch_delay is not None:
            return n > self.decay_epoch_delay
        else:
            return True
