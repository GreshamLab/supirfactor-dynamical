import torch

from .recurrent_models import TFRNNDecoder
from ._base_model import _TFMixin
from ._base_trainer import (
    _TrainingMixin,
    _TimeOffsetMixinRecurrent
)
from ._base_velocity_model import _VelocityMixin
from .decay_model import DecayModule


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
        'biophysical_decay'
    ]

    separately_optimize_decay_model = False
    time_dependent_decay = True
    decay_epoch_delay = 0
    decay_k = 20

    @property
    def has_decay(self):
        return self._decay_model is not None

    def __init__(
        self,
        prior_network,
        decay_model=None,
        decay_epoch_delay=None,
        decay_k=20,
        separately_optimize_decay_model=False,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        transcription_model=None,
        time_dependent_decay=True,
        output_relu=None
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

        if transcription_model is None:
            transcription_model = TFRNNDecoder

        self._transcription_model = transcription_model(
            prior_network=prior_network,
            use_prior_weights=use_prior_weights,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            output_relu=decay_model is not False
        )

        if decay_model is False:

            self._decay_model = None

        elif decay_model is not None:

            if isinstance(decay_model, str):
                from .._utils._loader import read
                decay_model = read(decay_model)

            self._decay_model = decay_model
            self.decay_k = self._decay_model.k
            self.time_dependent_decay = self._decay_model.time_dependent_decay

        else:

            self._decay_model = DecayModule(
                self.g,
                decay_k,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
                time_dependent_decay=time_dependent_decay
            )

            self.decay_k = decay_k
            self.time_dependent_decay = time_dependent_decay

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.decay_epoch_delay = decay_epoch_delay
        self.separately_optimize_decay_model = separately_optimize_decay_model

    def set_dropouts(
        self,
        input_dropout_rate,
        hidden_dropout_rate
    ):

        self._transcription_model.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        if self.has_decay:
            self._decay_model.set_dropouts(
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
        return_counts=False,
        hidden_state=False
    ):

        v = self.forward_model(
            x,
            return_submodels=return_submodels,
            hidden_state=hidden_state
        )

        if n_time_steps == 0 and not return_counts:
            return v

        counts = self.next_count_from_velocity(
            x,
            v
        )

        if n_time_steps == 0 and return_counts:
            return counts

        # Do forward predictions
        else:

            _output_velo = [v]
            _output_count = [counts]

            _x = self.get_last_step(counts)

            # Iterate through the number of steps for prediction
            for _ in range(n_time_steps):

                _v = self.forward_model(
                    _x,
                    hidden_state=True,
                    return_submodels=return_submodels
                )

                _x = self.next_count_from_velocity(
                    _x,
                    _v
                )

                _output_velo.append(_v)
                _output_count.append(_x)

            if return_counts:
                _output_data = _output_count
            else:
                _output_data = _output_velo

            v = _cat(_output_data, x)

        return v

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
        x_positive = self.forward_transcription_model(
            x,
            hidden_state
        )

        # Run the decay model
        x_negative = self.forward_decay_model(
            x,
            hidden_state=hidden_state
        )

        if return_submodels:
            return x_positive, x_negative

        elif x_negative is None:
            return x_positive

        else:
            return torch.add(x_positive, x_negative)

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
            hidden_state=_hidden
        )

    def forward_decay_model(
        self,
        x,
        hidden_state=False,
        return_decay_constants=False
    ):

        if not self.has_decay:
            return torch.zeros_like(x)

        return self._decay_model(
            x,
            hidden_state=hidden_state,
            return_decay_constants=return_decay_constants
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
        # models and pass them as tuples

        if self.has_decay:
            _decay_optimizer = self._decay_model.process_optimizer(
                optimizer
            )
        else:
            _decay_optimizer = False

        optimizer = (
            self.process_optimizer(optimizer),
            self._transcription_model.process_optimizer(
                optimizer
            ),
            _decay_optimizer
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

        if not isinstance(optimizer, tuple):
            main_optimizer = optimizer
        elif self._decay_optimize_epoch(epoch_num):
            main_optimizer = optimizer[0]
        else:
            main_optimizer = optimizer[1]

        full_loss = super()._training_step(
            epoch_num,
            train_x,
            main_optimizer,
            loss_function
        )

        if self._decay_optimize_separate(epoch_num):
            decay_loss = self._decay_model._training_step(
                epoch_num,
                train_x,
                optimizer[2],
                loss_function
            )
        else:
            decay_loss = 0.

        return (full_loss, decay_loss)

    @torch.inference_mode()
    def erv(
        self,
        data_loader,
        **kwargs
    ):

        self.eval()

        def _erv_input_wrapper(dl):

            for data in dl:

                yield self(
                    self.input_data(data),
                    n_time_steps=self.n_additional_predictions,
                    return_counts=True
                )

        def _erv_output_wrapper(dl):

            for data in dl:

                _data = self.output_data(
                    data,
                    no_loss_offset=True
                )

                # If there's a decay model included, run it and subtract it
                # from the output for the ERV model
                if not self.has_decay:
                    yield _data

                else:

                    _decay = self(
                        self.input_data(data),
                        n_time_steps=self.n_additional_predictions,
                        return_submodels=True
                    )[1]

                    yield torch.subtract(
                        _data,
                        self.output_data(
                            _decay,
                            no_loss_offset=True,
                            keep_all_dims=True
                        )
                    )

        return self._transcription_model.erv(
            _erv_input_wrapper(data_loader),
            output_data_loader=_erv_output_wrapper(
                data_loader
            ),
            **kwargs
        )

    def output_weights(self, *args, **kwargs):
        return self._transcription_model.output_weights(*args, **kwargs)

    @staticmethod
    def get_last_step(x):
        return x[:, [-1], :] if x.ndim == 3 else x[[-1], :]

    def next_count_from_velocity(self, x, v):
        if isinstance(v, tuple):
            v = torch.add(v[0], v[1])

        return super().next_count_from_velocity(x, v)

    def _decay_optimize_epoch(self, n):
        if self.decay_epoch_delay is not None:
            return n > self.decay_epoch_delay
        else:
            return True

    def _decay_optimize_separate(self, n):
        if not self.has_decay:
            return False
        elif not self.separately_optimize_decay_model:
            return False
        else:
            return self._decay_optimize_epoch(n)


def _cat(_data, x):

    if isinstance(_data[0], tuple):
        return (
            _cat([d[0] for d in _data], x),
            _cat([d[1] for d in _data], x)
        )

    else:
        return torch.cat(
            _data,
            dim=x.ndim - 2
        )
