import torch
import numpy as np

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
        'biophysical_count',
        'biophysical_decay'
    ]

    _pretrained_count = False
    _pretrained_decay = False
    _pretrained_decay_epoch_delay = 0

    time_dependent_decay = True

    separately_optimize_decay_model = False

    decay_loss = None
    decay_loss_weight = None

    def __init__(
        self,
        prior_network,
        trained_count_model=None,
        decay_model=None,
        separately_optimize_decay_model=False,
        decay_loss=None,
        decay_loss_weight=None,
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
        :param trained_count_model: Pretrained count->count learning model
            to denoise input counts. Will be frozen and not trained during
            the transcriptional model training. None disables.
            Defaults to None
        :type trained_count_model: torch.nn.Module, optional
        :param decay_model: A pretrained decay model which will be frozen
            for transcriptional model. None will create a new decay model
            that will be trained with the transcriptional model.
            False will disable decay model training. Defaults to None
        :type decay_model: torch.nn.Module, False, or None
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

        if trained_count_model is not None:

            if isinstance(trained_count_model, str):
                from .._utils._loader import read
                trained_count_model = read(trained_count_model)

            self._count_model = trained_count_model
            self._pretrained_count = True
            self.freeze(self._count_model)

        else:
            self._count_model = None

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
            self._pretrained_decay = True

        else:

            self._decay_model = DecayModule(
                self.g,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
                time_dependent_decay=time_dependent_decay
            )

            self.time_dependent_decay = time_dependent_decay

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.separately_optimize_decay_model = separately_optimize_decay_model

        self.decay_loss = decay_loss
        self.decay_loss_weight = decay_loss_weight

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        if self._count_model is not None:
            self._count_model.eval()

        return self

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

    @property
    def _do_decay_optimization(self):
        _optimize_decay = self._decay_model is not None
        return _optimize_decay & self.separately_optimize_decay_model

    @staticmethod
    def freeze(model):

        for param in model.parameters():
            param.requires_grad = False

    def forward(
        self,
        x,
        n_time_steps=0,
        return_submodels=False,
        return_counts=False
    ):

        v = self.forward_model(
            x,
            return_submodels=return_submodels
        )

        if n_time_steps == 0 and not return_counts:
            return v

        counts = self.next_count_from_velocity(
            x,
            torch.add(v[0], v[1]) if return_submodels else v
        )

        # Do forward predictions
        if n_time_steps > 0:

            _output_velo = [v]
            _output_count = [counts]

            _x = self.get_last_step(counts)

            # Iterate through the number of steps for prediction
            for i in range(n_time_steps):

                _v = self.forward_model(
                    _x,
                    hidden_state=True,
                    return_submodels=return_submodels
                )

                _x = self.next_count_from_velocity(
                    _x,
                    torch.add(_v[0], _v[1]) if return_submodels else _v
                )

                _output_velo.append(_v)
                _output_count.append(_x)

            if return_counts:
                _output_data = _output_count
            else:
                _output_data = _output_velo

            if return_submodels and not return_counts:

                v = (
                    _cat([d[0] for d in _output_data], x),
                    _cat([d[1] for d in _output_data], x)
                )

            else:
                v = _cat(_output_data, x)

        elif return_counts:
            return counts

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

        # Run the pretrained count model if provided
        x = self.forward_count_model(
            x,
            hidden_state
        )

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

        if self._decay_model is None:
            return torch.zeros_like(x)

        # Get the hidden state of the model if
        # hidden_state is true
        if hidden_state:
            hidden_state = self._decay_model.hidden_state
        else:
            hidden_state = None

        x_negative = self._decay_model(
            x,
            hidden_state=hidden_state,
            return_decay_constants=return_decay_constants
        )

        # Return decay constants unchanged
        if return_decay_constants:
            return x_negative
        else:
            return self.scale_count_to_velocity(x_negative)

    def forward_count_model(
        self,
        x,
        hidden_state=False
    ):
        if self._count_model is not None:

            if hidden_state:
                _hidden = self._count_model.hidden_final
            else:
                _hidden = None

            x = self._count_model(
                x,
                hidden_state=_hidden
            )

        return x

    def train_model(
        self,
        training_dataloader,
        epochs,
        validation_dataloader=None,
        loss_function=torch.nn.MSELoss(),
        optimizer=None
    ):

        # Create separate optimizers for the decay and transcription
        # models if joint_optimize is set
        if self._do_decay_optimization:
            self._decay_model.optimizer = self._decay_model.process_optimizer(
                optimizer
            )

            optimizer = self._transcription_model.process_optimizer(
                optimizer
            )

        # Otherwise use one optimizer for everything
        else:
            optimizer = self.process_optimizer(optimizer)

        super().train_model(
            training_dataloader,
            epochs,
            validation_dataloader,
            loss_function,
            optimizer
        )

    def _training_step(
        self,
        n_epochs,
        train_x,
        optimizer,
        loss_function
    ):

        # Check to see if the decay model should be separately optimized
        _optimize_decay = self._do_decay_optimization

        # Run the decay model training separately
        if _optimize_decay:

            if self._pretrained_decay_epoch_delay is not None:
                _allow_training = n_epochs > self._pretrained_decay_epoch_delay
            else:
                _allow_training = True

            if _allow_training:

                decay_mse = self._decay_model._training_step(
                    n_epochs,
                    train_x,
                    self._decay_model.optimizer,
                    self.decay_loss if self.decay_loss else loss_function,
                    loss_weight=self.decay_loss_weight,
                    x_hat=self(
                        self.input_data(train_x),
                        n_time_steps=self.n_additional_predictions,
                        return_submodels=True
                    )[1]
                )
            else:
                decay_mse = torch.Tensor([0])

        # Get the decay model loss and mix it in with the rest of the loss
        elif self._decay_model is not None:

            decay_mse = self._decay_model._calculate_loss(
                train_x,
                self.decay_loss if self.decay_loss else loss_function,
                loss_weight=self.decay_loss_weight,
                x_hat=self(
                    self.input_data(train_x),
                    n_time_steps=self.n_additional_predictions,
                    return_submodels=True
                )[1]
            )

        # If there is no decay model only use the other losses
        else:
            decay_mse = torch.Tensor([0])

        velocity_mse = self._calculate_loss(
            train_x,
            loss_function
        )

        count_mse = self._calculate_loss(
            train_x,
            loss_function,
            return_counts=True,
            output_kwargs={
                'counts': True
            }
        )

        if _optimize_decay:
            mse = velocity_mse + count_mse

        else:
            mse = velocity_mse + count_mse + decay_mse

        mse.backward()
        optimizer.step()
        optimizer.zero_grad()

        try:
            decay_mse = decay_mse.item()
        except AttributeError:
            pass

        return (
            velocity_mse.item(),
            count_mse.item(),
            decay_mse
        )

    def _calculate_validation_loss(
        self,
        validation_dataloader,
        loss_function
    ):
        # Get validation losses during training
        # if validation data was provided
        if validation_dataloader is not None:

            _validation_batch_losses = []

            with torch.no_grad():
                for val_x in validation_dataloader:

                    full_mse = self._calculate_loss(
                        val_x,
                        loss_function
                    ).item()

                    if self._do_decay_optimization:
                        decay_mse = self._decay_model._calculate_loss(
                            val_x,
                            loss_function
                        ).item()

                        _validation_batch_losses.append(
                            (full_mse + decay_mse, full_mse, decay_mse)
                        )

                    else:
                        _validation_batch_losses.append(
                            full_mse
                        )

            return np.mean(
                np.array(_validation_batch_losses),
                axis=0
            )

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
                if self._decay_model is None:
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

    def predict_perturbation(
        self,
        x,
        n_time_steps,
        perturbation,
        return_submodels=False,
        return_counts=False
    ):

        # Get results from the full model
        self._transcription_model.set_drop_tfs(None)

        if self._decay_model is not None:
            _x_bar = self(
                x,
                return_counts=True,
                n_time_steps=n_time_steps
            )
        else:
            _x_bar = None

        # Set the transcription model for perturbation
        self._transcription_model.set_drop_tfs(perturbation)

        _perturb_estimate = self(
            x,
            n_time_steps=n_time_steps,
            x_decay=_x_bar,
            return_submodels=return_submodels,
            return_counts=return_counts
        )

        self._transcription_model.set_drop_tfs(None)

        return _perturb_estimate

    @staticmethod
    def _x_decay(x_decay, len, start=0):
        if x_decay is None:
            return None

        return x_decay[:, start:start + len, ...]

    @staticmethod
    def _ntime(x):
        return x.shape[1]


def _cat(_data, x):
    return torch.cat(
        _data,
        dim=x.ndim - 2
    )
