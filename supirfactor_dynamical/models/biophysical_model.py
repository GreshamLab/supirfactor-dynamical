import torch

from .recurrent_models import TFRNNDecoder
from ._base_model import _TFMixin
from ._base_trainer import _TrainingMixin
from ._base_velocity_model import _VelocityMixin
from .decay_model import DecayModule


class SupirFactorBiophysical(
    torch.nn.Module,
    _VelocityMixin,
    _TFMixin,
    _TrainingMixin
):

    type_name = 'biophysical'

    _pretrained_decay = False
    _pretrained_count = False

    time_dependent_decay = True

    def __init__(
        self,
        prior_network,
        trained_count_model=None,
        decay_model=None,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        transcription_model=None,
        time_dependent_decay=True,
        output_relu=False
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

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        if decay_model is False:

            self._decay_model = None

        elif decay_model is not None:

            self._decay_model = decay_model
            self._pretrained_decay = True
            self.freeze(self._decay_model)

        else:

            self._decay_model = DecayModule(
                self.g,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
                time_dependent_decay=time_dependent_decay
            )

            self.time_dependent_decay = time_dependent_decay

        self.output_relu = output_relu

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        if self._count_model is not None:
            self._count_model.eval()

        if self._pretrained_decay:
            self._decay_model.eval()

    @staticmethod
    def freeze(model):

        for param in model.parameters():
            param.requires_grad = False

    def forward(
        self,
        x,
        n_time_steps=0,
        return_submodels=False
    ):

        v = self.forward_model(x, return_submodels=return_submodels)

        if n_time_steps > 0:

            _x = x[:, [-1], :] if x.ndim == 3 else x[[-1], :]
            _v = v[:, [-1], :] if x.ndim == 3 else v[[-1], :]

            _output_data = [v]

            for _ in range(n_time_steps):

                if return_submodels:
                    _x = self.next_count_from_velocity(
                        _x,
                        torch.add(_v[0], _v[1])
                    )
                else:
                    _x = self.next_count_from_velocity(
                        _x,
                        _v
                    )

                _v = self.forward_model(
                    _x,
                    hidden_state=True,
                    return_submodels=return_submodels
                )

                _output_data.append(_v)

            if return_submodels:

                v = (
                    torch.cat(
                        [d[0] for d in _output_data],
                        dim=x.ndim - 2
                    ),
                    torch.cat(
                        [d[1] for d in _output_data],
                        dim=x.ndim - 2
                    )
                )

            else:
                v = torch.cat(
                    _output_data,
                    dim=x.ndim - 2
                )

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
        if self._count_model is not None:

            if hidden_state:
                _hidden = self._count_model.hidden_final
            else:
                _hidden = None

            x = self._count_model(
                x,
                hidden_state=_hidden
            )

        # Run the transcriptional model
        if hidden_state:
            _hidden = self._transcription_model.hidden_final
        else:
            _hidden = None

        x_positive = self._transcription_model(
            x,
            hidden_state=_hidden
        )

        # Run the decay model
        if self._decay_model is not None:
            if hidden_state:
                _hidden = self._decay_model.hidden_state
            else:
                _hidden = None

            x_negative = self._decay_model(
                x,
                hidden_state=_hidden
            )
        else:
            x_negative = None

        if return_submodels:
            return x_positive, x_negative

        elif x_negative is None:
            return x_positive

        else:
            return torch.add(x_positive, x_negative)

    @torch.inference_mode()
    def counts(
        self,
        x,
        n_time_steps=0
    ):

        if self._count_model is not None:
            with torch.no_grad():
                # Run the pretrained count model
                return self._count_model(
                    self.input_data(x),
                    n_time_steps=n_time_steps
                )

        elif n_time_steps == 0:
            return self.input_data(x)

        else:
            raise RuntimeError(
                "No pretrained count model available for prediction"
            )

    @torch.inference_mode()
    def velocity(
        self,
        x,
        n_time_steps=0
    ):

        with torch.no_grad():
            return self(
                self.input_data(x),
                n_time_steps=n_time_steps
            )

    @torch.inference_mode()
    def decay(
        self,
        x,
        n_time_steps=0,
        return_decay_constants=False
    ):

        with torch.no_grad():
            return self._decay_model(
                self.counts(x, n_time_steps=n_time_steps),
                return_decay_constants=return_decay_constants
            )

    @torch.inference_mode()
    def transcription(
        self,
        x,
        n_time_steps=0
    ):

        with torch.no_grad():
            return self._transcription_model(
                self.counts(x, n_time_steps=n_time_steps)
            )

    @torch.inference_mode()
    def erv(
        self,
        data_loader,
        **kwargs
    ):

        def _erv_input_wrapper(dl):

            for data in dl:

                if self._count_model is None:
                    yield self.input_data(data)

                # If there's a count model included, run it as the input
                # to the ERV model
                else:
                    yield self._count_model(
                        self.input_data(data),
                        n_time_steps=self._count_model.n_additional_predictions
                    )

        def _erv_output_wrapper(dl, input_wrapper):

            for data, input_data in zip(dl, input_wrapper):

                # If there's a decay model included, run it and subtract it
                # from the output for the ERV model
                if self._decay_model is None:
                    yield self.output_data(data)

                else:
                    yield torch.subtract(
                        self.output_data(data),
                        self._decay_model(input_data)
                    )

        return self._transcription_model.erv(
            _erv_input_wrapper(data_loader),
            output_data_loader=_erv_output_wrapper(
                data_loader,
                _erv_input_wrapper(data_loader)
            ),
            **kwargs
        )

    def output_weights(self, *args, **kwargs):
        return self._transcription_model.output_weights(*args, **kwargs)
