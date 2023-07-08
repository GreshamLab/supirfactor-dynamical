import torch

from .recurrent_models import TFRNNDecoder
from ._base_trainer import _TrainingMixin
from ._base_velocity_model import (
    DecayModule,
    _VelocityMixin
)


class SupirFactorDynamical(
    torch.nn.Module,
    _VelocityMixin,
    _TrainingMixin
):

    name = 'dynamical'

    _velocity_scale_vector = None
    _expression_scale_vector = None

    def __init__(
        self,
        trained_count_model,
        prior_network,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        transcription_model=None
    ):
        super().__init__()

        if isinstance(trained_count_model, str):
            from .._utils._loader import read
            trained_count_model = read(trained_count_model)

        self._count_model = trained_count_model

        # Freeze trained count model
        for param in self._count_model.parameters():
            param.requires_grad = False

        if transcription_model is None:
            transcription_model = TFRNNDecoder

        self._transcription_model = transcription_model(
            prior_network=prior_network,
            use_prior_weights=use_prior_weights,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate
        )

        self._decay_model = DecayModule(
            self._count_model.g,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate
        )

    def set_scaling(
        self,
        velocity_scale_vector=None,
        expression_scale_vector=None
    ):

        if velocity_scale_vector is not None:
            self._velocity_scale_vector = torch.Tensor(
                velocity_scale_vector
            )

        if expression_scale_vector is not None:
            self._expression_scale_vector = torch.Tensor(
                expression_scale_vector
            )

        return self

    def forward(
        self,
        x,
        n_time_steps=0,
        return_submodels=False
    ):
        """
        Velocity from Count Data

        :param x: _description_
        :type x: _type_
        :param n_time_steps: _description_, defaults to 0
        :type n_time_steps: int, optional
        :param return_submodels: _description_, defaults to False
        :type return_submodels: bool, optional
        :return: _description_
        :rtype: _type_
        """

        # Run the pretrained count model
        x = self._count_model(x, n_time_steps=n_time_steps)

        # Run the transcriptional model
        x_positive = self._transcription_model(x)

        # Run the decay model
        x_negative = self._decay_model(
            x,
            velocity_scale_vector=self._velocity_scale_vector,
            expression_scale_vector=self._expression_scale_vector
        )

        if return_submodels:
            return x_positive, x_negative

        else:
            return torch.add(x_positive, x_negative)

    @torch.inference_mode()
    def counts(
        self,
        x,
        n_time_steps=0
    ):

        with torch.no_grad():
            # Run the pretrained count model
            return self._count_model(
                self.input_data(x),
                n_time_steps=n_time_steps
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
