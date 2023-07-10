import torch

from .recurrent_models import TFRNNDecoder
from ._base_trainer import _TrainingMixin
from ._base_velocity_model import (
    DecayModule,
    _VelocityMixin
)


class SupirFactorBiophysical(
    torch.nn.Module,
    _VelocityMixin,
    _TrainingMixin
):

    name = 'biophysical'

    def __init__(
        self,
        trained_count_model,
        prior_network,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        transcription_model=None,
        time_dependent_decay=True
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
            hidden_dropout_rate=hidden_dropout_rate,
            output_relu=False
        )

        self._decay_model = DecayModule(
            self._count_model.g,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            time_dependent_decay=time_dependent_decay,
            relu=False
        )

        # Use leakyrelu to address vanishing gradients
        self._transcription_model._decoder.append(
            torch.nn.LeakyReLU(1e-3)
        )

        self._decay_model._decoder.append(
            torch.nn.LeakyReLU(1e-3)
        )

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self._count_model.eval()

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
        x_negative = self._decay_model(x)

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

    @torch.inference_mode()
    def erv(
        self,
        data_loader,
        **kwargs
    ):

        def _count_wrapper():

            for data in data_loader:
                yield self.counts(data)

        return self._transcription_model(
            _count_wrapper(),
            **kwargs
        )

    def output_weights(self, *args, **kwargs):
        self._count_model.output_weights()
