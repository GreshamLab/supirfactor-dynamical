import torch

from .recurrent_models import TFRNNDecoder
from ._base_trainer import _TrainingMixin
from ._base_model import _PriorMixin
from ._writer import write
from ._base_velocity_model import (
    DecayModule,
    _VelocityMixin
)


class SupirFactorBiophysical(
    torch.nn.Module,
    _VelocityMixin,
    _PriorMixin,
    _TrainingMixin
):

    type_name = 'biophysical'

    _pretrained_decay = False

    def __init__(
        self,
        prior_network,
        trained_count_model=None,
        trained_decay_model=None,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        transcription_model=None,
        time_dependent_decay=True
    ):
        super().__init__()

        self.prior_network = self.process_prior(prior_network)

        if trained_count_model is not None:

            if isinstance(trained_count_model, str):
                from .._utils._loader import read
                trained_count_model = read(trained_count_model)

            self._count_model = trained_count_model
            self.freeze(self._count_model)

        else:
            self._count_model = None

        if transcription_model is None:
            transcription_model = TFRNNDecoder

        self._transcription_model = transcription_model(
            prior_network=prior_network,
            use_prior_weights=use_prior_weights,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate
        )

        if trained_decay_model is not None:

            self._decay_model = trained_decay_model
            self._pretrained_decay = True
            self.freeze(self._decay_model)

        else:

            self._decay_model = DecayModule(
                self.g,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
                time_dependent_decay=time_dependent_decay
            )

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

        # Run the pretrained count model if provided
        if self._count_model is not None:
            x = self._count_model(x, n_time_steps=n_time_steps)

        elif n_time_steps != 0:
            raise RuntimeError(
                "No pretrained count model available for prediction"
            )

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

        def _count_wrapper():

            for data in data_loader:
                yield self.counts(data)

        return self._transcription_model.erv(
            _count_wrapper(),
            **kwargs
        )

    def output_weights(self, *args, **kwargs):
        return self._transcription_model.output_weights(*args, **kwargs)

    def save(self, file_name):
        write(self, file_name)
