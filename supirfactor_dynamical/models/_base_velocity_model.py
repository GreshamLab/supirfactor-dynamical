import torch

from .recurrent_models import TFRNNDecoder
from ._base_trainer import _TrainingMixin


class _VelocityMixin:

    _velocity_model = True

    def input_data(self, x, **kwargs):

        return super().input_data(x[..., 0], **kwargs)

    def output_data(self, x, keep_all_dims=False, **kwargs):

        if keep_all_dims:
            return super().output_data(x, **kwargs)
        else:
            return super().output_data(x[..., 1], **kwargs)


class DecayModule(torch.nn.Module):

    def __init__(
        self,
        g,
        k=50,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0
    ):
        super().__init__()

        self._encoder = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout_rate),
            torch.nn.Linear(
                g,
                k,
                bias=False
            ),
            torch.nn.Sigmoid()
        )

        self._intermediate = torch.nn.RNN(
            k,
            k,
            1,
            bias=False
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Dropout(hidden_dropout_rate),
            torch.nn.Linear(
                k,
                g
            ),
            torch.nn.ReLU(),
            lambda x: torch.mul(x, -1.0)
        )

    def forward(
        self,
        x,
        hidden_state=None
    ):

        _x = self._encoder(x)
        _x, self.hidden_state = self._intermediate(_x, hidden_state)
        _x = self._decoder(_x)

        return torch.mul(x, _x)


class SupirFactorDynamical(
    torch.nn.Module,
    _VelocityMixin,
    _TrainingMixin
):

    def __init__(
        self,
        trained_count_model,
        prior_network,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
    ):
        super().__init__()

        self._count_model = trained_count_model

        # Freeze trained count model
        for param in self._count_model.parameters():
            param.requires_grad = False

        self._transcription_model = TFRNNDecoder(
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

    def forward(
        self,
        x,
        n_time_steps=None,
        return_submodels=False
    ):

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
