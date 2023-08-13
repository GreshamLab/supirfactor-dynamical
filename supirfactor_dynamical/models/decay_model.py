import torch

from ._base_trainer import (
    _TrainingMixin
)

from ._model_mixins import (
    _ScalingMixin
)


class DecayModule(
    torch.nn.Module,
    _TrainingMixin,
    _ScalingMixin
):

    type_name = 'decay'

    hidden_state = None

    time_dependent_decay = True

    g = None
    k = None

    def __init__(
        self,
        g,
        k=20,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        time_dependent_decay=True
    ):
        super().__init__()

        self._encoder = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout_rate),
            torch.nn.Linear(
                g,
                k,
                bias=False
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(hidden_dropout_rate),
            torch.nn.Linear(
                k,
                k,
                bias=False
            ),
            torch.nn.Tanh(),
        )

        if time_dependent_decay:
            self._intermediate = torch.nn.RNN(
                k,
                k,
                1,
                bias=False
            )

        else:
            self._intermediate = torch.nn.Linear(
                k,
                k,
                bias=False
            )

        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(
                k,
                g
            ),
            torch.nn.ReLU()
        )

        self.time_dependent_decay = time_dependent_decay
        self.g = g
        self.k = k

    def forward(
        self,
        x,
        hidden_state=False,
        return_decay_constants=False
    ):

        # Encode into latent layer
        # and then take the mean over the batch axis (0)
        _x = self._encoder(x)

        if self.time_dependent_decay:

            _x = _x.mean(axis=0)

            _x, self.hidden_state = self._intermediate(
                _x,
                self.hidden_state if hidden_state else None
            )

        else:
            _x = self._intermediate(
                _x.mean(axis=(0, 1))
            )

        # Make the decay rate negative
        _x = torch.mul(
            self._decoder(_x),
            -1.0
        )

        if return_decay_constants:
            return _x
        else:
            return self.rescale_velocity(
                torch.mul(x, _x[None, ...])
            )

    def _slice_data_and_forward(
        self,
        x
    ):

        return self(x[..., 0])

    def output_data(
        self,
        x,
        keep_all_dims=False,
        **kwargs
    ):

        if keep_all_dims:
            return x
        else:
            return x[..., -1]
