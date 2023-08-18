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

    g = None
    k = None

    def __init__(
        self,
        g,
        k=20,
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
            torch.nn.Tanh(),
            torch.nn.Dropout(hidden_dropout_rate),
            torch.nn.Linear(
                k,
                k,
                bias=False
            ),
            torch.nn.Tanh()
        )

        self._intermediate = torch.nn.RNN(
            k,
            k,
            1,
            bias=False,
            batch_first=True
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(
                k,
                g
            ),
            torch.nn.Softplus()
        )

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

        _x, self.hidden_state = self._intermediate(
            _x,
            self.hidden_state if hidden_state else None
        )

        # Make the decay rate negative
        _x = torch.mul(
            self._decoder(_x),
            -1.0
        )

        # Multiply decay rate by input counts to get
        # decay velocity
        _v = self.rescale_velocity(
            torch.mul(x, _x)
        )

        if return_decay_constants:
            return _v, _x
        else:
            return _v

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


class DecayModuleSimple(DecayModule):

    @property
    def decay_rates(self):
        return self._activation(
            self._decay_rates
        ) * -1

    def __init__(
        self,
        g,
        k=20,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0
    ):
        torch.nn.Module.__init__(self)

        self._decay_rates = torch.nn.Parameter(
            self.to_tensor(g)
        )

        self._activation = torch.nn.Softplus(threshold=4)

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def forward(
        self,
        x,
        hidden_state=False,
        return_decay_constants=False
    ):

        _decay_rates = self.decay_rates

        # Multiply decay rate by input counts to get
        # decay velocity
        _v = self.rescale_velocity(
            torch.mul(
                x,
                _decay_rates[None, ...]
            )
        )

        if return_decay_constants:
            return _v, _decay_rates
        else:
            return _v
