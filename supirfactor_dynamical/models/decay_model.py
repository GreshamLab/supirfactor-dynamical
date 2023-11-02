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

    n_genes = None
    hidden_layer_width = None

    def __init__(
        self,
        n_genes,
        hidden_layer_width=20,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0
    ):
        super().__init__()

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self._encoder = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout_rate),
            torch.nn.Linear(
                n_genes,
                hidden_layer_width,
                bias=False
            ),
            torch.nn.Tanh(),
            torch.nn.Dropout(hidden_dropout_rate),
            torch.nn.Linear(
                hidden_layer_width,
                hidden_layer_width,
                bias=False
            ),
            torch.nn.Softplus(threshold=5)
        )

        self._intermediate = torch.nn.RNN(
            hidden_layer_width,
            hidden_layer_width,
            1,
            bias=False,
            batch_first=True
        )

        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_layer_width,
                n_genes,
                bias=False
            ),
            torch.nn.Softplus(threshold=5)
        )

        self.n_genes = n_genes
        self.hidden_layer_width = hidden_layer_width

    def forward(
        self,
        x,
        hidden_state=False,
        return_decay_constants=False
    ):

        # Encode into latent layer
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
        n_genes=None,
        initial_values=None,
        hidden_layer_width=20,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0
    ):
        torch.nn.Module.__init__(self)

        if initial_values is None:
            _decay_rates = torch.rand(n_genes)
            _decay_rates /= 1 / torch.sqrt(torch.Tensor(n_genes))

            self._decay_rates = torch.nn.Parameter(
                _decay_rates
            )

        else:
            self._decay_rates = torch.nn.Parameter(
                self.to_tensor(initial_values)
            )

        self._activation = torch.nn.Softplus(threshold=5)

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

        if x.ndim > 1:
            _decay_rates = torch.unsqueeze(
                self._decay_rates,
                0
            )
        else:
            _decay_rates = self._decay_rates

        if x.ndim == 3:
            _decay_rates = torch.unsqueeze(
                _decay_rates,
                0
            )

        _decay_rates = self._activation(_decay_rates)
        _decay_rates = torch.mul(_decay_rates, -1)

        # Multiply decay rate by input counts to get
        # decay velocity
        _v = self.rescale_velocity(
            torch.mul(
                x,
                _decay_rates
            )
        )

        if return_decay_constants:
            return _v, _decay_rates
        else:
            return _v
