import torch


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
            torch.nn.Sigmoid()
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
            torch.nn.Dropout(hidden_dropout_rate),
            torch.nn.Linear(
                k,
                g
            ),
            torch.nn.LeakyReLU(1e-3)
        )

        self.time_dependent_decay = time_dependent_decay

    def forward(
        self,
        x,
        hidden_state=None,
        return_decay_constants=False
    ):

        # Encode into latent layer
        # and then take the mean over the batch axis (0)
        _x = self._encoder(x)

        if self.time_dependent_decay:
            _x = _x.mean(axis=0)
            _x, self.hidden_state = self._intermediate(_x, hidden_state)
        else:
            _x = _x.mean(axis=(0, 1))
            _x = self._intermediate(_x)

        _x = self._decoder(_x)

        if not self.training:
            _x = torch.nn.ReLU()(_x)

        _x = torch.mul(_x, -1.0)

        if return_decay_constants:
            return _x
        else:
            return torch.mul(x, _x[None, ...])
