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
            torch.nn.ReLU()
        )

    def forward(
        self,
        x,
        hidden_state=None,
        return_decay_constants=False,
        velocity_scale_vector=None,
        expression_scale_vector=None
    ):

        # Encode into latent layer
        # and then take the mean over the batch axis (0)
        _x = self._encoder(x)
        _x = _x.mean(axis=0)

        _x, self.hidden_state = self._intermediate(_x, hidden_state)
        _x = self._decoder(_x)
        _x = torch.matmul(
            _x,
            self._scale_diagonal(
                _x.shape[-1],
                velocity_scale_vector,
                expression_scale_vector
            )
        )

        if return_decay_constants:
            return _x
        else:
            return torch.mul(x, _x[None, ...])

    @staticmethod
    def _scale_diagonal(
        g,
        velocity_scale_vector=None,
        expression_scale_vector=None
    ):
        """
        Rescale here to address differences in X and dX/dt scaling

        :param g: Number of genes
        :type g: int
        :param velocity_scale_vector: Scale vector [G] for dX/dt,
            defaults to None
        :type velocity_scale_vector: torch.tensor, optional
        :param expression_scale_vector: Scale vector [G] for X,
            defaults to None
        :type expression_scale_vector: torch.tensor, optional
        :return: [G x G] diagonal matrix to scale decay component
        :rtype: torch.tensor
        """

        if velocity_scale_vector is None and expression_scale_vector is None:
            return torch.diag(
                torch.full((g, ), -1.0)
            )

        if velocity_scale_vector is not None:
            _scale_vector = torch.clone(velocity_scale_vector)
        else:
            _scale_vector = torch.ones(g)

        if expression_scale_vector is not None:
            _scale_vector /= expression_scale_vector

        _scale_vector *= -1.0

        return torch.diag(_scale_vector)
