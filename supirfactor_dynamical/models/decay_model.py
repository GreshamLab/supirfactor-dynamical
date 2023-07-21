import torch

from ._base_trainer import _TrainingMixin


class DecayModule(
    torch.nn.Module,
    _TrainingMixin
):

    hidden_state = None

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
            )
        )

        self.time_dependent_decay = time_dependent_decay

    def forward(
        self,
        x,
        hidden_state=None,
        return_decay_constants=False,
        base_decay=None
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

        if self.training:
            _x = torch.nn.LeakyReLU(1e-4)(_x)
        else:
            _x = torch.nn.ReLU()(_x)

        if base_decay is not None:
            _x = torch.add(_x, base_decay)

        _x = torch.mul(_x, -1.0)

        if return_decay_constants:
            return _x
        else:
            return torch.mul(x, _x[None, ...])

    def train_model(
        self,
        data,
        epochs,
        optimizer=None
    ):
        """
        Training stub for decay module.
        In general this should be trained as a module of a larger
        model.
        """

        optimizer = self.process_optimizer(optimizer)
        loss_function = torch.nn.MSELoss()

        for _ in range(epochs):

            self.train()

            mse = loss_function(
                self._slice_data_and_forward(data),
                self.output_data(data)
            )

            mse.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.training_loss.append(mse.item())

    def _slice_data_and_forward(self, x):

        return self(x[..., 0])

    def output_data(self, x, **kwargs):

        return torch.multiply(
            torch.multiply(x[..., 1], x[..., 0]),
            -1.0
        )
