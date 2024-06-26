import torch
from ._model_mixins import (
    _TrainingMixin,
    _ScalingMixin
)


class LogisticRegressionTorch(
    torch.nn.Module,
    _TrainingMixin,
    _ScalingMixin
):

    prior_network = None
    input_dropout_rate = 0.0
    bias = False

    def __init__(
        self,
        prior_network=None,
        input_dropout_rate=0.0,
        bias=False
    ):

        super().__init__()

        self.prior_network = prior_network
        self.input_dropout = input_dropout_rate,
        self.bias = bias
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout_rate),
            torch.nn.Linear(*prior_network, bias=bias),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

    def input_data(self, x):
        if isinstance(x, (tuple, list)):
            return x[0]
        else:
            return x

    def output_data(self, x):
        if isinstance(x, (tuple, list)):
            return x[1]
        else:
            return x

    def _slice_data_and_forward(self, x):
        return self.forward(x)
