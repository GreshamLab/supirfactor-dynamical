import torch

from ._base_trainer import (
    _TrainingMixin
)


class ChromatinModule(
    torch.nn.Module,
    _TrainingMixin
):

    type_name = 'chromatin'

    hidden_state = None

    g = None
    k = None
    p = None

    def __init__(
        self,
        g,
        p,
        k=50,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0
    ):
        """
        Initialize a chromatin state model

        :param g: Number of genes/transcripts
        :type g: int
        :param p: Number of peaks
        :type p: p
        :param k: Number of internal model nodes
        :type k: int
        :param input_dropout_rate: _description_, defaults to 0.5
        :type input_dropout_rate: float, optional
        :param hidden_dropout_rate: _description_, defaults to 0.0
        :type hidden_dropout_rate: float, optional
        """
        super().__init__()

        self.g = g
        self.k = k
        self.p = p

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.model = torch.nn.Sequential(
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
            torch.nn.Softplus(threshold=5),
            torch.nn.Linear(
                k,
                p,
                bias=False
            ),
            torch.nn.Sigmoid()
        )

    def forward(
        self,
        x
    ):
        return self.model(x)

    def _slice_data_and_forward(self, train_x):
        return self.output_model(
            self(
                self.input_data(train_x)
            )
        )

    def input_data(self, x):
        return x[0]

    def output_data(self, x, **kwargs):
        return x[1]

    def output_model(self, x):
        return x
