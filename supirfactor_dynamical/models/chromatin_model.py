import torch

from ._base_trainer import (
    _TrainingMixin,
    _TimeOffsetMixinStatic
)

from ._base_model import (
    _TFMixin
)

from .._utils import _process_weights_to_tensor


class ChromatinAwareModel(
    torch.nn.Module,
    _TimeOffsetMixinStatic,
    _TFMixin,
    _TrainingMixin
):

    type_name = 'chromatin_aware'

    g = None
    k = None
    p = None

    def __init__(
        self,
        gene_peak_mask,
        peak_tf_mask,
        chromatin_model=None,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0
    ):

        super().__init__()

        self.g, self.p = gene_peak_mask.shape
        _, self.k = peak_tf_mask.shape

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.peak_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.g, self.p, bias=False),
            torch.nn.Softplus(threshold=5)
        )

        gene_peak_mask, _ = _process_weights_to_tensor(
            gene_peak_mask
        )

        self.mask_input_weights(
            gene_peak_mask,
            module=self.peak_encoder[0],
            layer_name='weight'
        )

        peak_tf_mask, _ = _process_weights_to_tensor(
            peak_tf_mask
        )

        self.tf_encoder = torch.nn.Sequential(
            self.input_dropout,
            torch.nn.Linear(self.p, self.k, bias=False),
            torch.nn.Softplus(threshold=5)
        )

        self.mask_input_weights(
            peak_tf_mask,
            module=self.tf_encoder[1],
            layer_name='weight'
        )

        self.decoder = torch.nn.Sequential(
            self.hidden_dropout,
            torch.nn.Linear(self.k, self.k, bias=False),
            torch.nn.Softplus(threshold=5),
            torch.nn.Linear(self.k, self.g, bias=False),
            torch.nn.Softplus(threshold=5)
        )

        if chromatin_model is not None:

            if isinstance(chromatin_model, str):
                from .._utils._loader import read
                chromatin_model = read(chromatin_model)

            self.chromatin_model = chromatin_model

        else:

            self.chromatin_model = ChromatinModule(
                self.g,
                self.p,
                self.k,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
            )

    def forward(self, x, return_tfa=False, n_time_steps=None):

        peak_state = self.chromatin_model(x)

        peak_activity = self.peak_encoder(x)
        peak_activity = torch.mul(
            peak_activity,
            peak_state
        )

        tfa = self.tf_encoder(peak_activity)
        x_out = self.decoder(tfa)

        if return_tfa:
            return x_out, tfa
        else:
            return x_out


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
