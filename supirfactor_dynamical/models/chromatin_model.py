import torch
import numpy as np
import warnings

from ._model_mixins import (
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

    train_chromatin_model = True
    chromatin_model_threshold = 0.5

    g = None
    k = None
    p = None

    def __init__(
        self,
        gene_peak_mask=None,
        peak_tf_prior_network=None,
        chromatin_model=None,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        train_chromatin_model=True,
        chromatin_model_threshold=0.5
    ):

        super().__init__()

        self.peak_tf_prior_network = peak_tf_prior_network
        self.gene_peak_mask = gene_peak_mask

        self.g, self.p = gene_peak_mask.shape
        _, self.k = peak_tf_prior_network.shape

        self.train_chromatin_model = train_chromatin_model
        self.chromatin_model_threshold = chromatin_model_threshold

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.peak_encoder = torch.nn.Sequential(
            self.input_dropout,
            torch.nn.Linear(self.g, self.p, bias=False),
            torch.nn.Softplus(threshold=5)
        )

        gene_peak_mask, gene_peak_labels = _process_weights_to_tensor(
            gene_peak_mask
        )

        self.mask_input_weights(
            gene_peak_mask,
            module=self.peak_encoder[1],
            layer_name='weight'
        )

        self.tf_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.p, self.k, bias=False),
            torch.nn.Softplus(threshold=5)
        )

        peak_tf_prior_network, peak_tf_labels = _process_weights_to_tensor(
            peak_tf_prior_network
        )

        self.prior_network_labels = (
            gene_peak_labels[0],
            peak_tf_labels[1]
        )

        if not np.all(gene_peak_labels[1] == peak_tf_labels[0]):
            warnings.warn(
                "Peak labels do not match"
            )

        self.mask_input_weights(
            peak_tf_prior_network,
            module=self.tf_encoder[0],
            layer_name='weight'
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.k, self.k, bias=False),
            torch.nn.Softplus(threshold=5),
            torch.nn.Linear(self.k, self.g, bias=False),
            torch.nn.Softplus(threshold=5)
        )

        if chromatin_model is not None:

            if isinstance(chromatin_model, str):
                from .._io._loader import read
                chromatin_model = read(chromatin_model)

            self.chromatin_model = chromatin_model

        else:

            if not train_chromatin_model:
                raise RuntimeError(
                    "`train_chromatin_model` cannot be False "
                    "unless a pre-trained chromatin model is provided"
                )

            self.chromatin_model = ChromatinModule(
                self.g,
                self.p,
                self.k,
                input_dropout_rate=input_dropout_rate,
                hidden_dropout_rate=hidden_dropout_rate,
            )

    def chromatin_encoder(self, x):

        peak_status = self.chromatin_model(x)

        if self.chromatin_model_threshold is not None:
            peak_status = peak_status > self.chromatin_model_threshold
            peak_status = peak_status.float()

        return peak_status

    def encoder(self, x):

        peak_activity = torch.mul(
            self.peak_encoder(x),
            self.chromatin_encoder(x)
        )

        return self.tf_encoder(peak_activity)

    def forward(
        self,
        x,
        return_tfa=False,
        n_time_steps=None
    ):

        tfa = self.drop_encoder(x)
        x_out = self.decoder(tfa)

        if return_tfa:
            return x_out, tfa
        else:
            return x_out

    def train_model(
        self,
        training_dataloader,
        epochs,
        validation_dataloader=None,
        loss_function=torch.nn.MSELoss(),
        optimizer=None
    ):

        # Optimize entire module
        if self.train_chromatin_model:
            optimizer = self.process_optimizer(optimizer)

        # Optimize all the modules except the chromatin module
        else:
            optimizer = self.process_optimizer(
                optimizer,
                params=[
                    x for x in self.peak_encoder.parameters()
                ] + [
                    x for x in self.tf_encoder.parameters()
                ] + [
                    x for x in self.decoder.parameters()
                ]
            )

        return super().train_model(
            training_dataloader,
            epochs,
            validation_dataloader,
            loss_function,
            optimizer
        )

    @property
    def decoder_weights(self):
        return self.decoder[2].weight.to('cpu')


class ChromatinModule(
    torch.nn.Module,
    _TrainingMixin
):

    type_name = 'chromatin'

    hidden_state = None

    n_genes = None
    hidden_layer_width = None
    n_peaks = None

    def __init__(
        self,
        n_genes=None,
        n_peaks=None,
        hidden_layer_width=50,
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

        self.n_genes = n_genes
        self.hidden_layer_width = hidden_layer_width
        self.n_peaks = n_peaks

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.model = torch.nn.Sequential(
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
            torch.nn.Softplus(threshold=5),
            torch.nn.Linear(
                hidden_layer_width,
                n_peaks,
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
