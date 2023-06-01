import torch

from ._base_model import _TFMixin


class TFAutoencoder(torch.nn.Module, _TFMixin):

    type_name = "static"

    _serialize_args = [
        'input_dropout_rate',
        'layer_dropout_rate',
        'output_relu',
        'prediction_offset'
    ]

    input_dropout_rate = 0.5
    layer_dropout_rate = 0.0
    output_relu = True

    def __init__(
        self,
        prior_network,
        use_prior_weights=False,
        decoder_weights=None,
        input_dropout_rate=0.5,
        layer_dropout_rate=0.0,
        output_relu=True,
        prediction_offset=None
    ):
        """
        Create a TF Autoencoder module

        :param prior_network: 2D mask to connect genes to the TF hidden layer,
            where genes are on 0 (index) and TFs are on 1 (columns).
            Nonzero values are connections.
            Must match training data gene order.
        :type prior_network: pd.DataFrame [G x K], torch.Tensor [G x K]
        :param use_prior_weights: Use values in the prior_network as the
            initalization for encoder weights, defaults to False
        :type use_prior_weights: bool, optional
        :param decoder_weights: Values to use as the initialization for
            decoder weights. Any values that are zero will be pruned to enforce
            the same sparsity structure after training. Defaults to None
        :type decoder_weights: pd.DataFrame [G x K], np.ndarray, optional
        :param input_dropout_rate: Training dropout for input genes,
            defaults to 0.5
        :type input_dropout_rate: float, optional
        :param layer_dropout_rate: Training dropout for hidden layer TFs,
            defaults to 0.0
        :type layer_dropout_rate: float, optional
        :param output_relu: Apply activation function (ReLU) to output
            layer, constrains to positive, defaults to True
        :type output_relu: bool, optional
        """

        super().__init__()
        prior_network = self.process_prior(prior_network)

        # Build the encoder module
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.g, self.k, bias=False),
            torch.nn.ReLU()
        )

        # Replace initialized encoder weights with prior weights
        self.mask_input_weights(
            prior_network,
            use_mask_weights=use_prior_weights,
            layer_name='weight'
        )

        self.decoder = self.set_decoder(
            relu=output_relu,
            decoder_weights=decoder_weights
        )

        self.input_dropout = torch.nn.Dropout(
            p=input_dropout_rate
        )

        self.layer_dropout = torch.nn.Dropout(
            p=layer_dropout_rate
        )

        self.input_dropout_rate = input_dropout_rate
        self.layer_dropout_rate = layer_dropout_rate
        self.output_relu = output_relu
        self.prediction_offset = prediction_offset

    def forward(self, x, hidden_state=None):

        x = self.input_dropout(x)
        x = self.encoder(x)
        x = self.layer_dropout(x)
        x = self.decoder(x)

        return x
