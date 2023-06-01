import torch

from ._base_model import _TF_RNN_mixin


class TFLSTMAutoencoder(torch.nn.Module, _TF_RNN_mixin):

    type_name = "lstm"

    def __init__(
        self,
        prior_network,
        use_prior_weights=False,
        decoder_weights=None,
        initial_state=None,
        recurrency_mask=None,
        input_dropout_rate=0.5,
        layer_dropout_rate=0.0,
        output_relu=True,
        prediction_offset=None
    ):
        """
        Create a recurrent TF autoencoder

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
        :param initial_state: Initial time-zero TF hidden layer values,
            defaults to all-zero initialization.
        :type initial_state: np.ndarray, torch.Tensor, optional
        :param recurrency_mask: Connectivity matrix connecting hidden layer to
            hidden layer, defaults to diagonal matrix. Pass False to disable
            masking (fully connected hidden layer).
        :type recurrency_mask: torch.Tensor, optional
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

        # Build LSTM model
        # to encode single hidden (TF) layer
        self.encoder = torch.nn.LSTM(
            self.g,
            self.k,
            1,
            bias=False,
            dropout=layer_dropout_rate,
            batch_first=True
        )

        # Build linear decode layer
        # to connect hidden (TF) layer to
        # output gene expression
        self.decoder = self.set_decoder(
            relu=output_relu,
            decoder_weights=decoder_weights
        )

        self.input_dropout = torch.nn.Dropout(
            p=input_dropout_rate
        )

        self.mask_input_weights(
            prior_network,
            weight_vstack=4,
            use_mask_weights=use_prior_weights
        )

        self.mask_recurrent_weights(
            recurrency_mask,
            n_to_stack=4
        )

        self.hidden_activation = torch.nn.ReLU()
        self.initial_state = initial_state
        self.prediction_offset = prediction_offset

    def forward(self, x, hidden_state=None):

        x = self.input_dropout(x)
        x, self.hidden_forward = self.encoder(x, hidden_state)

        if self.hidden_activation is not None:
            x = self.hidden_activation(x)

        x = self.decoder(x)

        return x


class TFLSTMDecoder(torch.nn.Module, _TF_RNN_mixin):

    type_name = "lstm_decoder"

    _serialize_args = [
        'input_dropout_rate',
        'layer_dropout_rate',
        'output_relu',
        'prediction_offset',
        'initial_state'
    ]

    input_dropout_rate = 0.5
    layer_dropout_rate = 0.0
    output_relu = True
    prediction_offset = None
    initial_state = None

    @property
    def encoder_weights(self):
        return self.encoder[0].weight

    @property
    def recurrent_weights(self):
        return self._intermediate.weight_hh_l0

    @property
    def decoder_weights(self):
        return self._decoder[0].weight

    def __init__(
        self,
        prior_network,
        use_prior_weights=False,
        initial_state=None,
        input_dropout_rate=0.5,
        layer_dropout_rate=0.0,
        output_relu=True,
        prediction_offset=None,
        decoder_weights=None,
        recurrency_mask=False
    ):
        """
        Create a recurrent TF autoencoder

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
        :param initial_state: Initial time-zero TF hidden layer values,
            defaults to all-zero initialization.
        :type initial_state: np.ndarray, torch.Tensor, optional
        :param recurrency_mask: Connectivity matrix connecting hidden layer to
            hidden layer, defaults to diagonal matrix.
        :type recurrency_mask: torch.Tensor, optional
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

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.g, self.k, bias=False),
            torch.nn.ReLU()
        )

        # Build standard ReLU RNN
        # to encode connection between TF layer
        # and output layer

        self._intermediate = torch.nn.LSTM(
            self.k,
            self.k,
            1,
            bias=False,
            dropout=layer_dropout_rate,
            batch_first=True
        )

        self._decoder = self.set_decoder(
            relu=output_relu,
            decoder_weights=decoder_weights
        )

        self.input_dropout = torch.nn.Dropout(
            p=input_dropout_rate
        )

        self.mask_input_weights(
            prior_network,
            use_mask_weights=use_prior_weights,
            layer_name='weight'
        )

        self.initial_state = initial_state
        self.prediction_offset = prediction_offset
        self.input_dropout_rate = input_dropout_rate
        self.layer_dropout_rate = layer_dropout_rate
        self.output_relu = output_relu

    def forward(self, x, hidden_state=None):

        x = self.input_dropout(x)
        x = self.encoder(x)
        x = self.decoder(x, hidden_state)

        return x

    def decoder(self, x, hidden_state=None):

        x, self.hidden_final = self._intermediate(x, hidden_state)
        x = self._decoder(x)

        return x

    @torch.inference_mode()
    def _latent_layer_values(self, x):
        return self.encoder(x).detach()
