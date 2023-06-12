import torch

from ._base_model import _TFMixin


class TFAutoencoder(torch.nn.Module, _TFMixin):

    type_name = "static"

    def __init__(
        self,
        prior_network,
        use_prior_weights=False,
        decoder_weights=None,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        output_relu=True
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
        :param output_relu: Apply activation function (ReLU) to output
            layer, constrains to positive, defaults to True
        :type output_relu: bool, optional
        """

        super().__init__()

        self.set_encoder(
            prior_network,
            use_prior_weights=use_prior_weights
        )

        self.decoder = self.set_decoder(
            relu=output_relu,
            decoder_weights=decoder_weights
        )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def forward(
        self,
        x,
        hidden_state=None,
        n_time_steps=0
    ):

        return self._forward(
            x,
            hidden_state,
            n_time_steps
        )

    def _forward_step(
        self,
        x,
        hidden_state=None
    ):

        x = self.drop_encoder(x)
        x = self.hidden_dropout(x)
        x = self.decoder(x)

        return x


class TFMetaAutoencoder(torch.nn.Module, _TFMixin):

    type_name = "static_meta"

    def __init__(
        self,
        prior_network,
        use_prior_weights=False,
        decoder_weights=None,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        output_relu=True
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
        :param output_relu: Apply activation function (ReLU) to output
            layer, constrains to positive, defaults to True
        :type output_relu: bool, optional
        """

        super().__init__()

        self.set_encoder(
            prior_network,
            use_prior_weights=use_prior_weights
        )

        self._intermediate = torch.nn.Sequential(
            torch.nn.Linear(self.k, self.k, bias=False),
            torch.nn.ReLU()
        )

        self.decoder = self.set_decoder(
            relu=output_relu,
            decoder_weights=decoder_weights
        )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def forward(
        self,
        x,
        hidden_state=None,
        n_time_steps=0
    ):

        return self._forward(
            x,
            hidden_state,
            n_time_steps
        )

    def _forward_step(
        self,
        x,
        hidden_state=None
    ):

        x = self.drop_encoder(x)
        x = self.hidden_dropout(x)
        x = self._intermediate(x)
        x = self.hidden_dropout(x)
        x = self.decoder(x)

        return x
