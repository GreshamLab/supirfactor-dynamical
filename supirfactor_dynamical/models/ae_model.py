import torch

from ._base_model import _TFMixin
from ._model_mixins import (
    _TrainingMixin,
    _TimeOffsetMixinStatic
)


class TFMultilayerAutoencoder(
    torch.nn.Module,
    _TimeOffsetMixinStatic,
    _TFMixin,
    _TrainingMixin
):

    type_name = "static_multilayer"

    _serialize_args = [
        'prior_network',
        'input_dropout_rate',
        'hidden_dropout_rate',
        'intermediate_dropout_rate',
        'intermediate_sizes',
        'decoder_sizes',
        'tfa_activation',
        'activation',
        'output_activation',
        'output_nodes'
    ]

    intermediate_sizes = None
    decoder_sizes = None

    intermediate_dropout_rate = 0.2

    tfa_activation = 'relu'

    @property
    def intermediate_weights(self):
        if self.intermediate_sizes is None:
            return None

        return [
            self._intermediate[3 * i].weight
            for i in range(len(self.intermediate_sizes))
        ]

    @property
    def decoder_weights(self):
        return self._decoder[-2].weight

    def __init__(
        self,
        prior_network=None,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        intermediate_dropout_rate=0.0,
        intermediate_sizes=(100, ),
        decoder_sizes=(100, ),
        tfa_activation='relu',
        activation='relu',
        output_activation='relu',
        output_nodes=None
    ):
        super().__init__()

        self.set_encoder(
            prior_network,
            use_prior_weights=use_prior_weights,
            activation=tfa_activation
        )

        self.intermediate_sizes = intermediate_sizes
        self.decoder_sizes = decoder_sizes

        self.tfa_activation = tfa_activation
        self.activation = activation
        self.output_activation = output_activation
        self.output_nodes = output_nodes

        intermediates = [self.k]

        if intermediate_sizes is not None:
            intermediates = intermediates + list(intermediate_sizes)

            self._intermediate = self.create_submodule(
                intermediates,
                activation=activation,
                dropout_rate=intermediate_dropout_rate
            )
        else:
            self._intermediate = torch.nn.Sequential()

        decoders = [intermediates[-1]]

        if decoder_sizes is not None:
            decoders = decoders + list(decoder_sizes)

        if output_nodes is None:
            output_nodes = self.g

        self._decoder = self.set_decoder(
            output_activation,
            intermediate_activation=activation,
            decoder_sizes=decoders,
            dropout_rate=intermediate_dropout_rate,
            output_nodes=output_nodes
        )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate,
            intermediate_dropout_rate
        )

    def decoder(
        self,
        x,
        hidden_state=None
    ):

        x = self._intermediate(x)
        x = self._decoder(x)

        return x

    def latent_embedding(
        self,
        x
    ):

        x = self.drop_encoder(x)
        return self._intermediate(x)

    def forward(
        self,
        x,
        n_time_steps=0,
        return_tfa=False
    ):

        return self._forward(
            x,
            n_time_steps=n_time_steps,
            return_tfa=return_tfa
        )


class Autoencoder(TFMultilayerAutoencoder):

    type_name = "autoencoder"

    _serialize_args = [
        'prior_network',
        'n_genes',
        'hidden_layer_width',
        'n_hidden_layers',
        'input_dropout_rate',
        'hidden_dropout_rate',
        'activation',
        'output_activation'
    ]

    def __init__(
        self,
        prior_network=None,
        n_genes=None,
        hidden_layer_width=50,
        n_hidden_layers=1,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        activation='relu',
        output_activation='relu'
    ):
        """
        Create a black-box Autoencoder module

        :param input_dropout_rate: Training dropout for input genes,
            defaults to 0.5
        :type input_dropout_rate: float, optional
        :param activation: Apply activation function to hidden
            layer, defaults to ReLU
        :type activation: bool, optional
        :param output_activation: Apply activation function to output
            layer, defaults to ReLU
        :type output_activation: bool, optional
        """

        self.n_hidden_layers = n_hidden_layers
        self.n_genes = n_genes

        if prior_network is None:
            self.g = n_genes
            self.k = hidden_layer_width
        else:
            self.process_prior(
                prior_network
            )

        if n_hidden_layers > 1:
            intermediate_sizes = (hidden_layer_width, ) * (n_hidden_layers - 1)
        else:
            intermediate_sizes = None

        super().__init__(
            prior_network=(self.g, self.k),
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            intermediate_sizes=intermediate_sizes,
            decoder_sizes=None,
            activation=activation,
            tfa_activation=activation,
            output_activation=output_activation
        )


class TFMetaAutoencoder(TFMultilayerAutoencoder):

    type_name = "static_meta"

    _serialize_args = [
        'prior_network',
        'input_dropout_rate',
        'hidden_dropout_rate',
        'activation',
        'output_activation'
    ]

    @property
    def intermediate_weights(self):
        return self._intermediate[0].weight

    @property
    def decoder_weights(self):
        return self._decoder[0].weight

    def __init__(
        self,
        prior_network=None,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        activation='relu',
        output_activation='relu'
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
        :param input_dropout_rate: Training dropout for input genes,
            defaults to 0.5
        :type input_dropout_rate: float, optional
        :param activation: Apply activation function to hidden
            layer, defaults to ReLU
        :type activation: bool, optional
        :param output_activation: Apply activation function to output
            layer, defaults to ReLU
        :type output_activation: bool, optional
        """

        self.process_prior(
            prior_network
        )

        super().__init__(
            prior_network=prior_network,
            use_prior_weights=use_prior_weights,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            intermediate_sizes=(self.k, ),
            decoder_sizes=None,
            activation=activation,
            tfa_activation=activation,
            output_activation=output_activation
        )


class TFAutoencoder(TFMultilayerAutoencoder):

    type_name = "static"

    _serialize_args = [
        'prior_network',
        'input_dropout_rate',
        'hidden_dropout_rate',
        'activation',
        'output_activation'
    ]

    def __init__(
        self,
        prior_network=None,
        use_prior_weights=False,
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        activation='relu',
        output_activation='relu'
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
        :param input_dropout_rate: Training dropout for input genes,
            defaults to 0.5
        :type input_dropout_rate: float, optional
        :param activation: Apply activation function to hidden
            layer, defaults to ReLU
        :type activation: bool, optional
        :param output_activation: Apply activation function to output
            layer, defaults to ReLU
        :type output_activation: bool, optional
        """

        super().__init__(
            prior_network=prior_network,
            use_prior_weights=use_prior_weights,
            input_dropout_rate=input_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            intermediate_sizes=None,
            decoder_sizes=None,
            activation=activation,
            tfa_activation=activation,
            output_activation=output_activation
        )
