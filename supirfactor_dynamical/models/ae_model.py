import torch
import pandas as pd
import numpy as np

from ._base_model import _TFMixin
from ._model_mixins import (
    _TrainingMixin,
    _TimeOffsetMixinStatic
)


class Autoencoder(
    torch.nn.Module,
    _TimeOffsetMixinStatic,
    _TFMixin,
    _TrainingMixin
):

    type_name = "autoencoder"

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

        super().__init__()

        self.n_hidden_layers = n_hidden_layers

        if prior_network is None:
            self.g = n_genes
            self.k = hidden_layer_width
            self.prior_network_labels = (
                pd.Index(np.arange(self.g)).astype(str),
                pd.Index(np.arange(self.k)).astype(str)
            )
        else:
            self.process_prior(
                prior_network
            )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

        self.encoder = torch.nn.Sequential()

        for i in range(n_hidden_layers):
            self.encoder.append(
                torch.nn.Linear(
                    self.k if i > 0 else self.g,
                    self.k,
                    bias=False
                )
            )
            self.append_activation_function(
                self.encoder,
                activation
            )

        self.activation = activation

        self.decoder = self.set_decoder(
            activation=output_activation
        )

    def forward(
        self,
        x,
        hidden_state=None,
        n_time_steps=0,
        return_tfa=False
    ):

        return self._forward(
            x,
            hidden_state,
            n_time_steps,
            return_tfa
        )


class TFAutoencoder(
    torch.nn.Module,
    _TimeOffsetMixinStatic,
    _TFMixin,
    _TrainingMixin
):

    type_name = "static"

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

        super().__init__()

        self.set_encoder(
            prior_network,
            use_prior_weights=use_prior_weights,
            activation=activation
        )

        self.decoder = self.set_decoder(
            activation=output_activation
        )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def forward(
        self,
        x,
        hidden_state=None,
        n_time_steps=0,
        return_tfa=False
    ):

        return self._forward(
            x,
            hidden_state,
            n_time_steps,
            return_tfa
        )


class TFMetaAutoencoder(
    torch.nn.Module,
    _TimeOffsetMixinStatic,
    _TFMixin,
    _TrainingMixin
):

    type_name = "static_meta"

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

        super().__init__()

        self.set_encoder(
            prior_network,
            use_prior_weights=use_prior_weights,
            activation=activation
        )

        self._intermediate = self.append_activation_function(
            torch.nn.Sequential(
                torch.nn.Linear(self.k, self.k, bias=False)
            ),
            activation
        )

        self._decoder = self.set_decoder(
            activation=output_activation
        )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    def forward(
        self,
        x,
        hidden_state=None,
        n_time_steps=0,
        return_tfa=False
    ):

        return self._forward(
            x,
            hidden_state,
            n_time_steps,
            return_tfa
        )

    def decoder(
        self,
        x,
        hidden_state=None
    ):

        x = self._intermediate(x)
        x = self._decoder(x)

        return x
