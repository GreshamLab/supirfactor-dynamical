import torch
import pandas as pd
import warnings

from torch.nn.utils import prune
from supirfactor_dynamical._utils import _process_weights_to_tensor


class _PriorMixin:

    g = 0
    k = 0

    _drop_tf = None

    prior_network_labels = (None, None)

    @property
    def prior_network_dataframe(self):
        if self.prior_network is None:
            return None

        else:
            return self._to_dataframe(
                self.prior_network,
                transpose=True
            )

    def drop_encoder(
        self,
        x
    ):

        x = self.encoder(x)

        if self._drop_tf is not None:

            _mask = ~self.prior_network_labels[1].isin(self._drop_tf)

            x = x @ torch.diag(
                torch.Tensor(_mask.astype(int))
            ).to(self.device)

        x = self.hidden_dropout(x)

        return x

    def process_prior(
        self,
        prior_network
    ):
        """
        Process a G x K prior into a K x G tensor.
        Sets instance variables for labels and layer size

        :param prior_network: Prior network topology matrix
        :type prior_network: pd.DataFrame, np.ndarray, torch.Tensor
        :return: Prior network topology K x G tensor
        :rtype: torch.Tensor
        """

        # Process prior into a [K x G] tensor
        # and extract labels if provided
        prior_network, prior_network_labels = _process_weights_to_tensor(
            prior_network
        )

        # Set prior instance variables
        self.prior_network_labels = prior_network_labels
        self.k, self.g = prior_network.shape

        return prior_network

    def set_encoder(
        self,
        prior_network,
        use_prior_weights=False,
        activation='softplus'
    ):

        if isinstance(prior_network, tuple):
            self.g, self.k = prior_network
            self.prior_network = prior_network

        else:
            self.prior_network = self.process_prior(
                prior_network
            )

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.g, self.k, bias=False)
        )

        self.append_activation_function(
            self.encoder,
            activation
        )

        self.activation = activation

        if not isinstance(prior_network, tuple):
            # Replace initialized encoder weights with prior weights
            self.mask_input_weights(
                self.prior_network,
                use_mask_weights=use_prior_weights,
                layer_name='weight'
            )

        return self

    def set_decoder(
        self,
        activation='softplus',
        width=None
    ):
        """
        Set decoder

        :param activation: Apply activation function to decoder output
            layer, defaults to 'softplus'
        :type relu: bool, optional
        """

        self.output_activation = activation

        if width is None:
            width = self.k

        decoder = self.append_activation_function(
            torch.nn.Sequential(
                torch.nn.Linear(width, self.g, bias=False),
            ),
            activation
        )

        return decoder

    def mask_input_weights(
        self,
        mask,
        module=None,
        use_mask_weights=False,
        layer_name='weight',
        weight_vstack=None
    ):
        """
        Apply a mask to layer weights

        :param mask: Mask tensor. Non-zero values will be retained,
            and zero values will be masked to zero in the layer weights
        :type mask: torch.Tensor
        :param encoder: Module to mask, use self.encoder if this is None,
            defaults to None
        :type encoder: torch.nn.Module, optional
        :param use_mask_weights: Set the weights equal to values in mask,
            defaults to False
        :type use_mask_weights: bool, optional
        :param layer_name: Module weight name,
            defaults to 'weight'
        :type layer_name: str, optional
        :param weight_vstack: Number of times to stack the mask, for cases
            where the layer weights are also stacked, defaults to None
        :type weight_vstack: _type_, optional
        :raises ValueError: Raise error if the mask and module weights are
            different sizes
        """

        if module is not None:
            encoder = module
        elif isinstance(self.encoder, torch.nn.Sequential):
            encoder = self.encoder[0]
        else:
            encoder = self.encoder

        if weight_vstack is not None and weight_vstack > 1:
            mask = torch.vstack([mask for _ in range(weight_vstack)])

        if mask.shape != getattr(encoder, layer_name).shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match weights {layer_name} "
                f"shape {getattr(encoder, layer_name).shape}"
            )

        # Replace initialized encoder weights with prior weights
        if use_mask_weights:
            setattr(
                encoder,
                layer_name,
                torch.nn.parameter.Parameter(
                    torch.clone(mask)
                )
            )

        # Mask to prior
        prune.custom_from_mask(
            encoder,
            name=layer_name,
            mask=mask != 0
        )

    def set_drop_tfs(
        self,
        drop_tfs,
        raise_error=True
    ):
        """
        Remove specific TF nodes from the TF layer
        by zeroing their activity

        :param drop_tfs: TF name(s) matching TF prior labels.
            None disables TF dropout.
        :type drop_tfs: list, pd.Series
        """

        if drop_tfs is None:
            self._drop_tf = None
            return self

        elif self.prior_network_labels[1] is None:
            raise RuntimeError(
                "Unable to exclude TFs without TF labels; "
                "use a labeled DataFrame for the prior network"
            )

        if not isinstance(
            drop_tfs,
            (tuple, list, pd.Series, pd.Index)
        ):
            drop_tfs = [drop_tfs]

        else:
            drop_tfs = drop_tfs

        _no_match = set(drop_tfs).difference(
            self.prior_network_labels[1]
        )

        if len(_no_match) != 0:

            _msg = f"{len(_no_match)} / {len(drop_tfs)} labels don't match "
            _msg += f"model labels: {list(_no_match)}"

            if raise_error:
                raise RuntimeError(_msg)

            else:
                warnings.warn(_msg, RuntimeWarning)

        self._drop_tf = drop_tfs

        return self

    @torch.inference_mode()
    def _to_dataframe(
        self,
        x,
        transpose=False
    ):
        """
        Turn a tensor or numpy array into a dataframe
        labeled using the genes & regulators from the prior

        :param x: [G x K] data, or [K x G] data if transpose=True
        :type x: torch.tensor, np.ndarray
        :param transpose: Transpose [K x G] data to [G x K],
            defaults to False
        :type transpose: bool, optional
        :return: [G x K] DataFrame
        :rtype: pd.DataFrame
        """

        try:
            with torch.no_grad():
                x = x.numpy()

        except AttributeError:
            pass

        return pd.DataFrame(
            x.T if transpose else x,
            index=self.prior_network_labels[0],
            columns=self.prior_network_labels[1]
        )

    @staticmethod
    def append_activation_function(
        module,
        activation,
        **kwargs
    ):

        # Build the encoder module
        if activation is None:
            pass
        else:
            module.append(
                _PriorMixin.get_activation_function(
                    activation,
                    **kwargs
                )
            )

        return module

    @staticmethod
    def get_activation_function(
        activation,
        **kwargs
    ):
        if activation is None:
            return None
        elif activation.lower() == 'sigmoid':
            return torch.nn.Sigmoid(**kwargs)
        elif activation.lower() == 'softplus':
            return torch.nn.Softplus(**kwargs)
        elif activation.lower() == 'relu':
            return torch.nn.ReLU(**kwargs)
        elif activation.lower() == 'tanh':
            return torch.nn.Tanh(**kwargs)
        else:
            raise ValueError(
                f"Activation {activation} unknown"
            )
