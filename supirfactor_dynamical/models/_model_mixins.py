import torch
import pandas as pd
import warnings

from torch.nn.utils import prune
from .._utils import _process_weights_to_tensor


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
            )

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

        # Replace initialized encoder weights with prior weights
        self.mask_input_weights(
            self.prior_network,
            use_mask_weights=use_prior_weights,
            layer_name='weight'
        )

        return self

    def set_decoder(
        self,
        activation='softplus'
    ):
        """
        Set decoder

        :param activation: Apply activation function to decoder output
            layer, defaults to 'softplus'
        :type relu: bool, optional
        """

        self.output_activation = activation

        decoder = self.append_activation_function(
            torch.nn.Sequential(
                torch.nn.Linear(self.k, self.g, bias=False),
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


class _ScalingMixin:

    _count_inverse_scaler = None
    _velocity_inverse_scaler = None

    count_to_velocity_scaler = None
    velocity_to_count_scaler = None

    @property
    def count_scaler(self):
        if self._count_inverse_scaler is not None:
            return torch.diag(self._count_inverse_scaler)
        else:
            return None

    @property
    def velocity_scaler(self):
        if self._velocity_inverse_scaler is not None:
            return torch.diag(self._velocity_inverse_scaler)
        else:
            return None

    def set_scaling(
        self,
        count_scaling=False,
        velocity_scaling=False
    ):
        """
        If count or velocity is scaled, fix the scaling so that
        they match.

        Needed to calculate t+1 from count and velocity at t

        x_(t+1) = x(t) + dx/dx * (count_scaling / velocity_scaling)

        :param count_scaling: Count scaler [G] vector,
            None disables count scaling.
        :type count_scaling: torch.Tensor, np.ndarray, optional
        :param velocity_scaling: Velocity scaler [G] vector,
            None disables velocity scaling
        :type velocity_scaling: torch.Tensor, np.ndarray, optional
        """

        if count_scaling is None:
            self._count_inverse_scaler = None

        elif count_scaling is not False:
            self._count_inverse_scaler = self.to_tensor(count_scaling)

        if velocity_scaling is None:
            self._velocity_inverse_scaler = None

        elif velocity_scaling is not False:
            self._velocity_inverse_scaler = self.to_tensor(velocity_scaling)

        self.count_to_velocity_scaler = self._zero_safe_div(
            self._count_inverse_scaler,
            self._velocity_inverse_scaler
        )

        self.velocity_to_count_scaler = self._zero_safe_div(
            self._velocity_inverse_scaler,
            self._count_inverse_scaler
        )

        return self

    def unscale_counts(self, x):
        if self._count_inverse_scaler is not None:
            return torch.matmul(x, self.count_scaler)
        else:
            return x

    def unscale_velocity(self, x):
        if self._velocity_inverse_scaler is not None:
            return torch.matmul(x, self.velocity_scaler)
        else:
            return x

    def rescale_velocity(self, x):
        if self._velocity_inverse_scaler is not None:
            return torch.matmul(
                x,
                self._zero_safe_div(
                    None,
                    self._velocity_inverse_scaler
                )
            )
        else:
            return x

    def rescale_counts(self, x):
        if self._count_inverse_scaler is not None:
            return torch.matmul(
                x,
                self._zero_safe_div(
                    None,
                    self._count_inverse_scaler
                )
            )
        else:
            return x

    def scale_count_to_velocity(
        self,
        count
    ):
        if self.count_to_velocity_scaler is not None:
            return torch.matmul(count, self.count_to_velocity_scaler)
        else:
            return count

    def scale_velocity_to_count(
        self,
        velocity
    ):
        if self.velocity_to_count_scaler is not None:
            return torch.matmul(velocity, self.velocity_to_count_scaler)
        else:
            return velocity

    @staticmethod
    def _zero_safe_div(x, y):
        """
        Return z = x / y
        z = 1 for y == 0
        Allow for Nones
        """

        if x is None and y is None:
            return None

        elif x is None:
            _z = 1 / y
            _z[y == 0] = 1

        elif y is None:
            _z = x

        else:
            _z = torch.div(x, y)
            _z[y == 0] = 1

        return torch.diag(_z)
