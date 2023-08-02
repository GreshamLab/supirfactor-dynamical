import torch
import pandas as pd
import warnings

from .._utils import _process_weights_to_tensor


class _PriorMixin:

    g = 0
    k = 0

    _drop_tf = None

    prior_network = None
    prior_network_labels = (None, None)

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
        self.prior_network = prior_network
        self.prior_network_labels = prior_network_labels
        self.k, self.g = prior_network.shape

        return prior_network

    def set_encoder(
        self,
        prior_network,
        use_prior_weights=False,
        sigmoid=False
    ):

        prior_network = self.process_prior(
            prior_network
        )

        # Build the encoder module
        if sigmoid:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(self.g, self.k, bias=False),
                torch.nn.Sigmoid()
            )
        else:
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

    def set_drop_tfs(
        self,
        drop_tfs
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
            return

        elif self.prior_network_labels[1] is None:
            raise RuntimeError(
                "Unable to exclude TFs without TF labels; "
                "use a labeled DataFrame for the prior network"
            )

        elif not isinstance(
            drop_tfs,
            (tuple, list, pd.Series, pd.Index)
        ):
            self._drop_tf = [drop_tfs]

        else:
            self._drop_tf = drop_tfs

        _no_match = set(self._drop_tf).difference(
            self.prior_network_labels[1]
        )

        if len(_no_match) != 0:
            warnings.warn(
                f"{len(_no_match)} / {len(self._drop_tf)} labels don't match "
                f"model labels: {list(_no_match)}",
                RuntimeWarning
            )

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

        scalers = self.make_scalers(
            self._count_inverse_scaler,
            self._velocity_inverse_scaler
        )

        self.count_to_velocity_scaler = scalers[0]
        self.velocity_to_count_scaler = scalers[1]

        return self

    def unscale_counts(self, x):
        if self.count_scaler is not None:
            return torch.matmul(x, self.count_scaler)
        else:
            return x

    def unscale_velocity(self, x):
        if self.velocity_scaler is not None:
            return torch.matmul(x, self.velocity_scaler)
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
    def make_scalers(
        count_vec,
        velo_vec=None
    ):
        """
        Make scaling matrices to scale & unscale
        count and velocity matrix
        """

        if count_vec is None and velo_vec is None:
            return None, None

        # Build scaler matrix to go from count to velocity
        # and back
        count_velocity_scaler = _ScalingMixin._zero_safe_div(
            count_vec,
            velo_vec
        )

        velocity_count_scaler = _ScalingMixin._zero_safe_div(
            velo_vec,
            count_vec
        )

        return (
            count_velocity_scaler,
            velocity_count_scaler
        )

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
