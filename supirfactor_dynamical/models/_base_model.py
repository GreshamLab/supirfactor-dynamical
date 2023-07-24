import torch
import numpy as np
import pandas as pd
import warnings

from torch.nn.utils import prune
from torch.utils.data import DataLoader

from .._utils import (
    _process_weights_to_tensor,
    _calculate_erv,
    _calculate_rss
)


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


class _TFMixin(_PriorMixin):

    gene_loss_sum_axis = 0
    type_name = "base"

    output_relu = True

    hidden_final = None

    _velocity_model = False

    _velocity_inverse_scaler = None
    _count_inverse_scaler = None

    scaler = None
    inv_scaler = None

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

        self.scaler, self.inv_scaler = self.make_scalers(
            self._count_inverse_scaler,
            self._velocity_inverse_scaler
        )

        return self

    @property
    def encoder_weights(self):
        return self.encoder[0].weight

    @property
    def intermediate_weights(self):
        return None

    @property
    def recurrent_weights(self):
        return None

    @property
    def decoder_weights(self):
        return self.decoder[0].weight

    def _forward(
        self,
        x,
        hidden_state=None,
        n_time_steps=0
    ):
        """
        Forward pass for data X with prediction if n_time_steps > 0.
        Calls forward_model and _forward_loop.


        :param x: Input data
        :type x: torch.Tensor
        :param hidden_state: h_0 hidden state, defaults to None
        :type hidden_state: torch.Tensor, tuple, optional
        :param n_time_steps: Number of forward prediction time steps,
            defaults to 0
        :type n_time_steps: int, optional
        :return: Output data (N, L, K)
        :rtype: torch.Tensor
        """

        x = self.input_dropout(x)
        x = self.forward_model(x, hidden_state)

        if n_time_steps > 0:

            # Force 1D data into 2D
            if x.ndim == 1:
                x = torch.unsqueeze(x, dim=0)

            # Feed it into the start of the forward loop
            forward_x = self._forward_loop(x, n_time_steps)

            # Add a new dimension for sequence length
            if x.ndim == 2:
                x = torch.unsqueeze(x, dim=1)

            # Cat together on time dimension 1
            x = torch.cat(
                (
                    x,
                    forward_x
                ),
                dim=1
            )

        return x

    def forward_model(
        self,
        x,
        hidden_state=None
    ):
        """
        Forward step to the model,
        override by inheritence for other models
        """
        return self.forward_tf_model(x, hidden_state=hidden_state)

    def forward_tf_model(
        self,
        x,
        hidden_state=None
    ):
        """
        Forward prop

        :param x: Input data
        :type x: torch.Tensor
        :param hidden_state: Model hidden state h_0,
            defaults to None
        :type hidden_state: torch.Tensor, optional
        :return: Predicted output
        :rtype: torch.Tensor
        """

        x = self.drop_encoder(x)

        if hidden_state is not None:
            x = self.decoder(x, hidden_state)
        else:
            x = self.decoder(x)

        return x

    def _forward_loop(
        self,
        x_tensor,
        n_time_steps
    ):
        """
        Forward prop recursively to predict future states.
        Models with hidden states will be use h_final from one
        iteration to initialize h_0 in the next iteration

        :param x: Input data
        :type x: torch.Tensor
        :param n_time_steps: Number of time iterations
        :type hidden_state: int
        :return: Predicted output. Adds a dimension if
            input data does not already have a sequence dimension
        :rtype: torch.Tensor
        """

        output_state = []

        for _ in range(n_time_steps):

            x_tensor = self.forward_model(x_tensor, self.hidden_final)
            output_state.append(x_tensor)

        return self._forward_loop_merge(output_state)

    @staticmethod
    def _forward_loop_merge(tensor_list):
        """
        Merge data that does not have a sequence length dimension
        by adding a new dimension

        Merge data that does have a sequence length dimension
        by concatenating on that dimension

        :param tensor_list: List of predicted tensors
        :type tensor_list: list(torch.Tensor)
        :return: Stacked tensor
        :rtype: torch.Tensor
        """
        if tensor_list[0].ndim < 3:
            return torch.stack(
                tensor_list,
                dim=tensor_list[0].ndim - 1
            )

        else:
            return torch.cat(
                tensor_list,
                dim=tensor_list[0].ndim - 2
            )

    def set_decoder(
        self,
        relu=True,
        decoder_weights=None
    ):
        """
        Set decoder, apply weights and enforce decoder sparsity structure

        :param relu: Apply activation function (ReLU) to decoder output
            layer, defaults to True
        :type relu: bool, optional
        :param decoder_weights: Values to use as the initialization for
            decoder weights. Any values that are zero will be pruned to enforce
            the same sparsity structure after training. Will skip if None.
            Defaults to None.
        :type decoder_weights: pd.DataFrame [G x K], np.ndarray, optional
        """

        self.output_relu = relu

        decoder = torch.nn.Sequential(
            torch.nn.Linear(self.k, self.g, bias=False),
        )

        if relu:
            decoder.append(
                torch.nn.ReLU()
            )

        if decoder_weights is not None:

            decoder_weights, _ = _process_weights_to_tensor(
                decoder_weights,
                transpose=False
            )

            decoder[0].weight = torch.nn.parameter.Parameter(
                decoder_weights,
            )

            prune.custom_from_mask(
                decoder[0],
                name='weight',
                mask=decoder_weights != 0,
            )

        return decoder

    def mask_input_weights(
        self,
        mask,
        use_mask_weights=False,
        layer_name='weight_ih_l0',
        weight_vstack=None
    ):

        if isinstance(self.encoder, torch.nn.Sequential):
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

    @torch.inference_mode()
    def output_weights(
        self,
        as_dataframe=False,
        mask=None
    ):
        """
        Return a 2D [G x K] array or DataFrame with decoder weights

        :param as_dataframe: Return pandas DataFrame with labels
            (if labels exist), defaults to False
        :type as_dataframe: bool, optional
        :param mask: 2D mask for weights (False prunes weight to zero),
            defaults to None
        :type mask: np.ndarray, optional
        :return: 2D [G x K] model weights as a numpy array or pandas DataFrame
        :rtype: np.ndarray, pd.DataFrame
        """

        # Get G x K decoder weights
        with torch.no_grad():
            w = self.decoder_weights.numpy()
            w[np.abs(w) <= np.finfo(np.float32).eps] = 0

        if mask is not None:
            w[~mask] = 0

        if as_dataframe:
            return self._to_dataframe(w)

        else:
            return w

    @torch.inference_mode()
    def pruned_model_weights(
        self,
        erv=None,
        data_loader=None,
        erv_threshold=1e-4,
        as_dataframe=False
    ):
        """
        Gets output weights pruned to zeros based on an
        explained relative variance threshold

        :param erv: Precalculated ERV dataframe/array [G x K],
            defaults to None
        :type erv: np.ndarray, pd.DataFrame, optional
        :param data_loader: Dataloader to calcualte ERV,
            defaults to None
        :type data_loader: torch.utils.data.DataLoader, optional
        :param erv_threshold: Threshold for trimming based on ERV,
            defaults to 1e-4
        :type erv_threshold: float, optional
        :param as_dataframe: Return as dataframe instead of array,
            defaults to False
        :type as_dataframe: bool, optional
        :return: Model weights trimmed to zero
        :rtype: pd.DataFrame, np.ndarray
        """

        if erv is not None:
            erv_mask = erv >= erv_threshold
        elif data_loader is not None:
            erv_mask = self.erv(data_loader) >= erv_threshold
        else:
            raise ValueError(
                "Pass erv or data_loader to `pruned_model_weights`"
            )

        out_weights = self.output_weights(
            as_dataframe=as_dataframe,
            mask=erv_mask
        )

        return out_weights

    @torch.inference_mode()
    def latent_layer(self, x, layer=0, hidden_state=None):
        """
        Get detached tensor representing the latent layer values
        for some data X

        :param x: Input data tensor [N x G]
        :type x: torch.Tensor
        :return: Detached latent layer values [N x K]
        :rtype: torch.Tensor
        """

        with torch.no_grad():
            if isinstance(x, DataLoader):
                return torch.cat(
                    [
                        self._latent_layer_values(
                            batch,
                            layer=layer,
                            hidden_state=hidden_state
                        )
                        for batch in x
                    ],
                    dim=0
                )

            else:
                return self._latent_layer_values(
                    x,
                    layer=layer,
                    hidden_state=hidden_state
                )

    @torch.inference_mode()
    def _latent_layer_values(
        self,
        x,
        layer=0,
        hidden_state=None
    ):

        if layer == 0:
            return self.drop_encoder(x).detach()
        elif layer == 1:
            x = self.drop_encoder(x)

            if hidden_state is not None:
                x = self._intermediate(x, hidden_state)
            else:
                x = self._intermediate(x)

            if isinstance(x, tuple):
                return [xobj.detach() for xobj in x]
            else:
                return x.detach()

    @torch.inference_mode()
    def erv(
        self,
        data_loader,
        output_data_loader=None,
        return_rss=False,
        as_data_frame=False
    ):
        """
        Calculate explained relative variance for each
        hidden node as 1 - RSS_full / RSS_reduced

        :param data_loader: DataLoader with data to calculate ERV
        :type data_loader: torch.utils.data.DataLoader
        :param return_rss: Return RSS_full and RSS_reduced,
            defaults to False
        :type return_rss: bool, optional
        :param as_data_frame: Return DataFrames [G x K] instead,
            defaults to False
        :type as_data_frame: bool, optional
        :return: ERV [G x K], RSS_reduced [G x K], RSS_full [G]
        :rtype: np.ndarray, np.ndarray, np.ndarray
        """

        with torch.no_grad():

            self.eval()

            full_rss = torch.zeros(self.g)
            rss = torch.zeros((self.g, self.k))

            for data_x in data_loader:

                # Get TFA
                if self.n_additional_predictions > 0:

                    # Need to run the predictions to get values for TFA
                    hidden_x = self.latent_layer(
                        self(
                            self.input_data(data_x),
                            n_time_steps=self.n_additional_predictions
                        )
                    )

                else:
                    hidden_x = self.latent_layer(
                        self.input_data(data_x)
                    )

                if output_data_loader is not None:
                    data_x = next(output_data_loader)
                else:
                    data_x = self.output_data(data_x, no_loss_offset=True)

                full_rss += _calculate_rss(
                    self.decoder(hidden_x),
                    data_x
                )

                # For each node in the latent layer,
                # zero all values in the data and then
                # decode to full expression data
                for ik in range(self.k):
                    latent_dropout = torch.clone(hidden_x)

                    latent_dropout[..., ik] = 0.

                    rss[:, ik] += _calculate_rss(
                        self.decoder(latent_dropout),
                        data_x
                    )

            # Calculate explained variance from the
            # full model RSS and the reduced model RSS
            erv = _calculate_erv(full_rss, rss)

        if as_data_frame:
            erv = self._to_dataframe(erv)

        if as_data_frame and return_rss:

            rss = self._to_dataframe(rss.numpy())
            full_rss = pd.DataFrame(
                full_rss,
                index=self.prior_network_labels[0]
            )

            return erv, rss, full_rss

        elif return_rss:
            return erv, rss.numpy(), full_rss.numpy()

        else:
            return erv

    @torch.inference_mode()
    def predict(
        self,
        x,
        n_time_steps
    ):

        if not self.output_t_plus_one:
            raise RuntimeError(
                "Model not trained for prediction"
            )

        self.eval()

        # Recursive call if x is a DataLoader
        if isinstance(x, DataLoader):
            return torch.cat(
                [
                    self.forward(
                        batch_x,
                        n_time_steps=n_time_steps
                    )
                    for batch_x in x
                ],
                dim=0
            )

        elif not torch.is_tensor(x):
            x = torch.Tensor(x)

        return self.forward(
            x,
            n_time_steps=n_time_steps
        )

    @staticmethod
    def make_scalers(
        count_vec,
        velo_vec=None
    ):

        if count_vec is None and velo_vec is None:
            return None, None

        elif count_vec is not None and velo_vec is not None:
            _scaler = torch.div(
                count_vec,
                velo_vec
            )

            _scaler[velo_vec == 0] = 1

        elif count_vec is not None:
            _scaler = count_vec

        elif velo_vec is not None:
            _scaler = 1 / velo_vec
            _scaler[velo_vec == 0] = 1

        scaler_1 = torch.diag(_scaler)
        scaler_2 = 1 / _scaler
        scaler_2[_scaler == 0] = 0
        scaler_2 = torch.diag(scaler_2)

        return scaler_1, scaler_2
