import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from .._utils import (
    _calculate_erv,
    _calculate_rss
)

from ._model_mixins import (
    _PriorMixin,
    _ScalingMixin
)


class _TFMixin(
    _PriorMixin,
    _ScalingMixin
):

    gene_loss_sum_axis = 0
    type_name = "base"

    hidden_final = None

    _velocity_model = False

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
        Forward model.
        This is *post* input dropout.

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
