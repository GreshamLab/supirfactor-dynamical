import torch
import pandas as pd

from torch.utils.data import DataLoader

from .._utils import (
    _calculate_erv,
    _calculate_rss,
    _unsqueeze
)

from ._model_mixins import (
    _PriorMixin,
    _ScalingMixin
)


class _TFMixin(
    _PriorMixin,
    _ScalingMixin
):

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
        n_time_steps=0,
        return_tfa=False
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

        if x.ndim == 1:
            x = _unsqueeze(x, 0)

        x = self.input_dropout(x)
        x = self.forward_model(x, hidden_state, return_tfa)

        if n_time_steps > 0:

            # Add a new dimension for sequence length
            if return_tfa and x[0].ndim == 2:
                x = _unsqueeze(x, 1)
            elif not return_tfa and x.ndim == 2:
                x = _unsqueeze(x, 1)

            # Feed it into the start of the forward loop
            forward_x = self._forward_loop(
                x if not return_tfa else x[0],
                n_time_steps,
                return_tfa
            )

            # Cat together on time dimension 1
            x = self._forward_loop_merge(
                [
                    x,
                    forward_x
                ]
            )

        return x

    def forward_model(
        self,
        x,
        hidden_state=None,
        return_tfa=False
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

        if return_tfa:
            _tfa = torch.clone(x.detach())

        if hidden_state is not None:
            x = self.decoder(x, hidden_state)
        else:
            x = self.decoder(x)

        if return_tfa:
            return x, _tfa

        return x

    def _forward_loop(
        self,
        x_tensor,
        n_time_steps,
        return_tfa=False
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

            if return_tfa:
                x_tensor, tfa_tensor = self.forward_model(
                    x_tensor,
                    self.hidden_final,
                    return_tfa=return_tfa
                )
                output_state.append((x_tensor, tfa_tensor))
            else:
                x_tensor = self.forward_model(
                    x_tensor,
                    self.hidden_final
                )
                output_state.append(x_tensor)

        return self._forward_loop_merge(output_state)

    def _forward_loop_merge(self, tensor_list):
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
        if isinstance(tensor_list[0], tuple):
            return tuple(
                self._forward_loop_merge([
                    t[i] for t in tensor_list
                ])
                for i in range(2)
            )

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
    def latent_layer(self, x):
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
                        self.drop_encoder(batch).detach()
                        for batch in x
                    ],
                    dim=0
                )

            else:
                return torch.clone(
                    self.drop_encoder(x).detach()
                )

    @torch.inference_mode()
    def erv(
        self,
        data_loader,
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

                _full, _partial = self._calculate_error(
                    self.input_data(data_x),
                    self.output_data(data_x, no_loss_offset=True),
                    self.n_additional_predictions
                )

                full_rss += _full
                rss += _partial

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

    def _calculate_error(
        self,
        input_data,
        output_data,
        n_additional_predictions
    ):

        # Get TFA
        with torch.no_grad():
            _, hidden_x = self(
                input_data,
                n_time_steps=n_additional_predictions,
                return_tfa=True
            )

        full_rss = _calculate_rss(
            self.decoder(hidden_x),
            output_data
        )

        rss = torch.zeros((self.g, self.k))
        # For each node in the latent layer,
        # zero all values in the data and then
        # decode to full expression data
        for ik in range(self.k):
            latent_dropout = torch.clone(hidden_x)

            latent_dropout[..., ik] = 0.

            with torch.no_grad():
                rss[:, ik] = _calculate_rss(
                    self.decoder(latent_dropout),
                    output_data
                )

        return full_rss, rss
