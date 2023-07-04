import torch

from ._base_model import _TFMixin
from .._utils import _aggregate_r2


class _TF_RNN_mixin(
    torch.nn.Module,
    _TFMixin
):

    gene_loss_sum_axis = (0, 1)

    training_r2_over_time = None
    validation_r2_over_time = None

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
        input_dropout_rate=0.5,
        hidden_dropout_rate=0.0,
        output_relu=True,
        decoder_weights=None,
        sigmoid=False
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
        :param recurrency_mask: Removed
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
            use_prior_weights=use_prior_weights,
            sigmoid=sigmoid
        )

        # Build standard ReLU RNN
        # to encode connection between TF layer
        # and output layer

        self._intermediate = self._create_intermediate(self.k)

        self._decoder = self.set_decoder(
            relu=output_relu,
            decoder_weights=decoder_weights
        )

        self.set_dropouts(
            input_dropout_rate,
            hidden_dropout_rate
        )

    @staticmethod
    def _create_intermediate(k):
        raise NotImplementedError

    def input_data(self, x):

        if self.output_t_plus_one:
            input_offset, _ = self._get_data_offsets(x)
            return x[:, 0:input_offset, :]

        else:
            return x

    def output_data(self, x, truncate=True, **kwargs):
        return super().output_data(x, truncate=False, **kwargs)

    @torch.inference_mode()
    def r2_over_time(
        self,
        training_dataloader,
        validation_dataloader=None
    ):

        self.eval()

        self.training_r2_over_time = [
            _aggregate_r2(
                self._calculate_r2_score([x])
            )
            for x in training_dataloader.dataset.get_times_in_order()
        ]

        if validation_dataloader is not None:

            self.validation_r2_over_time = [
                _aggregate_r2(
                    self._calculate_r2_score([x])
                )
                for x in validation_dataloader.dataset.get_times_in_order()
            ]

        return self.training_r2_over_time, self.validation_r2_over_time

    def decoder(self, x, hidden_state=None, intermediate_only=False):

        x, self.hidden_final = self._intermediate(x, hidden_state)

        if not intermediate_only:
            x = self._decoder(x)

        return x

    def forward(
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
        :return: Output data
        :rtype: torch.Tensor
        """

        x = self.input_dropout(x)
        x = self.forward_model(x, hidden_state)

        if n_time_steps > 0:

            # Feed the last sequence value into the start of the forward loop
            # and glue it onto the earlier data
            x = torch.cat(
                (
                    x,
                    self._forward_loop(
                        x[:, [-1], :] if x.ndim == 3 else x[[-1], :],
                        n_time_steps
                    )
                ),
                dim=x.ndim - 2
            )

        return x

    @staticmethod
    def _forward_loop_merge(tensor_list):
        """
        Merge data that does have a sequence length dimension
        by concatenating on that dimension

        :param tensor_list: List of predicted tensors
        :type tensor_list: list(torch.Tensor)
        :return: Concatenated tensor
        :rtype: torch.Tensor
        """

        return torch.cat(
            tensor_list,
            dim=tensor_list[0].ndim - 2
        )
