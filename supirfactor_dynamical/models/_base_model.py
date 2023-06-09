import torch
import numpy as np
import pandas as pd
import tqdm
import warnings

from torch.nn.utils import prune
from torch.utils.data import DataLoader

from .._utils import (
    _process_weights_to_tensor,
    _calculate_erv,
    _calculate_rss,
    _calculate_tss,
    _calculate_r2,
    _aggregate_r2
)

from ._writer import write


DEFAULT_OPTIMIZER_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 1e-10
}


class _TFMixin:

    g = 0
    k = 0

    _drop_tf = None

    prior_network = None
    prior_network_labels = (None, None)

    training_loss = None
    validation_loss = None

    training_r2 = None
    validation_r2 = None

    gene_loss_sum_axis = 0
    type_name = "base"

    _serialize_args = [
        'input_dropout_rate',
        'output_relu',
        'prediction_length',
        'loss_offset'
    ]

    input_dropout_rate = 0.5
    output_relu = True

    prediction_offset = False
    prediction_length = 0
    loss_offset = 0

    hidden_final = None

    @property
    def encoder_weights(self):
        return self.encoder[0].weight

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
        Calls _forward_step and _forward_loop.


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
        x = self._forward_step(x, hidden_state)

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

    def _forward_step(
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
        x = self.decoder(x, hidden_state)

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

            x_tensor = self._forward_step(x_tensor, self.hidden_final)
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

    def drop_encoder(
        self,
        x
    ):
        x = self.encoder(x)

        if self._drop_tf is not None:
            x[:, self.prior_network_labels[1].isin(self._drop_tf)] = 0

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
        use_prior_weights=False
    ):

        prior_network = self.process_prior(
            prior_network
        )

        # Build the encoder module
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

    def set_time_parameters(
        self,
        prediction_length=None,
        loss_offset=None
    ):
        """
        Set parameters for dynamic prediction

        :param prediction_length: Number of time units to do forward
            prediction, defaults to None
        :type prediction_length: int, optional
        :param loss_offset: How much of the input data to exclude from
            the loss function, increasing the importance of prediction,
            defaults to None
        :type loss_offset: int, optional
        """

        if prediction_length is not None:
            self.prediction_length = prediction_length

        if loss_offset is not None:
            self.loss_offset = loss_offset

        self.prediction_offset = self.prediction_length > 0

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

    def save(
        self,
        file_name
    ):
        """
        Save this model to a file

        :param file_name: File name
        :type file_name: str
        """

        write(self, file_name)

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
                return torch.stack([
                    self._latent_layer_values(
                        batch,
                        layer=layer,
                        hidden_state=hidden_state
                    )
                    for batch in x
                ])

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
            x = self._intermediate(x, hidden_state)

            if isinstance(x, tuple):
                return [xobj.detach() for xobj in x]
            else:
                return x.detach()

    def input_data(self, x):
        """
        Process data from DataLoader for input nodes in training.
        If prediction_offset is set, return the first data point in the
        sequence length axis.

        :param x: Data
        :type x: torch.Tensor
        :return: Input node values
        :rtype: torch.Tensor
        """
        if self.prediction_offset:
            return x[:, [0], :]
        else:
            return x

    def output_data(self, x, offset_only=False):
        """
        Process data from DataLoader for output nodes in training.
        If prediction_offset is not None or zero, return the offset data in the
        sequence length axis.

        :param x: Data
        :type x: torch.Tensor
        :return: Output node values
        :rtype: torch.Tensor
        """

        # No need to do predictive offsets
        if not self.prediction_offset:
            return x

        # Don't shift for prediction if offset_only
        elif offset_only and self.loss_offset == 0:
            return x

        # Shift and truncate
        else:

            loss_offset = self.loss_offset

            if not offset_only:
                loss_offset += 1

            end_offset = 1 + self.prediction_length

            if x.ndim == 2:
                return x[loss_offset:end_offset, :]
            else:
                return x[:, loss_offset:end_offset, :]

    def train_model(
        self,
        training_dataloader,
        epochs,
        validation_dataloader=None,
        loss_function=torch.nn.MSELoss(),
        optimizer=None
    ):
        """
        Train this model

        :param training_dataloader: Training data in a torch DataLoader
        :type training_dataloader: torch.utils.data.DataLoader
        :param epochs: Number of training epochs
        :type epochs: int
        :param validation_dataloader: Validation data in a torch DataLoader,
            for calculating validation loss only. Defaults to None
        :type validation_dataloader: torch.utils.data.DataLoader, optional
        :param loss_function: Torch loss function object for training,
            defaults to torch.nn.MSELoss()
        :type loss_function: torch.nn.Loss(), optional
        :param optimizer: Torch optimizer for training. Can be an
            Optimizer instance, a dict of kwargs for torch.optim.Adam, or
            if None, defaults to Adam(lr=1e-3, weight_decay=1e-10)
        :type optimizer: torch.optim.object(), dict(), optional
        :return: Training losses and validation losses
            (if validation_dataloader is provided)
        :rtype: np.ndarray, np.ndarray
        """

        optimizer = self.process_optimizer(
            optimizer
        )

        losses = []

        if validation_dataloader is not None:
            validation_losses = []
        else:
            validation_losses = None

        for _ in tqdm.trange(epochs):

            self.train()

            _batch_losses = []
            for train_x in training_dataloader:

                mse = loss_function(
                    self._step_forward(train_x),
                    self.output_data(train_x)
                )

                mse.backward()
                optimizer.step()
                optimizer.zero_grad()

                _batch_losses.append(mse.item())

            losses.append(
                np.mean(_batch_losses)
            )

            # Get validation losses during training
            # if validation data was provided
            if validation_dataloader is not None:

                _validation_batch_losses = []

                with torch.no_grad():
                    for val_x in validation_dataloader:

                        _validation_batch_losses.append(
                            loss_function(
                                self._step_forward(val_x),
                                self.output_data(val_x)
                            ).item()
                        )

                validation_losses.append(
                    np.mean(_validation_batch_losses)
                )

            # Shuffle stratified time data
            # is a noop unless the underlying DataSet is a TimeDataset
            _shuffle_time_data(training_dataloader)
            _shuffle_time_data(validation_dataloader)

        self.training_loss = losses
        self.validation_loss = validation_losses

        self.eval()
        self.r2(
            training_dataloader,
            validation_dataloader
        )

        return losses, validation_losses

    def _step_forward(self, train_x):
        if self.prediction_length < 2:
            forward = self(
                self.input_data(train_x)
            )

        else:
            forward = self(
                self.input_data(train_x),
                n_time_steps=self.prediction_length - 1
            )

        return self.output_data(forward, offset_only=True)

    @torch.inference_mode()
    def r2(
        self,
        training_dataloader,
        validation_dataloader=None
    ):
        """
        Calculate unweighted-average R2 score and store in the model object

        :param training_dataloader: Training data in a torch DataLoader
        :type training_dataloader: torch.utils.data.DataLoader
        :param validation_dataloader: Validation data in a torch DataLoader,
            for calculating validation loss only. Defaults to None
        :type validation_dataloader: torch.utils.data.DataLoader, optional
        :returns: R2 for training and validation data (if provided)
        :rtype: float, float
        """

        self.eval()

        self.training_r2 = _aggregate_r2(
            self._calculate_r2_score(
                training_dataloader
            )
        )

        if validation_dataloader is not None:
            self.validation_r2 = _aggregate_r2(
                self._calculate_r2_score(
                    validation_dataloader
                )
            )

        return self.training_r2, self.validation_r2

    @torch.inference_mode()
    def _calculate_r2_score(
        self,
        dataloader
    ):

        if dataloader is None:
            return None

        _rss = 0
        _ss = 0

        with torch.no_grad():
            for data in dataloader:

                output_data = self.output_data(data)

                _rss += _calculate_rss(
                    output_data,
                    self._step_forward(data),
                )

                _ss += _calculate_tss(
                    output_data
                )

        return _calculate_r2(_rss, _ss)

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

            full_rss = torch.zeros(self.g)
            rss = torch.zeros((self.g, self.k))

            for data_x in data_loader:

                # Get TFA
                if self.prediction_length > 1:
                    hidden_x = self.latent_layer(
                        self._step_forward(data_x)
                    )

                else:
                    hidden_x = self.latent_layer(
                        self.input_data(data_x)
                    )

                data_x = self.output_data(data_x)

                full_rss += _calculate_rss(
                    self.output_data(
                        self.decoder(hidden_x),
                        offset_only=True
                    ),
                    data_x
                )

                # For each node in the latent layer,
                # zero all values in the data and then
                # decode to full expression data
                for ik in range(self.k):
                    latent_dropout = torch.clone(hidden_x)

                    if latent_dropout.ndim == 2:
                        latent_dropout[:, ik] = 0.
                    else:
                        latent_dropout[:, :, ik] = 0.

                    rss[:, ik] += _calculate_rss(
                        self.output_data(
                            self.decoder(latent_dropout),
                            offset_only=True
                        ),
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

    def process_optimizer(
        self,
        optimizer
    ):

        # If it's None, create a default Adam optimizer
        if optimizer is None:
            return torch.optim.Adam(
                self.parameters(),
                **DEFAULT_OPTIMIZER_PARAMS
            )

        # If it's an existing optimizer, return it
        elif isinstance(optimizer, torch.optim.Optimizer):
            return optimizer

        # Otherwise assume it's a dict of Adam kwargs
        else:
            return torch.optim.Adam(
                self.parameters(),
                **optimizer
            )

    @torch.inference_mode()
    def predict(
        self,
        x,
        n_time_steps
    ):

        if self.prediction_offset is None or self.prediction_offset == 0:
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


def _shuffle_time_data(dl):
    try:
        dl.dataset.shuffle()
    except AttributeError:
        pass
