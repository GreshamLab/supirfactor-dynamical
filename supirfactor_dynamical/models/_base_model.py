import h5py
import torch
import numpy as np
import pandas as pd
import tqdm

from torch.nn.utils import prune
from torch.utils.data import DataLoader

from ._utils import (
    _process_weights_to_tensor,
    _calculate_erv,
    _calculate_rss,
    _calculate_tss,
    _calculate_r2,
    _aggregate_r2
)

DEFAULT_OPTIMIZER_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 1e-10
}


class _TFMixin:

    g = 0
    k = 0

    prior_network = None
    prior_network_labels = None

    training_loss = None
    validation_loss = None

    training_r2 = None
    validation_r2 = None

    gene_loss_sum_axis = 0
    type_name = "base"

    _serialize_args = []

    prediction_offset = None
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

        with h5py.File(file_name, 'w') as f:
            for k, data in self.state_dict().items():
                f.create_dataset(
                    k,
                    data=data.numpy()
                )

            for s_arg in self._serialize_args:

                if getattr(self, s_arg) is not None:
                    f.create_dataset(
                        s_arg,
                        data=getattr(self, s_arg)
                    )

            f.create_dataset(
                'keys',
                data=np.array(
                    list(self.state_dict().keys()),
                    dtype=object
                )
            )

            f.create_dataset(
                'args',
                data=np.array(
                    self._serialize_args,
                    dtype=object
                )
            )

            f.create_dataset(
                'type_name',
                data=self.type_name
            )

        with pd.HDFStore(file_name, mode="a") as f:
            self._to_dataframe(
                self.prior_network,
                transpose=True
            ).to_hdf(
                f,
                'prior_network'
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

    def mask_recurrent_weights(
        self,
        mask,
        layer_name='weight_hh_l0',
        recurrent_object=None,
        n_to_stack=None
    ):
        """
        Apply a mask to the recurrency weights layer

        :param mask: Boolean mask
        :type mask: torch.Tensor
        """

        if recurrent_object is None:
            recurrent_object = self.encoder

        m, n = getattr(recurrent_object, layer_name).shape

        if mask is False:
            return

        elif mask is None:
            mask = torch.eye(
                n,
                dtype=bool
            )

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask != 0, dtype=torch.bool)

        if n_to_stack is not None and mask.shape[0] != m:
            mask = torch.vstack([mask for _ in range(n_to_stack)])

        prune.custom_from_mask(
            recurrent_object,
            name=layer_name,
            mask=mask
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
                return torch.stack([
                    self._latent_layer_values(batch)
                    for batch in x
                ])

            else:
                return self._latent_layer_values(x)

    @torch.inference_mode()
    def _latent_layer_values(self, x):
        return self.encoder(x).detach()

    def input_data(self, x):
        """
        Process data from DataLoader for input nodes in training.
        If prediction_offset is not None or zero, return the first data in the
        sequence length axis.

        :param x: Data
        :type x: torch.Tensor
        :return: Input node values
        :rtype: torch.Tensor
        """
        if self.prediction_offset is None or self.prediction_offset == 0:
            return x
        else:
            return x[:, 0, :]

    def output_data(self, x):
        """
        Process data from DataLoader for output nodes in training.
        If prediction_offset is not None or zero, return the offset data in the
        sequence length axis.

        :param x: Data
        :type x: torch.Tensor
        :return: Output node values
        :rtype: torch.Tensor
        """
        if self.prediction_offset is None or self.prediction_offset == 0:
            return x
        else:
            return x[:, self.prediction_offset, :]

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
                    self(self.input_data(train_x)),
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
                                self(self.input_data(val_x)),
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

                input_data = self.input_data(data)
                output_data = self.output_data(data)

                _rss += _calculate_rss(
                    output_data,
                    self(input_data)
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

                hidden_x = self.latent_layer(
                    self.input_data(data_x)
                )

                data_x = self.output_data(data_x)

                full_rss += _calculate_rss(
                    self.decoder(hidden_x),
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
            return torch.squeeze(
                torch.stack([
                    self.predict(batch_x, n_time_steps)
                    for batch_x in x
                ]),
                dim=1
            )

        elif not torch.is_tensor(x):
            x = torch.Tensor(x)

        return self._predict_loop(x, n_time_steps)

    @torch.inference_mode()
    def _predict_loop(
        self,
        x_tensor,
        n_time_steps
    ):

        output_state = []

        for _ in range(n_time_steps):

            x_tensor = self.forward(x_tensor)
            output_state.append(x_tensor)

        return torch.stack(
            output_state,
            dim=x_tensor.ndim - 1
        )


class _TF_RNN_mixin(_TFMixin):

    initial_state = None
    hidden_initial = None

    gene_loss_sum_axis = (0, 1)

    training_r2_over_time = None
    validation_r2_over_time = None

    @property
    def encoder_weights(self):
        return self.encoder.weight_ih_l0

    @property
    def recurrent_weights(self):
        return self.encoder.weight_hh_l0

    def input_data(self, x):

        if self.prediction_offset is None or self.prediction_offset == 0:
            return super().input_data(x)

        L = x.shape[-2]

        if x.ndim == 2:
            return x[0:L - self.prediction_offset, :]
        else:
            return x[:, 0:L - self.prediction_offset, :]

    def output_data(self, x):

        if self.prediction_offset is None or self.prediction_offset == 0:
            return super().output_data(x)

        if x.ndim == 2:
            return x[self.prediction_offset:, :]
        else:
            return x[:, self.prediction_offset:, :]

    @torch.inference_mode()
    def r2_over_time(
        self,
        training_dataloader,
        validation_dataloader=None
    ):

        self.eval()
        self.hidden_initial = None

        self.training_r2_over_time = [
            _aggregate_r2(
                self._calculate_r2_score([x])
            )
            for x in training_dataloader.dataset.get_times_in_order()
        ]

        if validation_dataloader is not None:
            self.hidden_initial = None

            self.validation_r2_over_time = [
                _aggregate_r2(
                    self._calculate_r2_score([x])
                )
                for x in validation_dataloader.dataset.get_times_in_order()
            ]

        return self.training_r2_over_time, self.validation_r2_over_time

    @torch.inference_mode()
    def _latent_layer_values(self, x):
        return self.encoder(x)[0].detach()

    @torch.inference_mode()
    def _predict_loop(
        self,
        x_tensor,
        n_time_steps
    ):

        # Reset hidden state and make first prediction
        x_tensor = self.forward(x_tensor)

        # Grab the first predicted sequence value
        # from batched
        if x_tensor.ndim == 3:
            x_tensor = x_tensor[:, -1:, :]
            output_state = [torch.squeeze(x_tensor, 1)]

        # or unbatched data
        elif x_tensor.ndim == 2:
            x_tensor = x_tensor[-1:, :]
            output_state = [torch.squeeze(x_tensor, 0)]

        # Run the prediction loop
        for _ in range(n_time_steps - 1):

            # Keep reusing the hidden layer output
            x_tensor = self.forward(x_tensor, self.hidden_final)

            # Squeeze out extra length dimension
            # will be added back in by stack
            output_state.append(
                torch.squeeze(
                    x_tensor,
                    dim=0 if x_tensor.ndim == 2 else 1
                )
            )

        output = torch.stack(
            output_state,
            dim=0 if x_tensor.ndim == 2 else 1
        )

        return output


def _shuffle_time_data(dl):
    try:
        dl.dataset.shuffle()
    except AttributeError:
        pass
