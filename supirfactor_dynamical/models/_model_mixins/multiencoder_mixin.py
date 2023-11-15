import pandas as pd


class _MultiencoderModuleMixin:

    active_encoder = None

    _encoder_bag = None
    _encoder_labels = None

    _freeze_non_encoder = False

    @property
    def encoder_bag(self):

        if self._encoder_bag is None:
            self._encoder_bag = {}
            self._encoder_bag['linear'] = self.encoder
            self.encoder_labels.append('linear')
            self.active_encoder = 'linear'

        return self._encoder_bag

    @property
    def encoder_labels(self):

        if self._encoder_labels is None:
            self._encoder_labels = []

        return self._encoder_labels

    def unfrozen_parameters(self):

        if self._freeze_non_encoder is None:
            return self.encoder.parameters()
        else:
            return self.parameters()

    def add_encoder(
        self,
        encoder_name,
        module
    ):

        self.encoder_bag[encoder_name] = module
        self.encoder_labels.append(encoder_name)

    def select_encoder(
        self,
        encoder_name
    ):

        if encoder_name not in self.encoder_bag.keys():
            raise ValueError(
                f"Cannot select {encoder_name} from available encoders: "
                f"{', '.join(self.encoder_bag.keys())}"
            )

        self.encoder = self.encoder_bag[encoder_name]
        self.active_encoder = encoder_name

    def train_model(
        self,
        *args,
        optimizer=None,
        **kwargs
    ):

        # Create separate optimizers for the decay and transcription
        # models and pass them as tuple
        optimizers = []

        for k in self.encoder_labels:
            self.select_encoder(k)
            optimizers.append(
                self.process_optimizer(
                    optimizer,
                    params=self.unfrozen_parameters()
                )
            )

        return super().train_model(
            *args,
            optimizer=optimizers,
            **kwargs
        )

    def _training_step(
        self,
        epoch_num,
        train_x,
        optimizer,
        loss_function
    ):
        """
        Do a training step for each encoder

        :param epoch_num: Epoch number
        :type epoch_num: int
        :param train_x: Training data (N, L, G, ...)
        :type train_x: torch.Tensor
        :param optimizer: Tuple of optimizers
        :type optimizer: tuple(torch.optim)
        :param loss_function: Loss function
        :type loss_function: torch.nn.Loss
        :return: Returns loss for each training
        :rtype: float
        """

        if not isinstance(loss_function, (tuple, list)):
            loss_function = [loss_function] * len(optimizer)

        losses = []
        for i, k in enumerate(self.encoder_labels):
            self.select_encoder(k)
            losses.append(
                super()._training_step(
                    epoch_num,
                    train_x,
                    optimizer[i],
                    loss_function[i]
                )
            )

        return losses

    def _loss_df(self, loss_array):

        if loss_array.size == 0:
            return None
        elif loss_array.ndim == 1:
            loss_array = loss_array.reshape(-1, 1)

        _loss = pd.DataFrame(loss_array.T)

        if self._loss_type_names is not None:
            _labeler = pd.DataFrame(
                self._loss_type_names,
                columns=['loss_model']
            )
            _labeler['encoder'] = self.encoder_labels * _labeler.shape[0]
            _loss = pd.concat(
                _loss,
                _labeler.explode('encoder').reset_index(drop=True)
            )

        else:
            _loss.insert(0, 'encoder', self.encoder_labels)

        return _loss
