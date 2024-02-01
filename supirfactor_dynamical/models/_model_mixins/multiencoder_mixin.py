class _MultiencoderModuleMixin:

    active_encoder = None

    _encoder_bag = None
    _encoder_labels = None

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

    def encoder_parameters(
        self,
        encoder_name
    ):

        self._check_encoder_label(encoder_name)
        return self.encoder_bag[encoder_name].parameters()

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

        self._check_encoder_label(encoder_name)
        self.encoder = self.encoder_bag[encoder_name]
        self.active_encoder = encoder_name

    def _check_encoder_label(self, label):

        if label not in self.encoder_bag.keys():
            raise ValueError(
                f"Cannot select {label} from available encoders: "
                f"{', '.join(self.encoder_bag.keys())}"
            )
