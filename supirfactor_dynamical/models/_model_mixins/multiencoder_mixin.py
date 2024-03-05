from supirfactor_dynamical._io._torch_state import (
    write_module,
    read_state_dict
)


_DEFAULT_MODELS = [
    'default_encoder',
    'default_decoder',
    'default_intermediate'
]


class _MultiSubmoduleMixin:

    _multisubmodel_model = True

    active_encoder = _DEFAULT_MODELS[0]
    active_decoder = _DEFAULT_MODELS[1]
    active_intermediate = _DEFAULT_MODELS[2]

    _module_bag = None

    @property
    def module_bag(self):

        if self._module_bag is None:
            self._module_bag = {}
            self._module_bag[_DEFAULT_MODELS[0]] = self.encoder
            self._module_bag[_DEFAULT_MODELS[1]] = self._decoder

            if hasattr(self, '_intermediate'):
                self._module_bag[_DEFAULT_MODELS[2]] = self._intermediate

        return self._module_bag

    @property
    def module_labels(self):

        if self._module_bag is None:
            return None

        else:
            return list(self._module_bag.keys())

    def submodel_parameters(
        self,
        module_name
    ):

        self._check_label(module_name)
        return self.module_bag[module_name].parameters()

    def add_submodel(
        self,
        module_name,
        module
    ):

        if (
            self.module_labels is not None and
            module_name in self.module_labels
        ):
            raise ValueError(
                f"Submodel {module_name} exists:\n"
                f"{module}"
            )

        self.module_bag[module_name] = module

    def select_submodel(
        self,
        module_name,
        model_type='encoder'
    ):

        if model_type == 'encoder':
            self.select_submodels(encoder=module_name)
        elif model_type == 'decoder':
            self.select_submodels(decoder=module_name)
        elif model_type == 'intermediate':
            self.select_submodels(intermediate=module_name)
        else:
            raise ValueError(f"model_type {model_type} unknown")

    def select_submodels(
        self,
        encoder=None,
        intermediate=None,
        decoder=None
    ):

        if encoder is not None:
            self._check_label(encoder)
            self.encoder = self.module_bag[encoder]
            self.active_encoder = encoder

        if decoder is not None:
            self._check_label(decoder)
            self._decoder = self.module_bag[decoder]
            self.active_decoder = decoder

        if intermediate is not None:
            self._check_label(intermediate)
            self._intermediate = self.module_bag[intermediate]
            self.active_intermediate = intermediate

        self.train(self.training)

    def freeze_submodel(
        self,
        model_type,
        unfreeze=False
    ):

        if model_type == 'encoder':
            _model_ref = self.encoder
        elif model_type == 'intermediate':
            _model_ref = self._intermediate
        elif model_type == 'decoder':
            _model_ref = self._decoder
        else:
            raise ValueError(
                f"model_type must be `encoder`, `decoder`, or "
                f"`intermediate`; {model_type} provided"
            )

        for param in _model_ref.parameters():
            param.requires_grad = unfreeze

    def _check_label(self, label):

        if self.module_labels is not None and label not in self.module_labels:
            raise ValueError(
                f"Cannot select {label} from "
                f"available models: {', '.join(self.module_labels)}"
            )

    def default_submodules(self):

        self.select_submodels(
            encoder=_DEFAULT_MODELS[0],
            decoder=_DEFAULT_MODELS[1],
            intermediate=_DEFAULT_MODELS[2]
        )

    def is_active_module(self, module_name):

        self._check_label(module_name)
        return (
            (self.active_encoder == module_name) or
            (self.active_intermediate == module_name) or
            (self.active_decoder == module_name)
        )

    def save(
        self,
        file_name,
        **kwargs
    ):

        self.default_submodules()

        super().save(
            file_name,
            **kwargs
        )

    def save_submodel_state(
        self,
        module_name,
        file_name,
        **kwargs
    ):

        self._check_label(module_name)

        write_module(
            self.module_bag[module_name],
            file_name,
            **kwargs
        )

    def load_submodel_state(
        self,
        module_name,
        file_name,
        **kwargs
    ):

        self._check_label(module_name)

        self.module_bag[module_name].load_state_dict(
            read_state_dict(
                file_name,
                **kwargs
            )
        )
