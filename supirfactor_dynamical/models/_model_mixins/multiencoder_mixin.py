import torch

from supirfactor_dynamical._io._torch_state import (
    write_module,
    read_state_dict
)


_DEFAULT_MODELS = [
    'default_encoder',
    'default_decoder',
    'default_intermediate'
]


class ModuleBag(torch.nn.Module):

    labels = None

    def __init__(self):
        super().__init__()
        self.labels = []

    def __setitem__(self, module_name, module):
        setattr(self, module_name, module)
        self.labels.append(module_name)

    def __getitem__(self, module_name):
        if module_name not in self.labels:
            raise KeyError(f"{module_name} not in labels {self.labels}")

        return getattr(self, module_name)


class _MultiSubmoduleMixin:

    _multisubmodel_model = True

    active_encoder = _DEFAULT_MODELS[0]
    active_decoder = _DEFAULT_MODELS[1]
    active_intermediate = _DEFAULT_MODELS[2]

    @property
    def module_bag(self):

        self._initialize_submodule_container()
        return self._module_bag

    @property
    def module_labels(self):

        self._initialize_submodule_container()
        return self._module_bag.labels

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

            if self.active_encoder != encoder:
                self.encoder = self.module_bag[encoder]
                self.active_encoder = encoder

        if decoder is not None:
            self._check_label(decoder)

            if self.active_decoder != decoder:
                self._decoder = self.module_bag[decoder]
                self.active_decoder = decoder

        if intermediate is not None:
            self._check_label(intermediate)

            if self.active_intermediate != intermediate:
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
        elif model_type in self.module_bag.labels:
            _model_ref = self.module_bag[model_type]
        else:
            raise ValueError(
                f"model_type must be `encoder`, `decoder`, "
                f"`intermediate`, or one of the module names "
                f"({', '.join(self.module_labels)}); {model_type} provided"
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

    def active_modules(self):
        return (
            self.active_encoder,
            self.active_intermediate,
            self.active_decoder
        )

    def save(
        self,
        file_name,
        **kwargs
    ):

        # Implementation detail
        # save expects default modules
        _modules = self.active_modules()
        self.default_submodules()

        super().save(
            file_name,
            **kwargs
        )

        # Restore existing modules
        self.select_submodels(*_modules)

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

    def _initialize_submodule_container(self):

        if not hasattr(self, '_module_bag'):
            setattr(self, '_module_bag', ModuleBag())
            self._module_bag[_DEFAULT_MODELS[0]] = self.encoder
            self._module_bag[_DEFAULT_MODELS[1]] = self._decoder

            if hasattr(self, '_intermediate'):
                self._module_bag[_DEFAULT_MODELS[2]] = self._intermediate

    def named_active_parameters(self):

        for name, value in self.named_parameters():
            if not name.startswith('_module_bag'):
                yield name, value

    def active_parameters(self):

        for name, value in self.named_active_parameters():
            yield value
