import h5py

from ._network import read_network
from ._args import (
    _SERIALIZE_TIME_ARGS,
    _SERIALIZE_NETWORKS,
    _SCALER_ARGS,
    _SERIALIZE_MODEL_TYPE_ATTRS,
    _SERIALIZE_RUNTIME_ATTRS,
    _SERIALIZE_ENCODED_ARGS,
    _ENCODE_ACTIVATIONS
)
from ._torch_state import (
    _read_torch_state,
    _load_h5_dataset
)

from ..models import (
    get_model
)

_DECODE_ACTIVATIONS = {
    v: k for k, v in _ENCODE_ACTIVATIONS.items()
}

_FORCE_UNIT = {
    'decay_k': int,
    'hidden_layer_width': int
}


def read(
    file_name,
    model_class=None,
    prefix=''
):
    """
    Load a model from a file

    :param file_name: File name
    :type file_name: str
    :param model_class: Load this model class instead of
        using the file to determine model class,
        defaults to None
    :type model_class: class
    """

    _pre_len = len(prefix)

    with h5py.File(file_name, 'r') as f:

        _state_dict = _read_torch_state(
            f,
            prefix=prefix
        )

        _state_args = [
            prefix + x.decode('utf-8')
            for x in _load_h5_dataset(f, prefix + 'args')
        ]

        _state_model = _load_h5_dataset(
            f,
            prefix + 'type_name'
        ).decode('utf-8')

        kwargs = {
            k[_pre_len:]: _load_h5_dataset(f, k)
            for k in _state_args
            if k not in _SERIALIZE_NETWORKS
        }

    for k, func in _FORCE_UNIT.items():
        if k in kwargs and kwargs[k] is not None:
            kwargs[k] = func(kwargs[k])

    # Get the network information
    for net_arg in _SERIALIZE_NETWORKS:
        _net = read_network(
            file_name,
            prefix + net_arg
        )

        if _net is not None:
            kwargs[net_arg] = _net

    time_kwargs = {
        k: kwargs.pop(k, None) for k in _SERIALIZE_TIME_ARGS
    }

    model_type_kwargs = {
        k: kwargs.pop(k, False) for k in _SERIALIZE_MODEL_TYPE_ATTRS
    }

    scaling_kwargs = {
        k[1]: kwargs.pop(k[0], None) for k in _SCALER_ARGS
    }

    info_kwargs = {
        k: kwargs.pop(k, None) for k in _SERIALIZE_RUNTIME_ATTRS
    }

    for encoded_arg in _SERIALIZE_ENCODED_ARGS:
        if encoded_arg in kwargs:
            kwargs[encoded_arg] = _DECODE_ACTIVATIONS[kwargs[encoded_arg]]

    # Do special loading stuff for the big biophysical model
    if _state_model == 'biophysical':

        # Load a decay model if one exists, otherwise no decay model
        if any([k.startswith("_decay") for k in _state_dict.keys()]):
            kwargs['decay_model'] = None
        else:
            kwargs['decay_model'] = False

    if model_class is None:
        model = get_model(
            _state_model,
            velocity=model_type_kwargs['_velocity_model'],
            multisubmodel=model_type_kwargs['_multisubmodel_model'],
            multimodal_data=model_type_kwargs['_multimodal_data_model']
        )(
            **kwargs
        )
    else:
        model = model_class(
            **kwargs
        )

    if hasattr(model, 'set_time_parameters'):
        model.set_time_parameters(
            **time_kwargs
        )

    if hasattr(model, 'set_scaling'):
        model.set_scaling(
            **scaling_kwargs
        )

    # Load all the tensors in
    model.load_state_dict(
        _state_dict
    )

    for k, v in info_kwargs.items():
        setattr(model, k, v)

    return model
