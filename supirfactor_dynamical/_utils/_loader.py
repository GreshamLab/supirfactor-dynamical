import h5py
import collections
import torch
import pandas as pd

from ..models import (
    get_model
)

TIME_KWARGS = [
    'output_t_plus_one',
    'n_additional_predictions',
    'loss_offset'
]

MODEL_TYPE_KWARGS = [
    '_velocity_model'
]

FREEZE_ARGS = [
    '_pretrained_count',
    '_pretrained_decay'
]

SCALER_ARGS = [
    ('_count_inverse_scaler', 'count_scaling'),
    ('_velocity_inverse_scaler', 'velocity_scaling')
]

DEPRECATED_ARGS = [
    '_decay_model',
    '_pretrained_count',
    '_pretrained_decay',
    'time_dependent_decay'
]

INFO_KWARGS = [
    'training_time',
    '_training_loss',
    '_validation_loss',
    'training_r2',
    'validation_r2'
]

_SERIALIZE_ENCODED_ARGS = [
    'output_activation',
    'activation'
]

_DECODE_ACTIVATIONS = {
    0: None,
    1: 'relu',
    2: 'softplus',
    3: 'sigmoid',
    4: 'tanh'
}

_FORCE_UNIT = {
    'decay_k': int,
    'k': int,
    'g': int,
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

        _state_dict = collections.OrderedDict()

        _state_dict_keys = [
            prefix + x.decode('utf-8')
            for x in _load_h5_dataset(f, prefix + 'keys')
        ]

        for k in _state_dict_keys:
            _state_dict[k[_pre_len:]] = torch.tensor(
                _load_h5_dataset(f, k)
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
        }

    for k, func in _FORCE_UNIT.items():
        if k in kwargs and kwargs[k] is not None:
            kwargs[k] = func(kwargs[k])

    # Get the prior and pop out the prior
    # size variables
    try:
        with pd.HDFStore(file_name, mode='r') as f:
            prior = pd.read_hdf(
                f,
                prefix + 'prior_network'
            )

        kwargs.pop('g', None)
        kwargs.pop('k', None)

    # Or get g out of the kwargs for decay model
    except KeyError:
        prior = kwargs.pop('g', None)

    time_kwargs = {
        k: kwargs.pop(k, None) for k in TIME_KWARGS
    }

    model_type_kwargs = {
        k: kwargs.pop(k, False) for k in MODEL_TYPE_KWARGS
    }

    freeze_kwargs = {
        k: kwargs.pop(k, False) for k in FREEZE_ARGS
    }

    scaling_kwargs = {
        k[1]: kwargs.pop(k[0], None) for k in SCALER_ARGS
    }

    info_kwargs = {
        k: kwargs.pop(k, None) for k in INFO_KWARGS
    }

    for dead_arg in DEPRECATED_ARGS:
        kwargs.pop(dead_arg, None)

    for encoded_arg in _SERIALIZE_ENCODED_ARGS:
        if encoded_arg in kwargs:
            kwargs[encoded_arg] = _DECODE_ACTIVATIONS[kwargs[encoded_arg]]

    # Do special loading stuff for the big biophysical model
    if _state_model == 'biophysical':

        # Load a decay model if one exists, otherwise no decay model
        if any([k.startswith("_decay") for k in _state_dict_keys]):
            kwargs['decay_model'] = None
        else:
            kwargs['decay_model'] = False

    if model_class is None:
        model = get_model(
            _state_model,
            velocity=model_type_kwargs['_velocity_model']
        )(
            prior,
            **kwargs
        )
    else:
        model = model_class(
            prior,
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

    if freeze_kwargs['_pretrained_decay']:
        model._pretrained_decay = True

    for k, v in info_kwargs.items():
        setattr(model, k, v)

    return model


def _load_h5_dataset(
    h5_handle,
    key
):

    if key in h5_handle.keys():
        return h5_handle[key][()]

    else:
        return None
