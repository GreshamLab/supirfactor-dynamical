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


def read(
    file_name,
    model_class=None
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

    with h5py.File(file_name, 'r') as f:

        _state_dict = collections.OrderedDict()
        _state_dict_keys = [
            x.decode('utf-8')
            for x in _load_h5_dataset(f, 'keys')
        ]

        for k in _state_dict_keys:
            _state_dict[k] = torch.tensor(
                _load_h5_dataset(f, k)
            )

        _state_args = [
            x.decode('utf-8')
            for x in _load_h5_dataset(f, 'args')
        ]

        _state_model = _load_h5_dataset(
            f,
            'type_name'
        ).decode('utf-8')

        kwargs = {
            k: _load_h5_dataset(f, k)
            for k in _state_args
        }

    with pd.HDFStore(file_name, mode='r') as f:
        prior = pd.read_hdf(
            f,
            'prior_network'
        )

    time_kwargs = {
        k: kwargs.pop(k, None) for k in TIME_KWARGS
    }

    model_type_kwargs = {
        k: kwargs.pop(k, False) for k in MODEL_TYPE_KWARGS
    }

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

    model.set_time_parameters(
        **time_kwargs
    )

    model.load_state_dict(
        _state_dict
    )

    return model


def _load_h5_dataset(
    h5_handle,
    key
):

    if key in h5_handle.keys():
        return h5_handle[key][()]

    else:
        return None
