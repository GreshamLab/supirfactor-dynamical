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

    with pd.HDFStore(file_name, mode='r') as f:
        prior = pd.read_hdf(
            f,
            prefix + 'prior_network'
        )

    time_kwargs = {
        k: kwargs.pop(k, None) for k in TIME_KWARGS
    }

    model_type_kwargs = {
        k: kwargs.pop(k, False) for k in MODEL_TYPE_KWARGS
    }

    freeze_kwargs = {
        k: kwargs.pop(k, False) for k in FREEZE_ARGS
    }

    if (
        (_state_model == 'biophysical') and
        freeze_kwargs['_pretrained_count']
    ):
        kwargs['trained_count_model'] = read(
            file_name,
            prefix=prefix + 'count_'
        )

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

    if freeze_kwargs['_pretrained_count']:
        model.freeze(model._count_model)
        model._pretrained_count = True

    if freeze_kwargs['_pretrained_decay']:
        model.freeze(model._decay_model)
        model._pretrained_decay = True

    return model


def _load_h5_dataset(
    h5_handle,
    key
):

    if key in h5_handle.keys():
        return h5_handle[key][()]

    else:
        return None
