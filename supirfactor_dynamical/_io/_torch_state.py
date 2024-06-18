import numpy as np
import torch
import h5py
import collections


def write_module(
    model_object,
    file_name,
    mode='w',
    prefix=''
):

    with h5py.File(file_name, mode) as f:
        _write_torch_state(f, model_object, prefix)


def _write_torch_state(
    f,
    torch_object,
    prefix=''
):

    keys = []

    for k, data in torch_object.state_dict().items():

        f.create_dataset(
            prefix + k,
            data=data.to('cpu').numpy()
        )
        keys.append(k)

    f.create_dataset(
        prefix + 'keys',
        data=np.array(keys, dtype=object)
    )


def read_state_dict(
    file_name,
    mode='w',
    prefix=''
):

    with h5py.File(file_name, mode) as f:
        return _read_torch_state(f, prefix)


def _read_torch_state(
    f,
    prefix=''
):
    _pre_len = len(prefix)
    _state_dict = collections.OrderedDict()

    _state_dict_keys = [
        prefix + x.decode('utf-8')
        for x in _load_h5_dataset(f, prefix + 'keys')
    ]

    for k in _state_dict_keys:
        _state_dict[k[_pre_len:]] = torch.tensor(
            _load_h5_dataset(f, k)
        )

    return _state_dict


def _load_h5_dataset(
    h5_handle,
    key
):

    if key in h5_handle.keys():
        return h5_handle[key][()]

    else:
        return None
