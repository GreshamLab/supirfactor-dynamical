import h5py
import numpy as np

from ._network import write_network
from ._args import (
    _SERIALIZE_RUNTIME_ATTRS,
    _SCALER_ARGS,
    _SERIALIZE_MODEL_TYPE_ATTRS,
    _SERIALIZE_ARGS,
    _SERIALIZE_NETWORKS,
    _SERIALIZE_TIME_ARGS,
    _SERIALIZE_ENCODED_ARGS,
    _ENCODE_ACTIVATIONS
)


def write(
    model_object,
    file_name,
    mode='w',
    prefix=''
):

    with h5py.File(file_name, mode) as f:
        _write_state(f, model_object, prefix)

    for net_attr in _SERIALIZE_NETWORKS:
        if hasattr(model_object, net_attr) and net_attr == "prior_network":
            write_network(
                file_name,
                model_object.prior_network_dataframe,
                prefix + 'prior_network'
            )

        elif hasattr(model_object, net_attr):
            write_network(
                file_name,
                getattr(model_object, net_attr),
                prefix + net_attr
            )


def _write_state(
    f,
    model_object,
    prefix=''
):

    if hasattr(model_object, '_serialize_args'):
        _serialize_args = model_object._serialize_args
    else:
        _serialize_args = _SERIALIZE_ARGS

    _serialize_args = [
        arg for arg in
        _serialize_args +
        _SERIALIZE_RUNTIME_ATTRS +
        _SERIALIZE_TIME_ARGS +
        [x[0] for x in _SCALER_ARGS] +
        _SERIALIZE_MODEL_TYPE_ATTRS
        if hasattr(model_object, arg)
    ]

    for k, data in model_object.state_dict().items():
        f.create_dataset(
            prefix + k,
            data=data.numpy()
        )

    for s_arg in _serialize_args:

        if getattr(model_object, s_arg) is not None:

            if s_arg in _SERIALIZE_ENCODED_ARGS:
                _d = _ENCODE_ACTIVATIONS[getattr(model_object, s_arg)]
            else:
                _d = getattr(model_object, s_arg)

            f.create_dataset(
                prefix + s_arg,
                data=np.array(_d)
            )

    f.create_dataset(
        prefix + 'keys',
        data=np.array(
            list(model_object.state_dict().keys()),
            dtype=object
        )
    )

    f.create_dataset(
        prefix + 'args',
        data=np.array(
            _serialize_args,
            dtype=object
        )
    )

    f.create_dataset(
        prefix + 'type_name',
        data=model_object.type_name
    )
