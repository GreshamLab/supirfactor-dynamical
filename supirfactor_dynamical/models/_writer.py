import h5py
import numpy as np
import pandas as pd

_SERIALIZE_ARGS = [
    'input_dropout_rate',
    'hidden_dropout_rate',
    'output_t_plus_one',
    'n_additional_predictions',
    'loss_offset',
    '_velocity_model',
    'time_dependent_decay',
    'decay_k',
    'decay_epoch_delay',
    '_velocity_inverse_scaler',
    '_count_inverse_scaler',
    '_training_loss',
    '_validation_loss',
    'training_time',
    'training_r2',
    'validation_r2',
    'g',
    'k',
    'output_activation',
    'activation'
]

_SERIALIZE_ENCODED_ARGS = [
    'output_activation',
    'activation'
]


_ENCODE_ACTIVATIONS = {
    None: 0,
    'relu': 1,
    'softplus': 2,
    'sigmoid': 3,
    'tanh': 4
}


def write(
    model_object,
    file_name,
    mode='w',
    prefix=''
):

    with h5py.File(file_name, mode) as f:
        _write_state(f, model_object, prefix)

    if hasattr(model_object, 'prior_network'):
        _write_df(
            file_name,
            model_object._to_dataframe(
                model_object.prior_network,
                transpose=True
            ),
            'prior_network'
        )


def _write_df(
    file_name,
    df,
    key
):

    with pd.HDFStore(file_name, mode="a") as f:
        df.to_hdf(
            f,
            key
        )


def _write_state(
    f,
    model_object,
    prefix=''
):

    _serialize_args = [
        arg
        for arg in _SERIALIZE_ARGS
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
