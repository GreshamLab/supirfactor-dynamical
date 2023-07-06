import h5py
import numpy as np
import pandas as pd

_SERIALIZE_ARGS = [
    'input_dropout_rate',
    'hidden_dropout_rate',
    'output_relu',
    'output_t_plus_one',
    'n_additional_predictions',
    'loss_offset',
    '_velocity_model',
    '_decay_model'
]


def write(
    model_object,
    file_name
):

    _serialize_args = [
        arg
        for arg in _SERIALIZE_ARGS
        if hasattr(model_object, arg)
    ]

    with h5py.File(file_name, 'w') as f:
        for k, data in model_object.state_dict().items():
            f.create_dataset(
                k,
                data=data.numpy()
            )

        for s_arg in _serialize_args:

            if getattr(model_object, s_arg) is not None:
                f.create_dataset(
                    s_arg,
                    data=getattr(model_object, s_arg)
                )

        f.create_dataset(
            'keys',
            data=np.array(
                list(model_object.state_dict().keys()),
                dtype=object
            )
        )

        f.create_dataset(
            'args',
            data=np.array(
                _serialize_args,
                dtype=object
            )
        )

        f.create_dataset(
            'type_name',
            data=model_object.type_name
        )

    with pd.HDFStore(file_name, mode="a") as f:
        model_object._to_dataframe(
            model_object.prior_network,
            transpose=True
        ).to_hdf(
            f,
            'prior_network'
        )
