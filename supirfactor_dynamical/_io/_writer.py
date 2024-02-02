import h5py
import numpy as np

from ._network import write_network


_SERIALIZE_ARGS = [
    'input_dropout_rate',
    'intermediate_dropout_rate',
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
    'n_genes',
    'hidden_layer_width',
    'n_peaks',
    'output_activation',
    'tfa_activation',
    'activation',
    'intermediate_sizes',
    'decoder_sizes'
]

_SERIALIZE_ENCODED_ARGS = [
    'output_activation',
    'activation',
    'tfa_activation'
]

_SERIALIZE_NETWORKS = [
    'prior_network',
    'peak_tf_prior_network',
    'gene_peak_mask'
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
