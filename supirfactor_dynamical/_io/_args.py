_SERIALIZE_ARGS = [
    'input_dropout_rate',
    'intermediate_dropout_rate',
    'hidden_dropout_rate',
    'time_dependent_decay',
    'decay_k',
    'decay_epoch_delay',
    'n_genes',
    'hidden_layer_width',
    'n_peaks',
    'output_activation',
    'tfa_activation',
    'activation',
    'intermediate_sizes',
    'decoder_sizes'
]

_SERIALIZE_TIME_ARGS = [
    'output_t_plus_one',
    'n_additional_predictions',
    'loss_offset'
]

_SERIALIZE_MODEL_TYPE_ATTRS = [
    '_velocity_model',
    '_multisubmodel_model'
]

_SERIALIZE_RUNTIME_ATTRS = [
    '_training_loss',
    '_validation_loss',
    'training_time',
    'training_r2',
    'validation_r2',
    'training_r2_over_time',
    'validation_r2_over_time',
    '_training_n',
    '_validation_n'
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

_SCALER_ARGS = [
    ('_count_inverse_scaler', 'count_scaling'),
    ('_velocity_inverse_scaler', 'velocity_scaling')
]
