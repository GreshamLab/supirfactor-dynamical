from .models import (
    _CLASS_DICT,
    TFAutoencoder,
    TFRecurrentAutoencoder
)

from .models._base_model import (
    _TFMixin
)

from .models._utils import evaluate_results


def dynamic_model_training(
    dynamic_training_dataloader,
    prior_network,
    epochs,
    dynamic_validation_dataloader=None,
    decoder_weights=None,
    optimizer_params=None,
    gold_standard=None,
    input_dropout=0.5,
    hidden_layer_dropout=0.0,
    prediction_time_offset=None,
    recurrency_mask=None,
    model_type=None
):

    # Get model type
    if model_type is None:
        dynamic_autoencoder = TFRecurrentAutoencoder

    elif _is_model(model_type):
        dynamic_autoencoder = model_type

    else:
        dynamic_autoencoder = _CLASS_DICT[model_type]

    ae_dynamic = dynamic_autoencoder(
        prior_network,
        recurrency_mask=recurrency_mask,
        input_dropout_rate=input_dropout,
        layer_dropout_rate=hidden_layer_dropout,
        decoder_weights=decoder_weights,
        prediction_offset=prediction_time_offset
    )

    ae_dynamic.train_model(
        dynamic_training_dataloader,
        epochs,
        validation_dataloader=dynamic_validation_dataloader,
        optimizer=optimizer_params
    )

    ae_dynamic.r2_over_time(
        dynamic_training_dataloader,
        dynamic_validation_dataloader
    )

    dyn_results = evaluate_results(
        ae_dynamic.output_weights(
            as_dataframe=True
        ),
        ae_dynamic.erv(
            dynamic_training_dataloader,
            as_data_frame=True
        ),
        prior_network,
        gold_standard if gold_standard is not None else prior_network
    )

    return ae_dynamic, dyn_results


def _is_model(model):

    try:
        return issubclass(model, _TFMixin)
    except TypeError:
        return False


def static_model_training(
    static_training_dataloader,
    prior_network,
    epochs,
    static_validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    input_dropout=0.5,
    hidden_layer_dropout=0.0,
    prediction_time_offset=None,
):

    ae_static = TFAutoencoder(
        prior_network,
        input_dropout_rate=input_dropout,
        layer_dropout_rate=hidden_layer_dropout,
        prediction_offset=prediction_time_offset
    )

    ae_static.train_model(
        static_training_dataloader,
        epochs,
        validation_dataloader=static_validation_dataloader,
        optimizer=optimizer_params
    )

    ae_results = evaluate_results(
        ae_static.output_weights(
            as_dataframe=True
        ),
        ae_static.erv(
            static_training_dataloader,
            as_data_frame=True
        ),
        prior_network,
        gold_standard if gold_standard is not None else prior_network
    )

    return ae_static, ae_results


def joint_model_training(
    static_training_dataloader,
    dynamic_training_dataloader,
    prior_network,
    epochs,
    static_validation_dataloader=None,
    dynamic_validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    input_dropout=0.5,
    hidden_layer_dropout=0.0,
    prediction_time_offset=None,
    dynamic_recurrency_mask=None,
    dynamic_model_type=None
):

    ae_static, ae_results = static_model_training(
        static_training_dataloader,
        prior_network,
        epochs,
        static_validation_dataloader=static_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        input_dropout=input_dropout,
        hidden_layer_dropout=hidden_layer_dropout,
        prediction_time_offset=prediction_time_offset,
    )

    ae_dynamic, dyn_results = dynamic_model_training(
        dynamic_training_dataloader,
        prior_network,
        epochs,
        dynamic_validation_dataloader=dynamic_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        input_dropout=input_dropout,
        hidden_layer_dropout=hidden_layer_dropout,
        prediction_time_offset=prediction_time_offset,
        recurrency_mask=dynamic_recurrency_mask,
        model_type=dynamic_model_type
    )

    return ae_static, ae_results, ae_dynamic, dyn_results


def highly_connected_dynamic_hidden_layer(
    dynamic_training_dataloader,
    dynamic_validation_dataloader,
    prior_network,
    epochs,
    optimizer_params=None,
    gold_standard=None,
    input_dropout=0.5,
    hidden_layer_dropout=0.0,
    erv_threshold=0.05,
    prediction_time_offset=None,
    dynamic_model_type=None
):

    ae_dynamic, dyn_results_first = dynamic_model_training(
        dynamic_training_dataloader,
        prior_network,
        epochs,
        dynamic_validation_dataloader=dynamic_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        input_dropout=input_dropout,
        hidden_layer_dropout=hidden_layer_dropout,
        prediction_time_offset=prediction_time_offset,
        recurrency_mask=None,
        model_type=dynamic_model_type
    )

    dynamic_erv = ae_dynamic.erv(
        dynamic_training_dataloader,
        as_data_frame=True
    )

    initialize_decoder = ae_dynamic.pruned_model_weights(
        erv=dynamic_erv,
        erv_threshold=erv_threshold,
        as_dataframe=True
    )

    ae_dynamic_second, dyn_results_second = dynamic_model_training(
        dynamic_training_dataloader,
        prior_network,
        epochs,
        dynamic_validation_dataloader=dynamic_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        input_dropout=input_dropout,
        hidden_layer_dropout=hidden_layer_dropout,
        prediction_time_offset=prediction_time_offset,
        recurrency_mask=False,
        decoder_weights=initialize_decoder,
        model_type=dynamic_model_type
    )

    return ae_dynamic, dyn_results_first, ae_dynamic_second, dyn_results_second
