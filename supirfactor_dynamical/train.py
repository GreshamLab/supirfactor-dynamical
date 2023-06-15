from .models import (
    _CLASS_DICT,
    TFAutoencoder,
    TFRNNDecoder
)

from .models._base_model import (
    _TFMixin
)

from . import evaluate_results


def _is_model(model):

    try:
        return issubclass(model, _TFMixin)
    except TypeError:
        return False


def dynamic_model_training(
    dynamic_training_dataloader,
    prior_network,
    epochs,
    dynamic_validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    model_type=None,
    prediction_length=None,
    prediction_loss_offset=None,
    **kwargs
):

    # Get model type
    if model_type is None:
        dynamic_autoencoder = TFRNNDecoder

    elif _is_model(model_type):
        dynamic_autoencoder = model_type

    else:
        dynamic_autoencoder = _CLASS_DICT[model_type]

    ae_dynamic = dynamic_autoencoder(
        prior_network,
        **kwargs
    )

    ae_dynamic.set_time_parameters(
        output_t_plus_one=prediction_length is not None,
        n_additional_predictions=prediction_length,
        loss_offset=prediction_loss_offset
    )

    ae_dynamic.train_model(
        dynamic_training_dataloader,
        epochs,
        validation_dataloader=dynamic_validation_dataloader,
        optimizer=optimizer_params
    )

    ae_dynamic.eval()

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


def static_model_training(
    static_training_dataloader,
    prior_network,
    epochs,
    static_validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    model_type=None,
    prediction_length=None,
    prediction_loss_offset=None,
    **kwargs
):

    # Get model type
    if model_type is None:
        static_autoencoder = TFAutoencoder

    elif _is_model(model_type):
        static_autoencoder = model_type

    else:
        static_autoencoder = _CLASS_DICT[model_type]

    ae_static = static_autoencoder(
        prior_network,
        **kwargs
    )

    ae_static.set_time_parameters(
        output_t_plus_one=prediction_length is not None,
        n_additional_predictions=prediction_length,
        loss_offset=prediction_loss_offset
    )

    ae_static.train_model(
        static_training_dataloader,
        epochs,
        validation_dataloader=static_validation_dataloader,
        optimizer=optimizer_params
    )

    ae_static.eval()

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
    prediction_length=None,
    prediction_loss_offset=None,
    static_model_type=None,
    dynamic_model_type=None,
    **kwargs
):

    ae_static, ae_results = static_model_training(
        static_training_dataloader,
        prior_network,
        epochs,
        static_validation_dataloader=static_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        prediction_length=prediction_length,
        prediction_loss_offset=prediction_loss_offset,
        model_type=static_model_type,
        **kwargs
    )

    ae_dynamic, dyn_results = dynamic_model_training(
        dynamic_training_dataloader,
        prior_network,
        epochs,
        dynamic_validation_dataloader=dynamic_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        prediction_length=prediction_length,
        prediction_loss_offset=prediction_loss_offset,
        model_type=dynamic_model_type,
        **kwargs
    )

    return ae_static, ae_results, ae_dynamic, dyn_results
