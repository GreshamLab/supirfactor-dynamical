from .models import (
    _CLASS_DICT,
    TFAutoencoder,
    TFRNNDecoder,
    SupirFactorBiophysical
)

from .models._base_model import (
    _TFMixin
)

from . import evaluate_results

DROPOUT_KWARGS = [
    ('input_dropout_rate', 0.5),
    ('hidden_dropout_rate', 0.0)
]


def _is_model(model):

    try:
        return issubclass(model, _TFMixin)
    except TypeError:
        return False


def pretrain_and_tune_dynamic_model(
    pretraining_training_dataloader,
    prediction_tuning_training_dataloader,
    prior_network,
    epochs,
    optimizer_params=None,
    pretraining_validation_dataloader=None,
    prediction_tuning_validation_dataloader=None,
    prediction_length=None,
    prediction_loss_offset=None,
    model_type=None,
    return_erv=False,
    **kwargs
):

    pretrain_dropout, tune_dropout = _dropout_kwargs(
        kwargs
    )

    model, pretrain_results = dynamic_model_training(
        pretraining_training_dataloader,
        prior_network,
        epochs,
        dynamic_validation_dataloader=pretraining_validation_dataloader,
        optimizer_params=optimizer_params,
        prediction_length=False,
        model_type=model_type,
        **kwargs,
        **pretrain_dropout
    )

    model.train()

    model.set_dropouts(**tune_dropout)

    model, tuned_results, _final_erv = dynamic_model_training(
        prediction_tuning_training_dataloader,
        prior_network,
        epochs,
        dynamic_validation_dataloader=prediction_tuning_validation_dataloader,
        optimizer_params=optimizer_params,
        prediction_length=prediction_length,
        prediction_loss_offset=prediction_loss_offset,
        model_type=model,
        return_erv=True,
        **kwargs
    )

    if return_erv:
        return model, pretrain_results, tuned_results, _final_erv
    else:
        return model, pretrain_results, tuned_results


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
    return_erv=False,
    **kwargs
):

    # If model_type is an instance of a model
    # use it as-is
    if isinstance(model_type, _TFMixin):
        ae_dynamic = model_type

    # Otherwise get a model and instantiate it
    else:

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

    _weights = ae_dynamic.output_weights(
        as_dataframe=True
    )

    _erv = ae_dynamic.erv(
        dynamic_training_dataloader,
        as_data_frame=True
    )

    dyn_results = evaluate_results(
        _weights,
        _erv,
        prior_network,
        gold_standard if gold_standard is not None else prior_network
    )

    if return_erv:
        return ae_dynamic, dyn_results, _erv
    else:
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
    return_erv=False,
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

    _weights = ae_static.output_weights(
        as_dataframe=True
    )

    _erv = ae_static.erv(
        static_training_dataloader,
        as_data_frame=True
    )

    ae_results = evaluate_results(
        _weights,
        _erv,
        prior_network,
        gold_standard if gold_standard is not None else prior_network
    )

    if return_erv:
        return ae_static, ae_results, _erv
    else:
        return ae_static, ae_results


def biophysical_model_training(
    training_dataloader,
    prior_network,
    epochs,
    trained_count_model=None,
    validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    prediction_length=None,
    prediction_loss_offset=None,
    **kwargs
):

    biophysical_model = SupirFactorBiophysical(
        prior_network,
        trained_count_model=trained_count_model,
        **kwargs
    )

    biophysical_model.set_time_parameters(
        output_t_plus_one=prediction_length is not None,
        n_additional_predictions=prediction_length,
        loss_offset=prediction_loss_offset
    )

    biophysical_model.train_model(
        training_dataloader,
        epochs,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer_params
    )

    biophysical_model.eval()

    _weights = biophysical_model.output_weights(
        as_dataframe=True
    )

    _erv = biophysical_model.erv(
        training_dataloader,
        as_data_frame=True
    )

    _results = evaluate_results(
        _weights,
        _erv,
        prior_network,
        gold_standard if gold_standard is not None else prior_network
    )

    return biophysical_model, _results, _erv


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


def _dropout_kwargs(kwargs):

    dropout_kwargs = [
        {
            k: v if not isinstance(v, tuple) else v[i]
            for k, v in {
                k: kwargs.pop(k, v)
                for k, v in DROPOUT_KWARGS
            }.items()
        }
        for i in range(2)
    ]

    return dropout_kwargs[0], dropout_kwargs[1]
