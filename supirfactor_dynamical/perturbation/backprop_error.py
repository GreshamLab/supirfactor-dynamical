import torch

from .predict import predict_perturbation


def perturbation_tfa_gradient(
    model,
    input_data,
    observed_data,
    perturbation=None,
    observed_data_delta_t=0,
    loss_function=torch.nn.MSELoss()
):

    _grads = []
    _handle = model._transcription_model.encoder[1].register_backward_hook(
        lambda m, gi, go: _grads.append(torch.clone(go[0]))
    )

    predicted_data = predict_perturbation(
        model,
        input_data,
        perturbation,
        observed_data_delta_t
    )[1]

    if observed_data_delta_t is not None:
        predicted_data = predicted_data[:, [-1], :]

    loss = loss_function(
        predicted_data,
        observed_data
    )

    loss.backward()

    _handle.remove()

    return loss, _grads[0]
