import torch


def predict_perturbation(
    model,
    data,
    perturbation,
    n_time_steps,
    return_submodels=False,
    unmodified_counts=False
):
    """
    Predict transcriptional perturbation by
    inferring decay rates from unperturbed model,
    and then re-predicting transcription from
    perturbed model, using the unperturbed decay

    :param model: Fit model
    :type model: SupirFactorBiophysical
    :param data: Input data for prediction
    :type data: torch.Tensor
    :param perturbation: TF(s) to perturb
    :type perturbation: str, list(str)
    :param n_time_steps: Number of time steps to predict
    :type n_time_steps: int
    :param return_submodels: Return positive and negative velocities
        separately, defaults to False
    :type return_submodels: bool, optional
    :param unmodified_counts: Return input data counts without modification,
        defaults to False
    :type unmodified_counts: bool, optional
    :return: Predicted expression data
    :rtype: torch.Tensor
    """

    model.eval().set_drop_tfs(None)

    with torch.no_grad():
        # Unperturbed decay
        decay_rates = model(
            data,
            n_time_steps=n_time_steps,
            unmodified_counts=unmodified_counts
        )[2]

    model.set_drop_tfs(perturbation)

    result = model._perturbed_model_forward(
        data,
        decay_rates,
        n_time_steps=n_time_steps,
        unmodified_counts=unmodified_counts,
        return_submodels=return_submodels
    )

    model.set_drop_tfs(None)

    return result
