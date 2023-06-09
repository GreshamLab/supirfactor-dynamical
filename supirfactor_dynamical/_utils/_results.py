from inferelator.postprocessing import ResultsProcessor


def evaluate_results(
    model_weights,
    model_erv,
    prior_network,
    gold_standard_network
):

    return ResultsProcessor(
        [model_weights],
        [model_erv],
        metric="combined"
    ).summarize_network(
        None,
        gold_standard_network,
        prior_network,
        full_model_betas=model_weights
    )
