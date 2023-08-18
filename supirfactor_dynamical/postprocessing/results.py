import pandas as pd
import numpy as np

from inferelator.postprocessing import ResultsProcessor

_LOSS_COLS = ["Model_Type", "Loss_Type"]
_RESULT_COLS = ["R2_training", "R2_validation"]


def evaluate_results(
    model_weights,
    model_erv,
    prior_network,
    gold_standard_network
):

    result = ResultsProcessor(
        [model_weights],
        [model_erv],
        metric="combined"
    ).summarize_network(
        None,
        gold_standard_network,
        prior_network,
        full_model_betas=model_weights
    )

    result.model_file_name = None

    return result


def process_results_to_dataframes(
    model_object=None,
    results_object=None,
    model_type=None,
    leader_columns=None,
    leader_values=None
):

    if model_object is None and results_object is None:
        return None, None, None

    if model_object:
        _n = model_object.training_loss_df.shape[0]
    else:
        _n = 1

    if leader_values is None:
        leader_values = []
        leader_columns = []

    if results_object is None:
        score_cols = []
    else:
        score_cols = results_object.all_names

    loss_leaders = {
        k: pd.concat(
            [
                pd.DataFrame(
                    [leader_values + [model_type, k]],
                    columns=leader_columns + _LOSS_COLS
                )
            ] * _n,
            ignore_index=True
        )
        for k in ['training', 'validation']
    }

    if model_object is not None:

        result_line = [model_type] + [
            results_object.all_scores[n]
            for n in score_cols
        ] + [
            model_object.training_r2,
            model_object.validation_r2
        ]

        loss_df = pd.concat([
            pd.concat(
                (loss_leaders[k], o),
                axis=1
            ) for k, o in zip(
                [
                    'training',
                    'validation'
                ],
                [
                    model_object.training_loss_df,
                    model_object.validation_loss_df
                ]
            ) if o is not None
        ])

    else:
        result_line = [model_type] + [
            results_object.all_scores[n]
            for n in score_cols
        ] + [
            None,
            None
        ]

        loss_df = None

    results = pd.DataFrame(
        [leader_values + result_line],
        columns=leader_columns + ["Model_Type"] + score_cols + _RESULT_COLS
    )

    if (
        model_object is not None and
        hasattr(model_object, "training_r2_over_time")
    ):

        time_dependent_loss = pd.concat([
            pd.concat(
                (
                    loss_leaders[k].iloc[[0], :],
                    pd.DataFrame(
                        o,
                        columns=np.arange(1, len(o) + 1)
                    )
                ),
                axis=1
            ) for k, o in zip(
                [
                    'training',
                    'validation'
                ],
                [
                    model_object.training_r2_over_time,
                    model_object.validation_r2_over_time
                ]
            ) if o is not None
        ])

    else:
        time_dependent_loss = None

    return results, loss_df, time_dependent_loss


def process_combined_results(
    results,
    gold_standard=None,
    prior_network=None,
    model_type=None,
    leader_columns=None,
    leader_values=None
):

    _combined_weights = _combine_weights(
        *[r[0].betas[0] for r in results]
    )

    _combined_ervs = results[0][1].copy()

    for r in results[1:]:
        np.maximum(
            _combined_ervs,
            r[1],
            out=_combined_ervs
        )

    r, _, _ = process_results_to_dataframes(
        results_object=ResultsProcessor(
            [_combined_weights],
            [_combined_ervs],
            metric="combined"
        ).summarize_network(
            None,
            gold_standard,
            prior_network,
            full_model_betas=None
        ),
        leader_columns=leader_columns,
        leader_values=leader_values,
        model_type=model_type,
    )

    return r


def _combine_weights(*args):

    weights = args[0].copy()

    for a in args[1:]:
        weights += a

    weights /= len(args)

    return weights
