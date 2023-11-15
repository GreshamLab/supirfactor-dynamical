def _check_data_offsets(
    L,
    output_t_plus_one=False,
    n_additional_predictions=0,
    loss_offset=0
):

    in_offset, out_offset = _get_data_offsets(
        L,
        output_t_plus_one,
        n_additional_predictions,
        loss_offset
    )

    if in_offset < 1 or out_offset >= L:
        raise ValueError(
            f"Cannot train on {L} sequence length with "
            f"{n_additional_predictions} additional predictions and "
            f"{loss_offset} values excluded from loss"
        )

    return in_offset, out_offset


def _get_data_offsets(
    L,
    output_t_plus_one=False,
    n_additional_predictions=0,
    loss_offset=0
):
    """
    Returns slice indices for input (O:input_offset) and
    for output (output_offset:L) based on slice parameters
    """

    if loss_offset is None:
        loss_offset = 0

    if n_additional_predictions is None:
        n_additional_predictions = 0

    input_offset = L - n_additional_predictions
    output_offset = loss_offset

    if output_t_plus_one:
        input_offset -= 1
        output_offset += 1

    return input_offset, output_offset
