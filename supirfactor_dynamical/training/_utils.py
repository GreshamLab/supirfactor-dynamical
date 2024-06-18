def _set_submodels(
    model,
    encoder=None,
    intermediate=None,
    decoder=None
):

    if encoder is not None:
        model.select_submodel(
            encoder,
            model_type='encoder'
        )

    if intermediate is not None:
        model.select_submodel(
            intermediate,
            model_type='intermediate'
        )

    if decoder is not None:
        model.select_submodel(
            decoder,
            model_type='decoder'
        )
