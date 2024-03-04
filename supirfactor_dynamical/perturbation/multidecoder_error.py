import torch


def decoder_loss_transfer(
    model,
    input_x,
    output_x,
    loss_decoder,
    new_output_decoder,
    loss_function=torch.nn.MSELoss(),
    encoder=None,
    intermediate=None
):

    if not model._multisubmodel_model:
        raise RuntimeError(
            "Unable to transfer loss to a new decoder; "
            "only one decoder available"
        )

    model.select_submodels(
        encoder=encoder,
        intermediate=intermediate,
        decoder=loss_decoder
    )

    # Add a model hook to get gradients at the last layer before decoder
    _grads = []
    _handle = model._intermediate[-2].register_backward_hook(
        lambda m, gi, go: _grads.append(torch.clone(go[0]))
    )

    loss_function(model(input_x), output_x).backward()

    _handle.remove()

    with torch.no_grad():
        _new_embedding = torch.mul(
            _grads[0],
            model.latent_embedding(input_x)
        )

        model.select_submodels(decoder=new_output_decoder)

        return _new_embedding, model._decoder(_new_embedding)
