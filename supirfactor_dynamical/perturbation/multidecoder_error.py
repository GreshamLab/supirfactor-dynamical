import torch


def decoder_loss_transfer(
    model,
    input_x,
    output_x,
    loss_decoder,
    new_output_decoder,
    loss_function=torch.nn.MSELoss(),
    encoder=None,
    intermediate=None,
    n_iterations=10,
    optimizer=None
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

    _embedding = torch.nn.Parameter(
        model.latent_embedding(input_x)
    )

    # Add a model hook to get gradients at the last layer before decoder
    optimizer = model.process_optimizer(optimizer, params=(_embedding,))

    for _ in range(n_iterations):
        loss_function(model._decoder(_embedding), output_x).backward()

        optimizer.step()
        optimizer.zero_grad()

    model.select_submodels(decoder=new_output_decoder)

    with torch.no_grad():
        return _embedding.detach(), model._decoder(_embedding)
