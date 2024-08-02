import torch


def get_activation_function(
    activation,
    **kwargs
):
    if activation is None:
        return torch.nn.Identity()
    elif activation.lower() == 'sigmoid':
        return torch.nn.Sigmoid(**kwargs)
    elif activation.lower() == 'softplus':
        return torch.nn.Softplus(**kwargs)
    elif activation.lower() == 'relu':
        return torch.nn.ReLU(**kwargs)
    elif activation.lower() == 'tanh':
        return torch.nn.Tanh(**kwargs)
    elif activation.lower() == 'softmax':
        return torch.nn.Softmax(**kwargs)
    else:
        raise ValueError(
            f"Activation {activation} unknown"
        )
