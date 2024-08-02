import torch
from supirfactor_dynamical._utils import get_activation_function


def basic_classifier(
    input_width,
    hl_width,
    output_classes,
    dropout=0.0,
    activation=None
):

    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(input_width, hl_width, bias=False),
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hl_width, output_classes, bias=False),
        get_activation_function(activation)
    )


def basic_count_predictor(
    input_width,
    widths,
    output_width,
    dropout=0.0,
    activation='ReLU'
):

    if not isinstance(widths, (tuple, list)):
        widths = [widths, widths]

    return torch.nn.Sequential(
        torch.nn.Linear(input_width, widths[0], bias=False),
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(widths[0], widths[1], bias=False),
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(widths[1], output_width, bias=False),
        get_activation_function(activation)
    )
