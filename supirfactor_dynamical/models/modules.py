import torch


def basic_classifier(
    input_width,
    hl_width,
    output_classes,
    dropout=0.0
):

    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(input_width, hl_width, bias=False),
        torch.nn.Tanh(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hl_width, output_classes, bias=False)
    )


def basic_count_predictor(
    input_width,
    widths,
    output_width,
    dropout=0.0,
    softplus=False
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
        torch.nn.Softplus() if softplus else torch.nn.ReLU()
    )
