import torch


def basic_classifier(input_width, hl_width, output_classes):

    return torch.nn.Sequential(
        torch.nn.Linear(input_width, hl_width, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(hl_width, hl_width, bias=False),
        torch.nn.Tanh(),
        torch.nn.Linear(hl_width, output_classes, bias=False),
        torch.nn.Sigmoid()
    )
