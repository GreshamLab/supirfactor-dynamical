import torch
from torch import Tensor


class ConsistentDropout(torch.nn.Dropout):

    _dropout = None

    def reset(self):
        self._dropout = None

    def _set_dropout(self, x):
        self._dropout = torch.diag(
            super().forward(
                torch.ones(x.shape[-1])
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        if self._dropout is None:
            self._set_dropout(input)

        return torch.matmul(
            input,
            self._dropout
        )
