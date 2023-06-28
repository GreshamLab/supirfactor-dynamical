import torch
import numpy as np


class _VelocityMixin:

    _velocity_model = True

    def input_data(self, x, **kwargs):

        return super().input_data(x[..., 0], **kwargs)

    def output_data(self, x, keep_all_dims=False, **kwargs):

        if keep_all_dims:
            return super().output_data(x, **kwargs)
        else:
            return super().output_data(x[..., 1], **kwargs)


class _DecayMixin:

    _decay_model = True

    decay_hidden = None
    decay_invert = None

    decay_scalar_initial = None
    decay_tensor_initial = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initalize_decay_module(
        self,
        L
    ):

        _no_tensor = self.decay_tensor_initial is None
        _no_scalar = self.decay_scalar_initial is None

        if L is None:
            tensor_shape = (self.g, )
        else:
            tensor_shape = (L, self.g)

        # Fill with 15 minute half-life if no intials provided
        if _no_tensor and _no_scalar:
            self.decay_tensor_initial = torch.full(
                tensor_shape,
                np.log(2) / 15.
            )

        # Fill with a given scalar value to initialize
        elif _no_tensor:
            self.decay_tensor_initial = torch.full(
                tensor_shape,
                self.decay_scalar_initial
            )
        # Fill with a given tensor of values to initialize
        # Needs to match the sequence length correctly
        else:
            self.decay_tensor_initial = torch.clone(
                self.decay_tensor_initial
            )

        self.decay_parameter = torch.nn.parameter.Parameter(
            self.decay_tensor_initial
        )

        self.decay_invert = torch.diag(
            torch.full(
                (self.g, ),
                -1.0
            )
        )

    def forward_decay_model(
        self,
        x
    ):

        if not hasattr(self, 'decay_parameter'):
            if x.ndim == 2:
                self.initalize_decay_module(None)
            else:
                self.initalize_decay_module(x.shape[1])

        x = torch.mul(x, self.decay_parameter[None, :])
        x = torch.matmul(x, self.decay_invert)

        return x

    def forward_model(
        self,
        x,
        hidden_state=None
    ):

        x_output_positive = self.forward_tf_model(
            x,
            hidden_state=hidden_state
        )

        x_output_negative = self.forward_decay_model(
            x,
        )

        x_output = torch.add(
            x_output_positive,
            x_output_negative
        )

        return x_output
