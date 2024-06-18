import torch


class _VelocityMixin:

    _velocity_model = True

    def input_data(self, x, **kwargs):
        return super().input_data(x[..., 0], **kwargs)

    def output_data(
        self,
        x,
        keep_all_dims=False,
        counts=False,
        decay=False,
        **kwargs
    ):

        if keep_all_dims:
            return super().output_data(x, **kwargs)
        elif counts:
            return super().output_data(x[..., 0], **kwargs)
        elif decay:
            return super().output_data(x[..., 2], **kwargs)
        else:
            return super().output_data(x[..., 1], **kwargs)

    def next_count_from_velocity(
        self,
        counts,
        velocity
    ):

        return torch.nn.ReLU()(
            torch.add(
                counts,
                self.scale_velocity_to_count(velocity)
            )
        )
