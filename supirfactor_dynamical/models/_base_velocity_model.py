import torch


class _VelocityMixin:

    _velocity_model = True

    def input_data(self, x, **kwargs):
        return super().input_data(x[..., 0], **kwargs)

    def output_data(self, x, keep_all_dims=False, **kwargs):

        if keep_all_dims:
            return super().output_data(x, **kwargs)
        else:
            return super().output_data(x[..., 1], **kwargs)

    def scale_count_to_velocity(
        self,
        count
    ):
        if self.inv_scaler is not None:
            return torch.matmul(count, self.scaler)
        else:
            return count

    def scale_velocity_to_count(
        self,
        velocity
    ):
        if self.scaler is not None:
            return torch.matmul(velocity, self.inv_scaler)
        else:
            return velocity

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
