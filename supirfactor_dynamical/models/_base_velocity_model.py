class _VelocityMixin:

    _velocity_model = True

    def input_data(self, x, **kwargs):

        if x.shape[-1] == 2:
            return super().input_data(x[..., 0], **kwargs)
        else:
            return (
                super().input_data(x[..., 0], **kwargs),
                super().input_data(x[..., 2], **kwargs)
            )

    def output_data(self, x, keep_all_dims=False, **kwargs):

        if keep_all_dims:
            return super().output_data(x, **kwargs)
        else:
            return super().output_data(x[..., 1], **kwargs)
