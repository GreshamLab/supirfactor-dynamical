import torch


class _VelocityMixin:

    _velocity_model = True

    _velocity_inverse_scaler = None
    _count_inverse_scaler = None
    _count_scaler = None

    def set_scaling(
        self,
        count_scaling=False,
        velocity_scaling=False
    ):
        """
        If count or velocity is scaled, fix the scaling so that
        they match.

        Needed to calculate t+1 from count and velocity at t

        :param count_scaling: Count scaler [G] vector,
            None disables count scaling.
        :type count_scaling: torch.Tensor, np.ndarray, optional
        :param velocity_scaling: Velocity scaler [G] vector,
            None disables velocity scaling
        :type velocity_scaling: torch.Tensor, np.ndarray, optional
        """

        if count_scaling is None:
            self._count_inverse_scaler = None
            self._count_scaler = None

        elif count_scaling is not False:
            self._count_inverse_scaler = torch.diag(
                self.to_tensor(count_scaling)
            )

            _count_scaler = torch.divide(1, self.to_tensor(count_scaling))
            _count_scaler[count_scaling == 0] = 1

            self._count_scaler = torch.diag(
                _count_scaler
            )

        if velocity_scaling is None:
            self._velocity_inverse_scaler = None

        elif velocity_scaling is not False:
            self._velocity_inverse_scaler = torch.diag(
                self.to_tensor(velocity_scaling)
            )

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

    def next_count_from_velocity(
        self,
        counts,
        velocity
    ):

        if self._count_inverse_scaler is not None:
            counts = torch.matmul(counts, self._count_inverse_scaler)

        if self._velocity_inverse_scaler is not None:
            velocity = torch.matmul(counts, self._velocity_inverse_scaler)

        counts = torch.add(counts, velocity)

        if self._count_scaler is not None:
            return torch.matmul(counts, self._count_scaler)

        else:
            return counts
