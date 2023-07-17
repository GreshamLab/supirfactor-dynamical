import torch


class _VelocityMixin:

    _velocity_model = True

    _velocity_inverse_scaler = None
    _count_inverse_scaler = None
    scaler = None

    def set_scaling(
        self,
        count_scaling=False,
        velocity_scaling=False
    ):
        """
        If count or velocity is scaled, fix the scaling so that
        they match.

        Needed to calculate t+1 from count and velocity at t

        x_(t+1) = x(t) + dx/dx * (count_scaling / velocity_scaling)

        :param count_scaling: Count scaler [G] vector,
            None disables count scaling.
        :type count_scaling: torch.Tensor, np.ndarray, optional
        :param velocity_scaling: Velocity scaler [G] vector,
            None disables velocity scaling
        :type velocity_scaling: torch.Tensor, np.ndarray, optional
        """

        if count_scaling is None:
            self._count_inverse_scaler = None

        elif count_scaling is not False:
            self._count_inverse_scaler = self.to_tensor(count_scaling)

        if velocity_scaling is None:
            self._velocity_inverse_scaler = None

        elif velocity_scaling is not False:
            self._velocity_inverse_scaler = self.to_tensor(velocity_scaling)

        _s1 = self._count_inverse_scaler is not None
        _s2 = self._velocity_inverse_scaler is not None

        if _s1 and _s2:
            _scaler = torch.div(
                self._count_inverse_scaler,
                self._velocity_inverse_scaler
            )

            _scaler[self._velocity_inverse_scaler == 0] = 0

            self.scaler = torch.diag(_scaler)

        elif _s1:
            self.scaler = torch.diag(self._count_inverse_scaler)
        elif _s2:
            self.scaler = torch.diag(self._velocity_inverse_scaler)
        else:
            self.scaler = None

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

        if self.scaler is not None:
            velocity = torch.matmul(velocity, self.scaler)

        return torch.add(counts, velocity)
