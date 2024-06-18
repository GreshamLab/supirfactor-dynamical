import torch
from supirfactor_dynamical._utils import (
    to_tensor_device,
    _to_tensor
)


class _ScalingMixin:

    _count_inverse_scaler = None
    _count_rescale_scaler = None
    _velocity_inverse_scaler = None
    _velocity_rescale_scaler = None

    _count_to_velocity_scaler = None
    _velocity_to_count_scaler = None

    @property
    def count_scaler(self):
        if self._count_inverse_scaler is not None:
            return self._count_inverse_scaler
        else:
            return None

    @property
    def count_rescaler(self):
        if self._count_rescale_scaler is not None:
            return self._count_rescale_scaler
        else:
            return None

    @property
    def velocity_scaler(self):
        if self._velocity_inverse_scaler is not None:
            return self._velocity_inverse_scaler
        else:
            return None

    @property
    def velocity_rescaler(self):
        if self._velocity_rescale_scaler is not None:
            return self._velocity_rescale_scaler
        else:
            return None

    @property
    def count_to_velocity_scaler(self):
        if self._count_to_velocity_scaler is not None:
            return self._count_to_velocity_scaler
        else:
            return None

    @property
    def velocity_to_count_scaler(self):
        if self._velocity_to_count_scaler is not None:
            return self._velocity_to_count_scaler
        else:
            return None

    def set_scaling(
        self,
        count_scaling=False,
        velocity_scaling=False
    ):
        """
        If count or velocity is scaled, fix the scaling so that
        they match.

        Scale is defined as x_scaled = x / scale_factor

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
            self._count_inverse_scaler = _to_tensor(count_scaling)
            self._count_rescale_scaler = self._zero_safe_div(
                None,
                self._count_inverse_scaler
            )

        if velocity_scaling is None:
            self._velocity_inverse_scaler = None

        elif velocity_scaling is not False:
            self._velocity_inverse_scaler = _to_tensor(velocity_scaling)
            self._velocity_rescale_scaler = self._zero_safe_div(
                None,
                self._velocity_inverse_scaler
            )

        self._count_to_velocity_scaler = self._zero_safe_div(
            self._count_inverse_scaler,
            self._velocity_inverse_scaler
        )

        self._velocity_to_count_scaler = self._zero_safe_div(
            self._velocity_inverse_scaler,
            self._count_inverse_scaler
        )

        return self

    def unscale_counts(self, x):
        """
        Take scaled counts and remove scaling
        """
        if self._count_inverse_scaler is not None:
            return torch.mul(
                x,
                to_tensor_device(
                    self.count_scaler,
                    x
                )[..., :]
            )
        else:
            return x

    def unscale_velocity(self, x):
        """
        Take scaled velocity and remove scaling
        """
        if self._velocity_inverse_scaler is not None:
            return torch.mul(
                x,
                to_tensor_device(
                    self.velocity_scaler,
                    x
                )[..., :]
            )
        else:
            return x

    def rescale_velocity(self, x):
        """
        Take unscaled velocity and scale it
        """
        if self._velocity_inverse_scaler is not None:
            return torch.mul(
                x,
                to_tensor_device(
                    self.velocity_rescaler,
                    x
                )[..., :]
            )
        else:
            return x

    def rescale_counts(self, x):
        """
        Take unscaled counts and scale them
        """
        if self._count_inverse_scaler is not None:
            return torch.mul(
                x,
                to_tensor_device(
                    self.count_rescaler,
                    x
                )[..., :]
            )
        else:
            return x

    def scale_count_to_velocity(
        self,
        count
    ):
        """
        Take scaled counts and modifiy them so that they
        are scaled to velocity
        """
        if self._count_to_velocity_scaler is not None:
            return torch.mul(
                count,
                to_tensor_device(
                    self.count_to_velocity_scaler,
                    count
                )[..., :]
            )
        else:
            return count

    def scale_velocity_to_count(
        self,
        velocity
    ):
        """
        Take scaled velocity and modify it so that its
        scaled to counts
        """
        if self._velocity_to_count_scaler is not None:
            return torch.mul(
                velocity,
                to_tensor_device(
                    self.velocity_to_count_scaler,
                    velocity
                )[..., :]
            )
        else:
            return velocity

    @staticmethod
    def _zero_safe_div(x, y):
        """
        Return z = x / y
        z = 1 for y == 0
        Allow for Nones
        """

        if x is None and y is None:
            return None

        elif x is None:
            _z = 1 / y
            _z[y == 0] = 1

        elif y is None:
            _z = x

        else:
            _z = torch.div(x, y)
            _z[y == 0] = 1

        return _z
