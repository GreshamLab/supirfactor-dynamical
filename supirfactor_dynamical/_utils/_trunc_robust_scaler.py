import numpy as np

from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler
)


class TruncRobustScaler(RobustScaler):

    def fit(self, X, y=None):
        super().fit(X, y)

        # Use StandardScaler to deal with sparse & dense easily
        _std_scale = StandardScaler(with_mean=False).fit(X)

        _post_robust_var = _std_scale.var_ / (self.scale_ ** 2)
        _rescale_idx = _post_robust_var > 1

        _scale_mod = np.ones_like(self.scale_)
        _scale_mod[_rescale_idx] = np.sqrt(_post_robust_var[_rescale_idx])

        self.scale_ *= _scale_mod

        return self
