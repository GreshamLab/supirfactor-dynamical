import torch
import numpy as np
from scipy.sparse import csr_matrix

from supirfactor_dynamical._utils._trunc_robust_scaler import TruncRobustScaler

_rng = np.random.default_rng(44)

# Generate 100, 4 test data
# and a 4, 3 linear map to 3 dimensions
A = np.array([
    [1, 0, 0],
    [0.5, 0.5, 0],
    [0.33, 0.33, 0.33],
    [0, 0, 0]
]).astype(np.float32)

X = np.abs(_rng.random((100, 4))).astype(np.float32)
X = X[np.argsort(X[:, 0]), :]
Y = X @ A
T = np.repeat(np.arange(4), 25)
X_SP = csr_matrix(X)

PEAKS = _rng.choice([0, 1], size=(100, 25), p=[0.95, 0.05])
PEAKS_SP = csr_matrix(PEAKS)

G_TO_PEAK_PRIOR = _rng.choice([0, 1], size=(4, 25), p=[0.75, 0.25])
PEAK_TO_TF_PRIOR = _rng.choice([0, 1], size=(25, 3), p=[0.75, 0.25])

X_tensor = torch.Tensor(X)

_V_BASE = np.diff(np.vstack((
    np.zeros(4),
    X[0:25, :].mean(axis=0),
    X[25:50, :].mean(axis=0),
    X[50:75, :].mean(axis=0),
    X[75:, :].mean(axis=0)
)), axis=0)

V = np.repeat(_V_BASE, 25, axis=0)
V += np.random.uniform(-0.1, 0.1, size=V.shape)
V = V.astype(np.float32)

V_tensor = torch.tensor(V)

XV_tensor = torch.stack(
    (
        torch.tensor(
            TruncRobustScaler(with_centering=False).fit_transform(X)
        ),
        torch.tensor(
            TruncRobustScaler(with_centering=False).fit_transform(V)
        )
    ),
    dim=-1
)

XVD_tensor = torch.stack(
    (
        torch.tensor(
            TruncRobustScaler(with_centering=False).fit_transform(X)
        ),
        torch.tensor(
            TruncRobustScaler(with_centering=False).fit_transform(V)
        ),
        torch.full_like(
            X_tensor,
            0.02
        )
    ),
    dim=-1
)

XTV_tensor = torch.clone(XV_tensor).reshape(25, 4, 4, 2)
XTVD_tensor = torch.clone(XVD_tensor).reshape(25, 4, 4, 3)
