import torch
import numpy as np


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

X_tensor = torch.Tensor(X)

V = np.diff(X, axis=0, prepend=np.zeros((1, 4), dtype=np.float32))
V_tensor = torch.tensor(V)

XV_tensor = torch.stack(
    (
        X_tensor,
        V_tensor
    ),
    dim=-1
)
