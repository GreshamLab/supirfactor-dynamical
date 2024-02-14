import scipy.sparse as _spsparse
import h5py
import pandas as pd
import anndata as ad
import numpy as np

_scipy_objs = {
    'csr': (_spsparse.csr_matrix, _spsparse.csr_array),
    'csc': (_spsparse.csc_matrix, _spsparse.csc_array)
}


def sparse_type(x):

    for matrix_type, classes in _scipy_objs.items():

        if isinstance(x, classes):
            return classes[0], matrix_type

        if isinstance(x, str) and x.lower() == matrix_type:
            return classes[0], matrix_type

        if bytearray(x).decode().lower() == matrix_type:
            return classes[0], matrix_type

    else:
        raise ValueError(
            "Sparse matrix must be CSR or CSC, "
            f"BSR and COO is not supported; {x} provided"
        )


def write_network(
    file_name,
    data,
    key
):

    if isinstance(data, pd.DataFrame):
        return _write_df(file_name, data, key)

    elif isinstance(data, (np.ndarray, tuple)):
        return _write_df(file_name, pd.DataFrame(data), key)

    elif isinstance(data, ad.AnnData):
        _write_ad(file_name, data, key)

    else:
        raise ValueError(
            f"Network data for {key} must be array, anndata, or dataframe; "
            f"{type(data)} provided"
        )


def read_network(
    file_name,
    key
):

    try:
        df = _read_df(file_name, key)

        # Assume that this is a null prior
        # and this encodes # genes & # tfs
        if df.shape == (2, 1):
            return (df.iloc[0, 0], df.iloc[1, 0])

        return df
    except (KeyError, TypeError):
        return _read_ad(file_name, key)


def _read_ad(
    file_name,
    key
):

    with h5py.File(file_name, "r") as f:

        if key not in f.keys():
            return None

        if key + "_indptr" in f.keys():
            adata = ad.AnnData(
                _read_sparse(f, key)
            )
        else:
            adata = ad.AnnData(
                f[key][()]
            )

        adata.obs_names = _read_index(f, key + "_obs")
        adata.var_names = _read_index(f, key + "_var")

    return adata


def _write_ad(
    file_name,
    data,
    key
):

    with h5py.File(file_name, "a") as f:

        if _spsparse.issparse(data.X):
            _write_sparse(f, data.X, key)
        else:
            f.create_dataset(
                key,
                data=data.X
            )

        _write_index(f, data.var_names, key + "_var")
        _write_index(f, data.obs_names, key + "_obs")


def _write_sparse(
    f,
    matrix,
    key
):

    f.create_dataset(
        key,
        data=bytearray(sparse_type(matrix)[1], 'utf8')
    )

    f.create_dataset(
        key + "_indptr",
        data=matrix.indptr
    )

    f.create_dataset(
        key + "_indices",
        data=matrix.indices
    )

    f.create_dataset(
        key + "_data",
        data=matrix.data
    )

    f.create_dataset(
        key + "_shape",
        data=np.array(matrix.shape)
    )


def _read_sparse(
    f,
    key
):

    if key not in f.keys():
        return None

    else:
        stype = f[key][()]

    return sparse_type(stype)[0](
        (
            f[key + "_data"][()],
            f[key + "_indices"][()],
            f[key + "_indptr"][()]
        ),
        shape=f[key + "_shape"][()].tolist()
    )


def _write_index(
    f,
    idx,
    key
):

    f.create_dataset(
        key,
        data=idx.values.astype(str).astype(bytearray)
    )


def _read_index(
    f,
    key
):

    if key not in f.keys():
        raise KeyError(
            f"Key {key} not in h5 file"
        )

    return pd.Index(
        f[key][()]
    ).astype(str)


def _write_df(
    file_name,
    df,
    key
):

    with pd.HDFStore(file_name, mode="a") as f:
        df.to_hdf(
            f,
            key=key
        )


def _read_df(
    file_name,
    key
):
    with pd.HDFStore(file_name, mode='r') as f:
        return pd.read_hdf(
            f,
            key=key
        )
