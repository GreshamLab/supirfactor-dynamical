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

    else:
        raise ValueError(
            "Sparse matrix must be CSR or CSC; "
            "BSR and COO is not supported"
        )


def write_network(
    file_name,
    data,
    key
):

    if isinstance(data, pd.DataFrame):
        return _write_df(file_name, data, key)

    elif isinstance(data, np.ndarray):
        return _write_df(file_name, pd.DataFrame(data), key)

    elif not isinstance(data, ad.AnnData):
        raise ValueError(
            f"Network data for {key} must be array, anndata, or dataframe; "
            f"{type(data)} provided"
        )

    elif not _spsparse.issparse(data.X):
        _write_df(file_name, data.to_df(), key)

    else:
        with h5py.File(file_name, "a") as f:
            _write_sparse(f, data.X, key)
            _write_index(f, data.var_names, key + "_var")
            _write_index(f, data.obs_names, key + "_obs")


def read_network(
    file_name,
    key
):

    try:
        return _read_df(file_name, key)
    except KeyError:
        return _read_ad(file_name, key)


def _read_ad(
    file_name,
    key
):

    with h5py.File(file_name, "r") as f:

        if key not in f.keys():
            return None

        adata = ad.AnnData(
            _read_sparse(f, key)
        )

        adata.obs_names = _read_index(f, key + "_obs")
        adata.var_names = _read_index(f, key + "_var")

    return adata


def _write_sparse(
    f,
    matrix,
    key
):

    f.create_dataset(
        key,
        data=sparse_type(matrix)[1]
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


def _read_sparse(
    f,
    key
):

    if key not in f.keys():
        return None

    else:
        stype = f[key][()]

    return sparse_type(stype)[0]((
        f[key + "_data"][()],
        f[key + "_indices"][()],
        f[key + "_indptr"][()]
    ))


def _write_index(
    f,
    idx,
    key
):

    f.create_dataset(
        key,
        data=idx.values.astype(str)
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
            key
        )


def _read_df(
    file_name,
    key
):
    with pd.HDFStore(file_name, mode='r') as f:
        return pd.read_hdf(
            f,
            key
        )
