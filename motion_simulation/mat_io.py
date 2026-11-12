"""MATLAB .mat reader supporting v7 (scipy) and v7.3 (HDF5) files."""
from __future__ import annotations

from typing import Any, Dict, Iterable


def _as_numeric_array(value, *, file_path: str, key: str):
    """Convert MATLAB/HDF5 numeric data to a regular NumPy array.

    MATLAB v7.3 stores complex arrays as an HDF5 compound dtype with
    ``real`` and ``imag`` fields; these are converted to a complex array.
    """
    import numpy as np

    array = np.asarray(value)
    field_names = array.dtype.names
    if field_names:
        fields = {str(name).lower(): name for name in field_names}
        real_name = fields.get("real")
        imag_name = fields.get("imag") or fields.get("imaginary")
        if real_name is not None and imag_name is not None:
            real = np.asarray(array[real_name])
            imag = np.asarray(array[imag_name])
            complex_dtype = np.result_type(real.dtype, imag.dtype, np.complex64)
            return real.astype(complex_dtype) + 1j * imag.astype(complex_dtype)

        raise TypeError(
            f"MATLAB variable {key!r} in {file_path} is a structured array "
            f"with fields {list(field_names)}, not a numeric image array."
        )

    if array.dtype.kind not in "biufc":
        raise TypeError(
            f"MATLAB variable {key!r} in {file_path} has unsupported dtype "
            f"{array.dtype}. Expected a real or complex numeric array."
        )
    return array


def _read_hdf5_dataset(dataset, *, file_path: str, key: str):
    """Read a MATLAB v7.3 dataset using MATLAB's logical axis order.

    MATLAB writes numeric arrays to HDF5 with dimensions reversed relative
    to NumPy. The ``MATLAB_class`` attribute distinguishes those datasets
    from ordinary HDF5 arrays written directly with h5py, whose dimensions
    are left unchanged.
    """
    import numpy as np

    array = _as_numeric_array(dataset[()], file_path=file_path, key=key)
    if "MATLAB_class" in dataset.attrs and array.ndim > 1:
        array = np.transpose(array, axes=tuple(reversed(range(array.ndim))))
    return array


def load_mat_var(file_path: str, preferred_keys: Iterable[str] = ("ima_comb1", "ima_comb")):
    """Load one variable from a MATLAB .mat file (v7 or v7.3).

    The first key of ``preferred_keys`` present in the file is returned; if
    none is present and the file holds exactly one variable, that variable
    is returned.
    """
    try:
        from scipy.io import loadmat  # type: ignore

        mat = loadmat(file_path, verify_compressed_data_integrity=False)
        for key in preferred_keys:
            if key in mat:
                return _as_numeric_array(mat[key], file_path=file_path, key=key)

        candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(candidate_keys) == 1:
            key = candidate_keys[0]
            return _as_numeric_array(mat[key], file_path=file_path, key=key)
        raise KeyError(
            f"None of preferred keys {list(preferred_keys)} found in {file_path}. "
            f"Available keys: {sorted(candidate_keys)}"
        )
    except NotImplementedError:
        import h5py  # type: ignore

        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            for key in preferred_keys:
                if key in f:
                    return _read_hdf5_dataset(f[key], file_path=file_path, key=key)

            if len(keys) == 1:
                key = keys[0]
                return _read_hdf5_dataset(f[key], file_path=file_path, key=key)

            raise KeyError(
                f"None of preferred keys {list(preferred_keys)} found in {file_path}. "
                f"Available keys: {sorted(keys)}"
            )


def inspect_mat(file_path: str) -> Dict[str, Any]:
    """Return variable names, shapes and dtypes of a .mat file (best effort)."""
    out: Dict[str, Any] = {"file": file_path}
    try:
        from scipy.io import loadmat  # type: ignore

        mat = loadmat(file_path, verify_compressed_data_integrity=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        shapes = {k: getattr(mat[k], "shape", None) for k in keys}
        dtypes = {k: str(getattr(mat[k], "dtype", None)) for k in keys}
        out.update({"format": "v7", "keys": keys, "shapes": shapes, "dtypes": dtypes})
        return out
    except NotImplementedError:
        import h5py  # type: ignore

        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            shapes = {}
            dtypes = {}
            for k in keys:
                try:
                    shapes[k] = f[k].shape
                    dtypes[k] = str(f[k].dtype)
                except Exception:
                    shapes[k] = None
                    dtypes[k] = None
            out.update({"format": "v7.3", "keys": keys, "shapes": shapes, "dtypes": dtypes})
            return out
