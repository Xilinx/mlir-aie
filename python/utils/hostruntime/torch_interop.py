# torch_interop.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Zero-copy bridging between numpy dtypes and their torch equivalents."""

import numpy as np

# Mapping from ml_dtypes (non-native numpy) types to their torch equivalents.
# Native numpy dtypes (float32, int32, …) are handled directly by torch.from_numpy
# and do not need an entry here.
# Populated lazily at first use to avoid importing torch/ml_dtypes at module load.
_ML_DTYPE_TO_TORCH: dict | None = None


def _ml_dtype_to_torch_map():
    global _ML_DTYPE_TO_TORCH
    if _ML_DTYPE_TO_TORCH is None:
        import ml_dtypes
        import torch  # pyright: ignore[reportMissingImports]

        _candidates = {
            ml_dtypes.bfloat16: torch.bfloat16,
        }
        for attr in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        ):
            ml_dt = getattr(ml_dtypes, attr, None)
            torch_dt = getattr(torch, attr, None)
            if ml_dt is not None and torch_dt is not None:
                _candidates[ml_dt] = torch_dt
        _ML_DTYPE_TO_TORCH = {
            np.dtype(ml_dt): torch_dt for ml_dt, torch_dt in _candidates.items()
        }
    return _ML_DTYPE_TO_TORCH


# Same-width unsigned integer dtype for the ND reinterpret-view trick.
_UINT_VIEW_DTYPE = {
    1: np.uint8,
    2: np.uint16,
    4: np.uint32,
    8: np.uint64,
}


def _array_to_torch(array: np.ndarray):
    """Convert a numpy array to a torch tensor, zero-copy.

    For native numpy dtypes (float32, float16, int32, …) torch.from_numpy is used directly
    (fastest path for these types).

    For ml_dtypes types (bfloat16, float8_*) that torch cannot consume via from_numpy:
    reinterpret as a same-width unsigned integer numpy view, wrap with from_numpy,
    then view as the target torch dtype.  This is guaranteed zero-copy for all ranks.

    Raises:
        ImportError: If torch is not installed.
    """
    # _ml_dtype_to_torch_map() imports torch (raising ImportError with a helpful message
    # if absent) and returns the ml_dtype -> torch dtype mapping.
    torch_dtype = _ml_dtype_to_torch_map().get(array.dtype)
    import torch  # pyright: ignore[reportMissingImports]  # already imported by _ml_dtype_to_torch_map(); cached by Python

    if torch_dtype is None:
        # Native numpy dtype: torch.from_numpy handles it directly and fastest.
        return torch.from_numpy(array)

    # ml_dtype: reinterpret memory as a same-width uint, then view as the torch dtype.
    uint_dtype = _UINT_VIEW_DTYPE[array.dtype.itemsize]
    return torch.from_numpy(array.view(uint_dtype)).view(torch_dtype)
