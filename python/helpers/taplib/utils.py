from copy import deepcopy
import numpy as np
from typing import Sequence


def ceildiv(a, b):
    """
    A helper function to calculate ceiling division without

    Args:
        a (_type_): The dividend
        b (_type_): The devisor

    Returns:
        _type_: The result
    """
    return -(a // -b)


def validate_and_clean_sizes_strides(
    sizes: Sequence[int] | None,
    strides: Sequence[int] | None,
    allow_none: bool = False,
    expected_dims: int | None = None,
) -> tuple[Sequence[int] | None, Sequence[int] | None]:
    """
    This is a helper function to validate sizes, strides and remove any
    unused values from upper dimensions if possible.

    Args:
        sizes (Sequence[int] | None): The transformation strides, or None
        strides (Sequence[int] | None): The transformation sizes, or None
        allow_none (bool, optional): Allow sizes and/or strides to be None. Defaults to False.
        expected_dims (int | None, optional): Number of dimensions expected for both sizes and strides. Defaults to None.

    Raises:
        ValueError: Validate sizes and strides

    Returns:
        tuple[Sequence[int] | None, Sequence[int] | None]: The 'cleaned' sizes and strides.
    """
    if not allow_none:
        if sizes is None:
            raise ValueError("Sizes is None, but expected Sequence[int]")
        if strides is None:
            raise ValueError("Strides is None, but expected Sequence[int]")
    # After this point can assume None is ok for sizes/strides

    if not (expected_dims is None):
        if expected_dims < 1:
            raise ValueError(f"Expected dimensions ({expected_dims}) should be >= 1")

    if sizes is None and strides is None:
        # nothing to do
        return None, None

    # Validate dimensions
    if (not (sizes is None)) and len(sizes) == 0:
        raise ValueError("len(sizes) must be >0")
    if (not (strides is None)) and len(strides) == 0:
        raise ValueError("len(strides) must be >0")

    if sizes and strides:
        if expected_dims:
            if len(sizes) != expected_dims:
                raise (
                    f"Num dimensions of sizes ({sizes}) is not expected number of dimensions ({expected_dims})"
                )
            if len(strides) != expected_dims:
                raise (
                    f"Num dimensions of strides ({strides}) is not expected number of dimensions ({expected_dims})"
                )
        elif len(strides) != len(sizes):
            raise ValueError(
                f"len(sizes) ({len(sizes)}) != len(strides) ({len(strides)})"
            )
    if strides:
        num_dims = len(strides)
    else:
        num_dims = len(sizes)

    # Validate sizes/strides values
    if sizes:
        sizes = deepcopy(sizes)
        for s in sizes:
            if s < 1:
                raise ValueError(f"All sizes must be >= 1, but got {sizes}")
    if strides:
        strides = deepcopy(strides)
        for s in strides:
            if s < 0:
                raise ValueError(f"All strides must be >= 0, but got {strides}")

    # Clean (set size=1, stride=0 for as many dims as possible)
    if sizes and strides:
        # Leave last dimension strides as whatever it happens to be
        for i in range(num_dims - 1):
            if sizes[i] == 1:
                if isinstance(strides, tuple):
                    # Tuple is immutable, so convert if necessary
                    strides = list(strides)
                strides[i] = 0
            else:
                break
    return sizes, strides


def validate_tensor_dims(
    tensor_dims: Sequence[int], expected_dims: int | None = None
) -> Sequence[int]:
    """
    This is a helper function used to validate dimensions of tensors, namely
    be ensuring each dimension is > 0 and the dimensionality is as expected.

    Args:
        tensor_dims (Sequence[int]): Tensor dimensions to check
        expected_dims (int | None, optional): Expected number of dimensions. Defaults to None.

    Raises:
        ValueError: Validate the tensor dimensions

    Returns:
        Sequence[int]: The validated tensor dimensions.
    """
    if not (expected_dims is None):
        if expected_dims < 1:
            raise ValueError(f"Expected dimensions ({expected_dims}) should be >= 1")
    tensor_dims = deepcopy(tensor_dims)

    # Validate tensor dims and offset, then set
    if len(tensor_dims) == 0:
        raise ValueError(
            f"Number of tensor dimensions must be >= 1 (dimensions={tensor_dims})"
        )
    for d in tensor_dims:
        if d <= 0:
            raise ValueError(
                f"Each tensor dimension must be >= 1 (dimensions={tensor_dims})"
            )

    # We can treat a 1-dimensional tensor as a 2-dimensional tensor,
    if len(tensor_dims) == 1:
        tensor_dims = [1, tensor_dims[0]]

    if not (expected_dims is None) and len(tensor_dims) != expected_dims:
        raise ValueError(
            f"Tensor dimension ({tensor_dims}) does not match expected dimension ({expected_dims})"
        )

    return tensor_dims


def validate_offset(offset: int, tensor_dims: Sequence[int] | None) -> int:
    """
    This is a helper function to validate an offset into the tensor.
    It primarily checks to see if the offset is a valid index to the tensor.

    Args:
        offset (int): The offset to check.
        tensor_dims (Sequence[int] | None): The dimensions of the tensor the offset corresponds to.

    Raises:
        ValueError: Validate the offset.

    Returns:
        int: The validated offset.
    """
    if offset < 0:
        raise ValueError(f"Offset must be >= 0 (offset={offset})")
    if tensor_dims:
        if offset >= np.prod(tensor_dims):
            raise ValueError(
                f"Offset too large: {offset}. Max value allowed for tensor: {np.prod(tensor_dims)}"
            )
    return offset
