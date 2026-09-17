# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import itertools
import operator
from copy import deepcopy
from typing import Any, Generator, Sequence

import numpy as np

from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)


class TensorAccessPattern:
    """A TensorAccessPattern represents a data access pattern applied to a tensor of a specific dimension.

    This is a base class meant to generically represent such as transformation using sizes, strides,
    and an offset.
    """

    _DTYPE = np.int32

    def __init__(
        self,
        tensor_dims: Sequence[int],
        offset: int,
        sizes: Sequence[int],
        strides: Sequence[int],
    ):
        """An object representing an access pattern applied to a tensor.

        Args:
            tensor_dims (Sequence[int]): Dimensions of the tensor
            offset (int): Offset into the tensor to begin the transformation
            sizes (Sequence[int]): Transformation sizes
            strides (Sequence[int]): Transformation strides
        """  # noqa: D401
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        self._offset = validate_offset(offset, tensor_dims)
        cleaned_sizes, cleaned_strides = validate_and_clean_sizes_strides(
            sizes, strides
        )
        assert cleaned_sizes is not None and cleaned_strides is not None
        self._sizes: Sequence[int] = cleaned_sizes
        self._strides: Sequence[int] = cleaned_strides

    @classmethod
    def from_slice(cls, tensor_dims: Sequence[int], key: Any) -> "TensorAccessPattern":
        """Build an access pattern from numpy basic-slice notation.

        Lets a transfer be described the way the data is thought about --
        ``tap = TensorAccessPattern.from_slice(t.shape, np.s_[0::2, 1::2, ...])``
        -- instead of by hand-deriving the offset, wraps and steps that slice
        implies.

        The key is read directly rather than applied to a stand-in array: an
        ellipsis expands to the axes it covers, a slice contributes its start
        to the offset and its step to the stride, an integer contributes to the
        offset and drops its axis, and ``None`` adds a dimension nothing steps
        along. That is the definition of a strided walk, so the arithmetic is
        the answer rather than something read back off one.

        No dtype is needed. An access pattern is element-granular and every
        term here is in elements, so the same key yields the same pattern
        whatever the tensor's element type.

        Args:
            tensor_dims (Sequence[int]): Dimensions of the tensor being sliced.
            key (Any): Any numpy basic-indexing key -- integers, slices,
                ``Ellipsis`` and ``None``, alone or in a tuple. For example
                ``np.s_[0::2, 1::2, ...]``.

        Returns:
            TensorAccessPattern: The access pattern the key describes.

        Raises:
            TypeError: If the key uses advanced (fancy or boolean) indexing,
                which reaches elements a strided walk cannot.
            IndexError: If the key has more entries than the tensor has
                dimensions, more than one ellipsis, or an out-of-range integer.
            ValueError: If the key implies a negative stride. Reverse steps are
                expressible in numpy but not in a buffer descriptor, which only
                steps forward.
        """
        dims = tuple(tensor_dims)
        entries = key if isinstance(key, tuple) else (key,)

        ellipses = [i for i, k in enumerate(entries) if k is Ellipsis]
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        # Every entry but an ellipsis or a None consumes one tensor dimension;
        # the ellipsis stands for however many are left over.
        covered = sum(k is not Ellipsis and k is not None for k in entries)
        if covered > len(dims):
            raise IndexError(f"too many indices for a tensor of {len(dims)} dimensions")
        at = ellipses[0] if ellipses else len(entries)
        fill = (slice(None),) * (len(dims) - covered)
        entries = entries[:at] + fill + entries[at + 1 :]

        c_strides = [1] * len(dims)
        for axis in reversed(range(len(dims) - 1)):
            c_strides[axis] = c_strides[axis + 1] * dims[axis + 1]

        offset, sizes, strides, axis = 0, [], [], 0
        for entry in entries:
            if entry is None:
                # np.newaxis: a dimension the tensor does not have, so nothing
                # steps along it.
                sizes.append(1)
                strides.append(0)
                continue
            dim, c_stride = dims[axis], c_strides[axis]
            axis += 1
            if isinstance(entry, slice):
                start, stop, step = entry.indices(dim)
                offset += start * c_stride
                sizes.append(len(range(start, stop, step)))
                strides.append(step * c_stride)
                continue
            try:
                # operator.index is the protocol numpy itself uses to decide
                # whether something is an integer index. A bool is excluded
                # because numpy reads it as a mask, which adds an axis.
                if isinstance(entry, bool):
                    raise TypeError
                i = operator.index(entry)
            except TypeError:
                raise TypeError(
                    f"index {entry!r} is advanced indexing; an access pattern "
                    "is a strided walk, which only basic indexing -- integers, "
                    "slices, Ellipsis and None -- describes."
                ) from None
            if not -dim <= i < dim:
                raise IndexError(f"index {i} is out of bounds for a dimension of {dim}")
            offset += (i + dim if i < 0 else i) * c_stride

        if any(stride < 0 for stride in strides):
            raise ValueError(
                f"slice {key!r} implies strides {strides}, but a buffer "
                "descriptor only expresses forward steps."
            )
        # An all-integer key names a single element, leaving no dimensions at
        # all; say that as the one-element walk it is.
        return cls(dims, offset, sizes or [1], strides or [1])

    @property
    def tensor_dims(self) -> Sequence[int]:
        """A copy of the dimensions of the tensor.

        Returns:
            Sequence[int]: Tensor dimensions
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._tensor_dims)

    @property
    def offset(self) -> int:
        """Return the offset into the tensor.

        Returns:
            int: offset
        """
        return self._offset

    @property
    def sizes(self) -> Sequence[int]:
        """A copy of the access pattern sizes.

        Returns:
            Sequence[int]: Transformation sizes
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._sizes)

    @property
    def strides(self) -> Sequence[int]:
        """A copy of the access pattern strides.

        Returns:
            Sequence[int]: Transformation strides
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._strides)

    @property
    def transformation_dims(self) -> Sequence[tuple[int, int]]:
        """The access pattern represented as a sequence of (size, stride) tuples.

        Returns:
            Sequence[tuple[int, int]]: Transformation dimensions
        """
        return list(zip(self._sizes, self._strides))

    def accesses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the access_order and access_count arrays.

        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

        Returns:
            tuple[np.ndarray, np.ndarray]: access_order, access_count
        """
        return self._calculate_accesses(calc_order=True, calc_count=True)

    def access_order(self) -> np.ndarray:
        """Return the access_order ndarray, which sequentially counts access to elements in the tensor.

        If an element is accessed more than once, only the last count is reflected.

        Returns:
            np.ndarray: access_order
        """
        access_order_tensor, _ = self._calculate_accesses(
            calc_order=True, calc_count=False
        )
        return access_order_tensor

    def access_count(self) -> np.ndarray:
        """Return the access_count ndarray, which contains the number of times each element is accessed.

        Returns:
            np.ndarray: access_count
        """
        _, access_count_tensor = self._calculate_accesses(
            calc_order=False, calc_count=True
        )
        return access_count_tensor

    def _calculate_accesses(
        self, calc_order: bool, calc_count: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        # This is an internal method for calculating both the access_order and access_count
        # arrays. If needed, it will create both at once to avoid looping through the tensor
        # more than necessary.

        # TODO: should access order be a list of lists instead of generate two separate tensors?
        # TODO: for performance, should cache and return copies? Or just cache?
        if not calc_order and not calc_count:
            raise ValueError("Must select calc_order, calc_count, or both")

        # Initialize access order and count maps; we create them as flat arrays
        total_elems = np.prod(self._tensor_dims)
        access_order_tensor = np.full(total_elems, -1, dtype=self._DTYPE)
        access_count_tensor = np.full(total_elems, 0, dtype=self._DTYPE)
        access_count = 0

        # Get an iterator for the access indices
        access_idx_generator = self.access_generator()

        for access_idx in access_idx_generator:
            # Count the accesses
            if calc_count:
                access_count_tensor[access_idx] += 1
            # Enumerate the accesses
            if calc_order:
                access_order_tensor[access_idx] = access_count
                access_count += 1

        # Reshape to match tensor type since we created them initially as flat arrays
        access_order_tensor = access_order_tensor.reshape(self._tensor_dims)
        access_count_tensor = access_count_tensor.reshape(self._tensor_dims)
        return access_order_tensor, access_count_tensor

    def access_generator(self) -> Generator[int, None, None]:
        """Return an iterator over the access indices into the flattened tensor that this access pattern represents.

        This can be used to calculate the access count or to enumerate accesses.

        Yields:
            int: The next access index
        """
        total_elems = np.prod(self._tensor_dims)

        # Use itertools.product to collapse len(sizes) nested forloop into one forloop
        for dims in itertools.product(*[range(0, n) for n in self._sizes]):
            yield (
                self._offset + np.sum(np.multiply(dims, self._strides))
            ) % total_elems

    def compare_access_orders(self, other: TensorAccessPattern) -> bool:
        """Compare access patterns for functional equivalency.

        Sometimes access patterns with different sizes/strides are functionally equivalent;
        to detect functional equivalency, this function uses iterators produced by
        access_generator() to compare the access patterns. This is more performant than
        comparing the numpy array access_order or access_count tensors.

        Args:
            other (TensorAccessPattern): The TensorAccessPattern to compare to

        Raises:
            ValueError: other must be of type TensorAccessPattern

        Returns:
            bool: True if the TensorAccessPatterns are functionally equivalent; false otherwise.
        """
        # This function compares using access generators, which is more performant
        # than actually generating the access order or access count tensors.
        if not isinstance(other, TensorAccessPattern):
            raise ValueError(
                "Can only compare access order against another TensorAccessPattern"
            )
        my_generator = self.access_generator()
        other_generator = other.access_generator()
        return all(
            my_idx == other_idx
            for my_idx, other_idx in itertools.zip_longest(
                my_generator, other_generator
            )
        )

    def visualize(
        self,
        show_arrows: bool | None = None,
        title: str | None = None,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        """Visualize the TensorAccessPattern using a graph.

        Args:
            show_arrows (bool | None, optional): Display arrows between sequentially accessed elements. Defaults to None.
            title (str | None, optional): Title of the produced graph. Defaults to None.
            file_path (str | None, optional): Path to save the graph at; if none, it is not saved. Defaults to None.
            show_plot (bool, optional): Show the plot (this is useful for Jupyter notebooks). Defaults to True.
            plot_access_count (bool, optional): Plot the access count in addition to the access order. Defaults to False.

        Raises:
            NotImplementedError: This function is not implemented for all dimensions.
        """
        from .visualization2d import visualize_from_accesses

        if len(self._tensor_dims) != 2:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )
        if plot_access_count:
            access_order, access_count = self.accesses()
        else:
            access_count = None
            access_order = self.access_order()
        if title is None:
            title = str(self)
        visualize_from_accesses(
            access_order,
            access_count,
            title=title,
            show_arrows=show_arrows,
            file_path=file_path,
            show_plot=show_plot,
        )

    def __str__(self) -> str:
        return f"TensorAccessPattern({self.tensor_dims} offset={self._offset}, sizes={self._sizes}, strides={self._strides})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self._tensor_dims == other._tensor_dims
                and self._offset == other._offset
                and self._sizes == other._sizes
                and self._strides == other._strides
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
