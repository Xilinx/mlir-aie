from __future__ import annotations

from copy import deepcopy
import numpy as np
import itertools
from typing import Sequence, Generator

from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)


class TensorAccessPattern:
    """
    A TensorAccessPattern represents a data access pattern applied to a tensor
    of a specific dimension. This is a base class meant to generically represent
    such as transformation using sizes, strides, and an offset.
    """

    _DTYPE = np.int32

    def __init__(
        self,
        tensor_dims: Sequence[int],
        offset: int,
        sizes: Sequence[int],
        strides: Sequence[int],
    ):
        """
        An object representing an access pattern applied to a tensor.

        Args:
            tensor_dims (Sequence[int]): Dimensions of the tensor
            offset (int): Offset into the tensor to begin the transformation
            sizes (Sequence[int]): Transformation sizes
            strides (Sequence[int]): Transformation strides
        """
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        self._offset = validate_offset(offset, tensor_dims)
        self._sizes, self._strides = validate_and_clean_sizes_strides(sizes, strides)

    @property
    def tensor_dims(self) -> Sequence[int]:
        """
        A copy of the dimensions of the tensor

        Returns:
            Sequence[int]: Tensor dimensions
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._tensor_dims)

    @property
    def offset(self) -> int:
        """
        Returns the offset into the tensor

        Returns:
            int: offset
        """
        return self._offset

    @property
    def sizes(self) -> Sequence[int]:
        """
        A copy of the access pattern sizes

        Returns:
            Sequence[int]: Transformation sizes
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._sizes)

    @property
    def strides(self) -> Sequence[int]:
        """
        A copy of the access pattern strides

        Returns:
            Sequence[int]: Trsnformation strides
        """
        # Copy to prevent callers from mutating self
        return deepcopy(self._strides)

    @property
    def transformation_dims(self) -> Sequence[tuple[int, int]]:
        """
        The access pattern represented as a sequence of (size, stride) tuples

        Returns:
            Sequence[tuple[int, int]]: Transformation dimensions
        """
        return list(zip(self._sizes, self._strides))

    def accesses(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the access_order and access_count arrays.

        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

        Returns:
            tuple[np.ndarray, np.ndarray]: access_order, access_count
        """
        return self._calculate_accesses(calc_order=True, calc_count=True)

    def access_order(self) -> np.ndarray:
        """
        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        Returns:
            np.ndarray: access_order
        """
        access_order_tensor, _ = self._calculate_accesses(
            calc_order=True, calc_count=False
        )
        return access_order_tensor

    def access_count(self) -> np.ndarray:
        """
        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

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
        access_order_tensor = None
        if calc_order:
            access_order_tensor = np.full(total_elems, -1, dtype=self._DTYPE)
            access_count = 0
        access_count_tensor = None
        if calc_count:
            access_count_tensor = np.full(total_elems, 0, dtype=self._DTYPE)

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
        if calc_order:
            access_order_tensor = access_order_tensor.reshape(self._tensor_dims)
        if calc_count:
            access_count_tensor = access_count_tensor.reshape(self._tensor_dims)
        return access_order_tensor, access_count_tensor

    def access_generator(self) -> Generator[int, None, None]:
        """This function returns an iterator that returns the access index
        into the flattened tensor that this access pattern represents. This can
        be used to calculate the access count or to enumerate accesses.

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
        """
        This function creates an alternative way to compare access patterns.
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
