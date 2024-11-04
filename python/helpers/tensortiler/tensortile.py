from copy import deepcopy
import numpy as np
import itertools
from typing import Sequence

from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)
from .visualization2d import visualize_from_access_tensors


class TensorTile:
    _DTYPE = np.int32

    def __init__(
        self,
        tensor_dims: Sequence[int],
        offset: int,
        sizes: Sequence[int],
        strides: Sequence[int],
    ):
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        self._offset = validate_offset(offset)
        if self._offset >= np.prod(tensor_dims):
            raise ValueError(
                f"Offset too large: {self._offset}. Max value allowed for tensor: {np.prod(tensor_dims)}"
            )
        self._sizes, self._strides = validate_and_clean_sizes_strides(sizes, strides)

    @property
    def tensor_dims(self) -> Sequence[int]:
        # Copy to prevent callers from mutating self
        return deepcopy(self._tensor_dims)

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def sizes(self) -> Sequence[int]:
        # Copy to prevent callers from mutating self
        return deepcopy(self._sizes)

    @property
    def strides(self) -> Sequence[int]:
        # Copy to prevent callers from mutating self
        return deepcopy(self._strides)

    @property
    def transformation_dims(self) -> Sequence[tuple[int, int]]:
        return list(zip(self._sizes, self._strides))

    def access_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        # TODO: should access order be a list of lists instead of generate two separate tensors?
        # TODO: for performance, should cache and return copies? Or just cache?

        # Initialize access order and count maps
        total_elems = np.prod(self._tensor_dims)
        access_order_tensor = np.full(total_elems, -1, dtype=self._DTYPE)
        access_count_tensor = np.full(total_elems, 0, dtype=self._DTYPE)

        # Use itertools.product to collapse len(sizes) nested forloop into one forloop
        access_count = 0
        for dims in itertools.product(*[range(0, n) for n in self._sizes]):
            access_idx = (
                self._offset + np.sum(np.multiply(dims, self._strides))
            ) % total_elems
            access_count_tensor[access_idx] += 1
            access_order_tensor[access_idx] = access_count
            access_count += 1

        access_order_tensor = access_order_tensor.reshape(self._tensor_dims)
        access_count_tensor = access_count_tensor.reshape(self._tensor_dims)
        return access_order_tensor, access_count_tensor

    def visualize(
        self,
        show_arrows: bool | None = None,
        title: str = None,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        access_order, access_count = self.access_tensors()
        if title is None:
            title = str(self)
        if not plot_access_count:
            access_count = None
        if len(self._tensor_dims) == 2:
            visualize_from_access_tensors(
                access_order,
                access_count,
                title=title,
                show_arrows=show_arrows,
                file_path=file_path,
                show_plot=show_plot,
            )
        else:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )

    def __str__(self) -> str:
        return f"TensorTile(offset={self._offset}, sizes={self._sizes}, strides={self._strides})"

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
