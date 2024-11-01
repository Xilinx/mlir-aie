import numpy as np
from typing import Sequence

from .tensortiler2d import TensorTiler2D
from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)


class TensorTile:
    def __init__(
        self,
        tensor_dims: Sequence[int],
        offset: int,
        sizes: Sequence[int],
        strides: Sequence[int],
    ):
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        self._offset = validate_offset(offset)
        self._sizes, self._strides = validate_and_clean_sizes_strides(sizes, strides)

    @property
    def transformation_dims(self) -> Sequence[tuple[int, int]]:
        return list(zip(self._sizes, self._strides))

    def visualize(
        self,
        show_arrows: bool = True,
        show_numbers: bool = False,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        if len(self._tensor_dims) == 2:
            TensorTiler2D.generate_access_graphs(
                self._tensor_dims,
                self._sizes,
                self._strides,
                offset=self._offset,
                show_arrows=show_arrows,
                show_numbers=show_numbers,
                file_path=file_path,
                show_plot=show_plot,
                plot_access_count=plot_access_count,
            )
        else:
            raise ValueError(
                "Not Yet Developed: "
                "TensorTile visualization only supported for 1- and 2-dimensional tensors. "
                "Contributions welcome!"
            )

    def access_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self._tensor_dims) == 2:
            return TensorTiler2D.get_access_tensors(
                self._tensor_dims,
                self._sizes,
                self._strides,
                offset=self._offset,
            )
        else:
            raise ValueError(
                "Not Yet Developed: "
                "Access tensors only supported for 1- and 2-dimensional tensors. "
                "Contributions welcome!"
            )

    def __str__(self) -> str:
        return (
            f"TensorTile(tensor_dims={self._tensor_dims}, "
            f"offset={self._offset}, sizes={self._sizes}, strides={self._strides})"
        )

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
