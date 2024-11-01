from typing import Callable, Sequence

from .tensortile import TensorTile
from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)


class TensorTileIter:
    def __init__(
        self,
        tensor_dims: Sequence[int],
        offset: int | None,
        sizes: Sequence[int] | None,
        strides: Sequence[int] | None,
        offset_fn: Callable[[int, int], int] | None,
        sizes_fn: Callable[[int, Sequence[int]], Sequence[int]] | None,
        strides_fn: Callable[[int, Sequence[int]], Sequence[int]] | None,
        num_steps: int,
    ):
        self._current_step = 0

        # Check tensor dims, offset, sizes, strides
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        self._offset = offset
        if not (self._offset is None):
            self._offset = validate_offset(self._offset)
        self._sizes, self._strides = validate_and_clean_sizes_strides(
            sizes, strides, allow_none=True
        )

        # Make sure values or not None if iteration functions are None; also set default iter fn
        if offset_fn is None:
            if self._offset is None:
                raise ValueError("Offset must be provided if offset_fn is None")
            self._offset_fn = lambda _x: self._offset
        else:
            self._offset_fn = offset_fn
        if sizes_fn is None:
            if self._sizes is None:
                raise ValueError("Sizes must be provided if size_fn is None")
            self._sizes_fn = lambda _x: self._sizes
        else:
            self._sizes_fn = sizes_fn
        if strides_fn is None:
            if self._strides is None:
                raise ValueError("Strides must be provided if stride_fn is None")
            self._strides_fn = lambda _x: self._strides
        else:
            self._strides_fn = strides_fn

        # Validate and set num steps
        if num_steps < 1:
            raise ValueError(f"Number of steps must be >= 1 (but is {num_steps})")
        self._num_steps = num_steps

        # Pre-calculate tiles, because better for error handling up-front (and for visualizing full iter)
        # This is somewhat against the mentality behind iterations, but should be okay at the scale this
        # class will be used for (e.g., no scalability concerns with keeping all tiles in mem)
        self._tiles = []
        for step in range(self._num_steps):
            self._offset = self._offset_fn(step, self._offset)
            self._sizes = self._sizes_fn(step, self._sizes)
            self._strides = self._strides_fn(step, self._strides)

            self._tiles.append(
                TensorTile(
                    self._tensor_dims,
                    self._offset,
                    self._sizes,
                    self._strides,
                )
            )

    def __iter__(self):
        return self

    def __next__(self) -> TensorTile:
        if self._current_step == self._num_steps:
            raise StopIteration
        self._current_step += 1
        return self._tiles[self._current_step - 1]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self._tiles == other._tiles
                and self._num_steps == other._num_steps
                and self._current_step == other._current_step
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
