from __future__ import annotations
from collections import abc
from copy import deepcopy
import matplotlib.animation as animation
import numpy as np
from typing import Callable, Sequence

from .tensortile import TensorTile
from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)
from .visualization2d import animate_from_access_tensors, visualize_from_access_tensors


class TensorTileSequence(abc.MutableSequence, abc.Iterable):
    """
    TensorTileSequence is a MutableSequence and an Iterable which is a thin wrapper around a list[TensorTiles].
    """

    def __init__(
        self,
        tensor_dims: Sequence[int],
        num_steps: int,
        offset: int | None = None,
        sizes: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        offset_fn: Callable[[int, int], int] | None = None,
        sizes_fn: Callable[[int, Sequence[int]], Sequence[int]] | None = None,
        strides_fn: Callable[[int, Sequence[int]], Sequence[int]] | None = None,
    ):
        self._current_step = 0

        # Check tensor dims, offset, sizes, strides
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        if not (offset is None):
            offset = validate_offset(offset)
        sizes, strides = validate_and_clean_sizes_strides(
            sizes, strides, allow_none=True
        )

        # Validate and set num steps
        if num_steps < 0:
            raise ValueError(f"Number of steps must be positive (but is {num_steps})")

        if num_steps == 0:
            if (
                offset != None
                or sizes != None
                or strides != None
                or offset_fn != None
                or sizes_fn != None
                or strides_fn != None
            ):
                raise ValueError(
                    f"If num_steps=0, no sizes/strides/offset information may be specified"
                )
            self._tiles = []
        else:
            # Make sure values or not None if iteration functions are None; also set default iter fn
            if offset_fn is None:
                if offset is None:
                    raise ValueError("Offset must be provided if offset_fn is None")
                offset_fn = lambda _step, _prev_offset: offset
            else:
                offset_fn = offset_fn
            if sizes_fn is None:
                if sizes is None:
                    raise ValueError("Sizes must be provided if size_fn is None")
                sizes_fn = lambda _step, _prev_sizes: sizes
            else:
                sizes_fn = sizes_fn
            if strides_fn is None:
                if strides is None:
                    raise ValueError("Strides must be provided if stride_fn is None")
                strides_fn = lambda _step, _prev_strides: strides
            else:
                strides_fn = strides_fn

            # Pre-calculate tiles, because better for error handling up-front (and for visualizing full iter)
            # This is somewhat against the mentality behind iterations, but should be okay at the scale this
            # class will be used for (e.g., no scalability concerns with keeping all tiles in mem)
            self._tiles = []
            for step in range(num_steps):
                offset = offset_fn(step, offset)
                sizes = sizes_fn(step, sizes)
                strides = strides_fn(step, strides)

                self._tiles.append(
                    TensorTile(
                        self._tensor_dims,
                        offset,
                        sizes,
                        strides,
                    )
                )

    @classmethod
    def from_tiles(cls, tiles: Sequence[TensorTile]) -> TensorTileSequence:
        if len(tiles) < 1:
            raise ValueError(
                "Received no tiles; must have at least one tile to create a tile sequence."
            )
        tensor_dims = tiles[0].tensor_dims
        for t in tiles:
            if t.tensor_dims != tensor_dims:
                raise ValueError(
                    f"Tiles have multiple tensor dimensions (found {tensor_dims} and {t.tensor_dims})"
                )
        tileseq = cls(
            tensor_dims,
            num_steps=1,
            offset=tiles[0].offset,
            sizes=tiles[0].sizes,
            strides=tiles[0].strides,
        )
        for t in tiles[1:]:
            tileseq.append(t)
        return tileseq

    def access_tensors(self) -> tuple[np.ndarray, np.ndarray]:
        total_elems = np.prod(self._tensor_dims)

        combined_access_order_tensor = np.full(
            total_elems, 0, TensorTile._DTYPE
        ).reshape(self._tensor_dims)
        combined_access_count_tensor = np.full(
            total_elems, 0, TensorTile._DTYPE
        ).reshape(self._tensor_dims)
        highest_count = 0
        for t in self._tiles:
            t_access_order, t_access_count = t.access_tensors()
            t_access_order[t_access_order != -1] += 1 + highest_count
            t_access_order[t_access_order == -1] = 0
            combined_access_order_tensor += t_access_order
            highest_count = np.max(combined_access_order_tensor)

            combined_access_count_tensor += t_access_count

        combined_access_order_tensor -= 1
        return (combined_access_order_tensor, combined_access_count_tensor)

    def animate(
        self, title: str = None, animate_access_count: bool = False
    ) -> animation.FuncAnimation:
        if title is None:
            title = "TensorTileSequence Animation"
        if len(self._tensor_dims) == 2:
            total_elems = np.prod(self._tensor_dims)

            animate_order_frames = [
                np.full(total_elems, -1, TensorTile._DTYPE).reshape(self._tensor_dims)
            ]
            if animate_access_count:
                animate_count_frames = [
                    np.full(total_elems, 0, TensorTile._DTYPE).reshape(
                        self._tensor_dims
                    )
                ]
            else:
                animate_count_frames = None

            for t in self._tiles:
                t_access_order, t_access_count = t.access_tensors()
                animate_order_frames.append(t_access_order)
                if animate_access_count:
                    animate_count_frames.append(t_access_count)

            return animate_from_access_tensors(
                animate_order_frames,
                animate_count_frames,
                title=title,
            )

        else:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )

    def visualize(
        self,
        title: str = None,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        if len(self._tensor_dims) != 2:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )

        if title is None:
            title = "TensorTileSequence"
        access_order_tensor, access_count_tensor = self.access_tensors()
        if not plot_access_count:
            access_count_tensor = None

        visualize_from_access_tensors(
            access_order_tensor,
            access_count_tensor,
            title=title,
            show_arrows=False,
            file_path=file_path,
            show_plot=show_plot,
        )

    def __contains__(self, tile: TensorTile):
        return tile in self._tiles

    def __iter__(self):
        return iter(deepcopy(self._tiles))

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> TensorTile:
        return self._tiles[idx]

    def __setitem__(self, idx: int, tile: TensorTile):
        if self._tensor_dims != tile.tensor_dims:
            raise ValueError(
                f"Cannot add tile with tensor dims {tile.tensor_dims} to sequence of tiles with tensor dims {self._tensor_dims}"
            )
        self._tiles[idx] = deepcopy(tile)

    def __delitem__(self, idx: int):
        del self._tiles[idx]

    def insert(self, idx: int, tile: TensorTile):
        if self._tensor_dims != tile.tensor_dims:
            raise ValueError(
                f"Cannot add tile with tensor dims {tile.tensor_dims} to sequence of tiles with tensor dims {self._tensor_dims}"
            )
        self._tiles.insert(idx, tile)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self._tiles == other._tiles
                and self._current_step == other._current_step
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
