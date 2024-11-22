from __future__ import annotations
from collections import abc
from copy import deepcopy
import matplotlib.animation as animation
import numpy as np
from typing import Callable, Sequence

from .tap import TensorAccessPattern
from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)
from .visualization2d import animate_from_accesses, visualize_from_accesses


class TensorAccessSequence(abc.MutableSequence, abc.Iterable):
    """
    TensorAccessSequence is a MutableSequence and an Iterable which is a thin wrapper around a list[TensorAccessPattern].
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
            offset = validate_offset(offset, self._tensor_dims)
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
            self._taps = []
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

            # Pre-calculate taps, because better for error handling up-front (and for visualizing full iter)
            # This is somewhat against the mentality behind iterations, but should be okay at the scale this
            # class will be used for (e.g., no scalability concerns with keeping all taps in mem)
            self._taps = []
            for step in range(num_steps):
                offset = offset_fn(step, offset)
                sizes = sizes_fn(step, sizes)
                strides = strides_fn(step, strides)

                self._taps.append(
                    TensorAccessPattern(
                        self._tensor_dims,
                        offset,
                        sizes,
                        strides,
                    )
                )

    @classmethod
    def from_taps(cls, taps: Sequence[TensorAccessPattern]) -> TensorAccessSequence:
        if len(taps) < 1:
            raise ValueError(
                "Received no TensorAccessPatterns; must have at least one TensorAccessPatterns to create a TensorAccessSequence."
            )
        tensor_dims = taps[0].tensor_dims
        for t in taps:
            if t.tensor_dims != tensor_dims:
                raise ValueError(
                    f"TensorAccessPatterns have multiple tensor dimensions (found {tensor_dims} and {t.tensor_dims})"
                )
        tas = cls(
            tensor_dims,
            num_steps=1,
            offset=taps[0].offset,
            sizes=taps[0].sizes,
            strides=taps[0].strides,
        )
        for t in taps[1:]:
            tas.append(t)
        return tas

    def accesses(self) -> tuple[np.ndarray, np.ndarray]:
        return self._calc_accesses(True, True)

    def access_order(self) -> np.ndarray:
        access_order, _ = self._calc_accesses(calc_order=True, calc_count=False)
        return access_order

    def access_count(self) -> np.ndarray:
        _, access_count = self._calc_accesses(calc_order=False, calc_count=True)
        return access_count

    def _calc_accesses(
        self, calc_order: bool, calc_count: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if not calc_order and not calc_count:
            raise ValueError("Must select calc_order, calc_count, or both")

        total_elems = np.prod(self._tensor_dims)
        combined_access_order_tensor = None
        combined_access_count_tensor = None

        if calc_order:
            combined_access_order_tensor = np.full(
                total_elems, 0, TensorAccessPattern._DTYPE
            ).reshape(self._tensor_dims)
            highest_count = 0
        if calc_count:
            combined_access_count_tensor = np.full(
                total_elems, 0, TensorAccessPattern._DTYPE
            ).reshape(self._tensor_dims)
        for t in self._taps:
            if calc_order and calc_count:
                t_access_order, t_access_count = t.accesses()
            elif calc_order:
                t_access_order = t.access_order()
            else:
                t_access_count = t.access_count()

            if calc_order:
                t_access_order[t_access_order != -1] += 1 + highest_count
                t_access_order[t_access_order == -1] = 0
                combined_access_order_tensor += t_access_order
                highest_count = np.max(combined_access_order_tensor)
            if calc_count:
                combined_access_count_tensor += t_access_count

        if calc_order:
            combined_access_order_tensor -= 1
        return (combined_access_order_tensor, combined_access_count_tensor)

    def animate(
        self, title: str = None, animate_access_count: bool = False
    ) -> animation.FuncAnimation:
        if len(self._tensor_dims) != 2:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )

        if title is None:
            title = "TensorAccessSequence Animation"
        total_elems = np.prod(self._tensor_dims)

        animate_order_frames = [
            np.full(total_elems, -1, TensorAccessPattern._DTYPE).reshape(
                self._tensor_dims
            )
        ]

        animate_count_frames = None
        if animate_access_count:
            animate_count_frames = [
                np.full(total_elems, 0, TensorAccessPattern._DTYPE).reshape(
                    self._tensor_dims
                )
            ]

        for t in self._taps:
            if animate_access_count:
                t_access_order, t_access_count = t.accesses()
                animate_count_frames.append(t_access_count)
            else:
                t_access_order = t.access_order()
            animate_order_frames.append(t_access_order)

        return animate_from_accesses(
            animate_order_frames,
            animate_count_frames,
            title=title,
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
            title = "TensorAccessSequence"
        if plot_access_count:
            access_order_tensor, access_count_tensor = self.accesses()
        else:
            access_order_tensor = self.access_order()
            access_count_tensor = None

        visualize_from_accesses(
            access_order_tensor,
            access_count_tensor,
            title=title,
            show_arrows=False,
            file_path=file_path,
            show_plot=show_plot,
        )

    def compare_access_orders(self, other: TensorAccessSequence) -> bool:
        if len(self._taps) != len(other._taps):
            return False
        for my_tap, other_tap in zip(self._taps, other._taps):
            if not my_tap.compare_access_orders(other_tap):
                return False
        return True

    def __contains__(self, tap: TensorAccessPattern):
        return tap in self._taps

    def __iter__(self):
        return iter(self._taps)

    def __len__(self) -> int:
        return len(self._taps)

    def __getitem__(self, idx: int) -> TensorAccessPattern:
        return self._taps[idx]

    def __setitem__(self, idx: int, tap: TensorAccessPattern):
        if self._tensor_dims != tap.tensor_dims:
            raise ValueError(
                f"Cannot add TensorAccessPattern with tensor dims {tap.tensor_dims} to TensorAccessSequence with tensor dims {self._tensor_dims}"
            )
        self._taps[idx] = deepcopy(tap)

    def __delitem__(self, idx: int):
        del self._taps[idx]

    def insert(self, idx: int, tap: TensorAccessPattern):
        if self._tensor_dims != tap.tensor_dims:
            raise ValueError(
                f"Cannot add TensorAccessPattern with tensor dims {tap.tensor_dims} to TensorAccessSequence with tensor dims {self._tensor_dims}"
            )
        self._taps.insert(idx, tap)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self._taps == other._taps and self._current_step == other._current_step
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
