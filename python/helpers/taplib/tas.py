from __future__ import annotations
from collections import abc
from copy import deepcopy
import numpy as np
from typing import Callable, Sequence

from .tap import TensorAccessPattern
from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_tensor_dims,
)


class TensorAccessSequence(abc.MutableSequence, abc.Iterable):
    """
    TensorAccessSequence is a MutableSequence and an Iterable. Generally, it is a thin wrapper around a list[TensorAccessPattern].

    The TensorAccessSequence is useful as a container of TensorAccessPatterns so that a grouping of patterns may be
    accessed in a particular order, or visualized or animated in sequence.
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
        """A TensorAccessSequence is a sequence of TensorAccessPatterns modelled after a list.

        The constructor can used given functions to generate a sequence of n steps. Allowed functions are the:
        * offset_fn(step: int, current_offset: int) -> new_offset: int
        * sizes_fn(step: int, current_sizes: Sequence[int]) -> new_sizes: Sequence[int]
        * strides_fn(step: int, currrent_strides: Sequence[int]) -> new_strides: Sequence[int]

        In lieu or in addition to a function, a default value for sizes/strides/offsets may also be set.

        Args:
            tensor_dims (Sequence[int]): Dimensions of the tensor. All TensorAccessPatterns in the sequence must share the tensor dimension.
            num_steps (int): Number of steps (elements) in the sequence.
            offset (int | None, optional): Offset into the sequence. Defaults to None.
            sizes (Sequence[int] | None, optional): Sizes for the TensorAccessPatterns Defaults to None.
            strides (Sequence[int] | None, optional): Strides for the TensorAccessPatterns in the sequence. Defaults to None.
            offset_fn (Callable[[int, int], int] | None, optional): A function to calculate the offset at each step. Defaults to None.
            sizes_fn (Callable[[int, Sequence[int]], Sequence[int]] | None, optional): A function to calculate the sizes at each step. Defaults to None.
            strides_fn (Callable[[int, Sequence[int]], Sequence[int]] | None, optional): A function to calculate the strides at teach step. Defaults to None.

        Raises:
            ValueError: Parameters are validated
        """
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
        """
        This alternative constructor creates a TensorAccessSequence from a sequence of TensorAccessPatterns.
        This is an alternative to the traditional constructor, and is useful for patterns that are difficult
        to express using the sizes/strides/offset functions.

        Args:
            taps (Sequence[TensorAccessPattern]): The sequence of tensor access patterns

        Raises:
            ValueError: At least one TensorAccessPattern must be specified
            ValueError: All TensorAccessPatterns in a sequence must share tensor dimensions

        Returns:
            TensorAccessSequence: A newly constructor TensorAccessSequence object
        """
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
        """
        Returns the access_order and access_count arrays of the TensorAccessPatterns in
        the sequence applied sequentially to the tensor.

        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

        Returns:
            tuple[np.ndarray, np.ndarray]: access_order, access_count
        """
        return self._calc_accesses(True, True)

    def access_order(self) -> np.ndarray:
        """
        The access_order ndarray sequentially counts access to elements in the
        tensor. If an element is accessed more than once, only the last count is reflected.

        The TensorAccessPatterns in the sequence are applied sequentially.

        Returns:
            np.ndarray: access_order
        """
        access_order, _ = self._calc_accesses(calc_order=True, calc_count=False)
        return access_order

    def access_count(self) -> np.ndarray:
        """
        The access_count ndarray contains the number of times each element is
        accessed by the tensor access pattern.

        The TensorAccessPatterns in the sequence are applied sequentially.

        Returns:
            np.ndarray: access_count
        """
        _, access_count = self._calc_accesses(calc_order=False, calc_count=True)
        return access_count

    def _calc_accesses(
        self, calc_order: bool, calc_count: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        # This is an internal method for calculating both the access_order and access_count
        # arrays. If needed, it will create both at once to avoid looping through the tensor
        # more than necessary.

        # TODO: this function is not particularly efficient, and could be improved.
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

    def animate(self, title: str | None = None, animate_access_count: bool = False):
        """
        Creates and returns a handle to a TensorAccessSequence animation. Each frame
        in the animation represents one TensorAccessPattern in the sequence.

        Args:
            title (str | None, optional): The title of the animation. Defaults to None.
            animate_access_count (bool, optional): Create an animation for the tensor access count, in addition to the tensor access order. Defaults to False.

        Raises:
            NotImplementedError: Not all dimensions of tensor may be visualized by animation at this time.

        Returns:
            animation.FuncAnimation: A handle to the animation, produced by the matplotlib.animation module.
        """
        from .visualization2d import animate_from_accesses

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
        title: str | None = None,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        """Provides a visual of the TensorAccessSequence using a graph.

        Args:
            title (str | None, optional): The title of the graph. Defaults to None.
            file_path (str | None, optional): The path to save the graph at. If None, it is not saved. Defaults to None.
            show_plot (bool, optional): Show the plot; this is useful when running in a Jupyter notebook. Defaults to True.
            plot_access_count (bool, optional): Plot the access count in addition to the access order. Defaults to False.

        Raises:
            NotImplementedError: Not all dimensions of tensor may be visualized.
        """
        from .visualization2d import visualize_from_accesses

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
        """
        This function creates an alternative way to compare access pattern sequences.
        Sometimes access patterns with different sizes/strides are functionally equivalent;
        to detect functional equivalency, this function uses iterators produced by
        access_generator() to compare the access patterns. This is more performant than
        comparing the numpy array access_order or access_count tensors, particularly
        when comparing sequences containing multiple tensor access patterns.

        Args:
            other (TensorAccessSequence): The TensorAccessSequence to compare to

        Returns:
            bool: True is functionally equivalent; False otherwise.
        """
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

    def insert(self, index: int, value: TensorAccessPattern):
        if self._tensor_dims != value.tensor_dims:
            raise ValueError(
                f"Cannot add TensorAccessPattern with tensor dims {value.tensor_dims} to TensorAccessSequence with tensor dims {self._tensor_dims}"
            )
        self._taps.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self._taps == other._taps and self._current_step == other._current_step
            )
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
