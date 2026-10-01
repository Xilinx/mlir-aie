# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import abc
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Iterator, NamedTuple, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation

from .symbolic import (
    is_sym,
    require,
    sceildiv,
    show,
    sint,
    smin,
    sprod,
    sselect,
    sym_any,
)
from .tap import TensorAccessPattern, _accesses
from .utils import (
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_permutation,
    validate_tensor_dims,
)

IntLike = Any  # int, np.integer, or a staged aie.ir.Value


def _constant_fn(value):
    def _fn(_step, _prev):
        return value

    return _fn


class TensorAccessSequence(abc.MutableSequence, abc.Iterable):
    """TensorAccessSequence is a MutableSequence and an Iterable. Generally, it is a thin wrapper around a list[TensorAccessPattern].

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
        """  # noqa: D401
        # Check tensor dims, offset, sizes, strides
        self._tensor_dims = validate_tensor_dims(tensor_dims)
        if offset is not None:
            offset = validate_offset(offset, self._tensor_dims)
        sizes, strides = validate_and_clean_sizes_strides(
            sizes, strides, allow_none=True
        )

        # Validate and set num steps
        if num_steps < 0:
            raise ValueError(f"Number of steps must be positive (but is {num_steps})")

        if num_steps == 0:
            if (
                offset is not None
                or sizes is not None
                or strides is not None
                or offset_fn is not None
                or sizes_fn is not None
                or strides_fn is not None
            ):
                raise ValueError(
                    "If num_steps=0, no sizes/strides/offset information may be specified"
                )
            self._taps = []
        else:
            # Make sure values or not None if iteration functions are None; also set default iter fn
            if offset_fn is not None:
                resolved_offset_fn = offset_fn
            else:
                if offset is None:
                    raise ValueError("Offset must be provided if offset_fn is None")
                resolved_offset_fn = _constant_fn(offset)

            if sizes_fn is not None:
                resolved_sizes_fn = sizes_fn
            else:
                if sizes is None:
                    raise ValueError("Sizes must be provided if size_fn is None")
                resolved_sizes_fn = _constant_fn(sizes)

            if strides_fn is not None:
                resolved_strides_fn = strides_fn
            else:
                if strides is None:
                    raise ValueError("Strides must be provided if stride_fn is None")
                resolved_strides_fn = _constant_fn(strides)

            # Pre-calculate taps, because better for error handling up-front (and for visualizing full iter)
            # This is somewhat against the mentality behind iterations, but should be okay at the scale this
            # class will be used for (e.g., no scalability concerns with keeping all taps in mem)
            self._taps = []
            cur_offset: Any = offset
            cur_sizes: Any = sizes
            cur_strides: Any = strides
            for step in range(num_steps):
                cur_offset = resolved_offset_fn(step, cur_offset)
                cur_sizes = resolved_sizes_fn(step, cur_sizes)
                cur_strides = resolved_strides_fn(step, cur_strides)

                self._taps.append(
                    TensorAccessPattern(
                        self._tensor_dims,
                        cur_offset,
                        cur_sizes,
                        cur_strides,
                    )
                )

    @classmethod
    def from_taps(cls, taps: Sequence[TensorAccessPattern]) -> TensorAccessSequence:
        """Create a TensorAccessSequence from a sequence of TensorAccessPatterns.

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

    @property
    def tensor_dims(self) -> Sequence[int]:
        """A copy of the dimensions of the tensor every pattern in the sequence walks."""
        return list(self._tensor_dims)

    def accesses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the access_order and access_count arrays of the patterns walked one after another.

        The access_order array numbers the accesses to each element of the
        tensor across the whole sequence, -1 where no pattern goes; an element
        accessed more than once holds its last number. The access_count array
        holds the number of times the sequence accesses each element.

        Returns:
            tuple[np.ndarray, np.ndarray]: access_order, access_count
        """
        walk = np.concatenate(
            [np.empty(0, np.int64)] + [t._walk().reshape(-1) for t in self._taps]
        )
        return _accesses(walk, self._tensor_dims)

    def access_order(self) -> np.ndarray:
        """Return the access_order array of `accesses()`."""
        return self.accesses()[0]

    def access_count(self) -> np.ndarray:
        """Return the access_count array of `accesses()`."""
        return self.accesses()[1]

    def animate(
        self, title: str | None = None, animate_access_count: bool = False
    ) -> "FuncAnimation":
        """Create and return a handle to a TensorAccessSequence animation.

        Each frame in the animation represents one TensorAccessPattern in the sequence.

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

        animate_count_frames: list[np.ndarray] | None = None
        if animate_access_count:
            animate_count_frames = [
                np.full(total_elems, 0, TensorAccessPattern._DTYPE).reshape(
                    self._tensor_dims
                )
            ]

        for t in self._taps:
            if animate_count_frames is not None:
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
        """Provide a visual of the TensorAccessSequence using a graph.

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
        """Compare access pattern sequences for functional equivalency.

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

    def __contains__(self, tap: object):
        return tap in self._taps

    def __iter__(self):
        return iter(self._taps)

    def __len__(self) -> int:
        return len(self._taps)

    def __getitem__(self, idx):
        return self._taps[idx]

    def __setitem__(self, idx, tap):
        if self._tensor_dims != tap.tensor_dims:
            raise ValueError(
                f"Cannot add TensorAccessPattern with tensor dims {tap.tensor_dims} to TensorAccessSequence with tensor dims {self._tensor_dims}"
            )
        self._taps[idx] = copy(tap)

    def __delitem__(self, idx):
        del self._taps[idx]

    def insert(self, index: int, value: TensorAccessPattern):
        if self._tensor_dims != value.tensor_dims:
            raise ValueError(
                f"Cannot add TensorAccessPattern with tensor dims {value.tensor_dims} to TensorAccessSequence with tensor dims {self._tensor_dims}"
            )
        self._taps.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, TensorAccessSequence):
            return list(self._taps) == list(other._taps)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class _GridAxis(NamedTuple):
    """One grid axis of a `TileGrid`.

    Attributes:
        steps (IntLike): Number of grid positions along the axis.
        stride (IntLike): Element distance between consecutive tiles.
        tiles (IntLike): Number of tiles along the axis.
        step (IntLike): Tiles between the members of one group; position
            `p` names the group whose first tile is
            `(p // step) * step * repeat + p % step`.
        repeat (IntLike): Nominal number of tiles in one group.
        rep_pos (int | None): Index of the group's repeat dimension among the
            tile dimensions, or None if the axis is not grouped.
    """

    steps: IntLike
    stride: IntLike
    tiles: IntLike
    step: IntLike
    repeat: IntLike
    rep_pos: int | None


class TileGrid(TensorAccessSequence):
    """A tiling of a `TensorAccessPattern`: a sequence of tiles.

    Made by `TensorAccessPattern.tile()` and `TensorAccessPattern.partition()`.
    `grid[step]` is the tile at a linear step and `grid[i, j]` the tile at a
    grid position; both are `TensorAccessPattern` objects. Indices may be
    staged runtime values (a `range_` induction variable, say), in which
    case the tile's offset is staged arithmetic: that is what lets one
    compiled runtime sequence walk a runtime-sized tensor.

    Refine the grid with `group()`, `order()`, `permute_tile()` and
    `repeat()`; each returns a new grid. A grid cannot be edited in place.
    Like any `TensorAccessSequence` it can be iterated,
    visualized and animated once its values are concrete.
    """

    def __init__(
        self,
        tensor_dims: Sequence[IntLike],
        offset: IntLike,
        grid: Sequence[_GridAxis],
        tile_sizes: Sequence[IntLike],
        tile_strides: Sequence[IntLike],
        order: Sequence[int] | None = None,
        partial: bool = False,
        tile_axes: Sequence[int] | None = None,
    ):
        """Create a grid; use `TensorAccessPattern.tile()` instead of calling this.

        Args:
            tensor_dims (Sequence[IntLike]): Shape of the tensor the tiles walk.
            offset (IntLike): Element offset of the first tile.
            grid (Sequence[_GridAxis]): The grid axes.
            tile_sizes (Sequence[IntLike]): Nominal extent of each tile dimension.
            tile_strides (Sequence[IntLike]): Element step of each tile dimension.
            order (Sequence[int] | None, optional): Step order over the grid axes,
                slowest first. Defaults to row-major.
            partial (bool, optional): Whether groups at the tensor edge may be
                short. Defaults to False.
            tile_axes (Sequence[int] | None, optional): The grid axis each tile
                dimension walks, -1 for a repeat. Defaults to one per axis.
        """
        if len(grid) == 0:
            raise ValueError("a TileGrid needs at least one grid axis")
        if len(tile_sizes) == 0 or len(tile_sizes) != len(tile_strides):
            raise ValueError(
                "a TileGrid needs matching, non-empty tile sizes and strides"
            )
        self._tensor_dims = list(tensor_dims)
        self._offset = offset
        self._grid = tuple(grid)
        self._tile_sizes = list(tile_sizes)
        self._tile_strides = list(tile_strides)
        self._order = (
            tuple(range(len(grid)))
            if order is None
            else validate_permutation(order, len(grid), "order")
        )
        self._partial = bool(partial)
        self._tile_axes = (
            tuple(range(len(tile_sizes))) if tile_axes is None else tuple(tile_axes)
        )
        if len(self._tile_axes) != len(tile_sizes):
            raise ValueError("tile_axes needs one entry per tile dimension")
        self._cached_taps: list[TensorAccessPattern] | None = None

    # ------------------------------------------------------------------ shape

    @property
    def grid_shape(self) -> list[IntLike]:
        """Steps along each grid axis."""
        return [a.steps for a in self._grid]

    @property
    def grid_strides(self) -> list[IntLike]:
        """Element stride between consecutive tiles along each grid axis."""
        return [a.stride for a in self._grid]

    @property
    def tile_shape(self) -> list[IntLike]:
        """Nominal tile dimensions (a ragged group's last step may be smaller)."""
        return list(self._tile_sizes)

    @property
    def tile_strides(self) -> list[IntLike]:
        """Element step of each tile dimension."""
        return list(self._tile_strides)

    @property
    def num_steps(self) -> IntLike:
        """Number of tiles; staged when the grid is runtime-sized."""
        return sprod(self.grid_shape)

    @property
    def is_grouped(self) -> bool:
        """Whether `group()` has been applied."""
        return any(a.rep_pos is not None for a in self._grid)

    @property
    def is_symbolic(self) -> bool:
        """Whether any value of the grid is staged."""
        vals = [
            self._offset,
            *self._tile_sizes,
            *self._tile_strides,
            *self._tensor_dims,
        ]
        for a in self._grid:
            vals += [a.steps, a.stride, a.tiles, a.step, a.repeat]
        return sym_any(vals)

    @property
    def tap(self) -> TensorAccessPattern:
        """The whole grid as one pattern: grid axes in step order, then the tile.

        Only an ungrouped grid is a single strided walk. With `order("col")`
        the grid axes come column-major, so the walk goes down a column of
        tiles before moving to the next column.

        Raises:
            ValueError: If the grid is grouped.
        """
        if self.is_grouped:
            raise ValueError(
                "a grouped TileGrid is not a single strided walk; index it instead"
            )
        axes = [self._grid[p] for p in self._order]
        return TensorAccessPattern._raw(
            self._tensor_dims,
            self._offset,
            [a.tiles for a in axes] + self._tile_sizes,
            [a.stride for a in axes] + self._tile_strides,
        )

    # --------------------------------------------------------------- indexing

    def _at(self, *index: IntLike) -> TensorAccessPattern:
        # The tile at a grid position, one index per grid axis. Negative
        # concrete indices count from the end; staged ones are guarded.
        if len(index) != len(self._grid):
            raise IndexError(
                f"expected {len(self._grid)} grid indices, got {len(index)}"
            )
        offset: IntLike = self._offset
        sizes = list(self._tile_sizes)
        for i, (idx, a) in enumerate(zip(index, self._grid)):
            idx = sint(idx)
            if is_sym(idx) or is_sym(a.steps):
                require(idx >= 0, f"grid index {i} must be >= 0")
                require(idx < a.steps, f"grid index {i} exceeds the grid")
            elif not -a.steps <= idx < a.steps:
                raise IndexError(
                    f"grid index {idx} out of range for grid axis {i} of {a.steps} steps"
                )
            elif idx < 0:
                idx += a.steps
            plain = (
                not is_sym(a.step)
                and not is_sym(a.repeat)
                and a.step == 1
                and a.repeat == 1
            )
            first = (
                idx if plain else (idx // a.step) * (a.step * a.repeat) + idx % a.step
            )
            offset = offset + first * a.stride
            if self._partial and a.rep_pos is not None:
                sizes[a.rep_pos] = smin(a.repeat, sceildiv(a.tiles - first, a.step))
        strides = list(self._tile_strides)
        # A group of exactly one tile is no repeat: leave it out, so the tile
        # keeps only dimensions that step. Tile dimensions stay even when they
        # are 1 (a 1-row tile is still a tile).
        drop = [
            a.rep_pos
            for a in self._grid
            if a.rep_pos is not None
            and not is_sym(sizes[a.rep_pos])
            and sizes[a.rep_pos] == 1
        ]
        if drop:
            sizes = [v for i, v in enumerate(sizes) if i not in drop]
            strides = [v for i, v in enumerate(strides) if i not in drop]
        return TensorAccessPattern._raw(self._tensor_dims, offset, sizes, strides)

    def _tile_at(self, step: IntLike) -> TensorAccessPattern:
        # The tile at a linear step, following order().
        step = sint(step)
        shape = self.grid_shape
        index: list[IntLike] = [0] * len(self._grid)
        rest = step
        # Fastest-varying grid axis last in `order`.
        for pos in reversed(range(len(self._grid))):
            dim = self._order[pos]
            n = shape[dim]
            if pos == 0:
                index[dim] = rest
            else:
                index[dim] = rest % n
                rest = rest // n
        return self._at(*index)

    @property
    def _taps(self) -> list[TensorAccessPattern]:
        # Every tile, for the TensorAccessSequence methods that walk them all.
        if self._cached_taps is None:
            self._cached_taps = [self._tile_at(step) for step in range(len(self))]
        return self._cached_taps

    def __len__(self) -> int:
        n = self.num_steps
        if is_sym(n):
            raise TypeError(
                "len() of a runtime-sized TileGrid; use .num_steps (a staged value) instead"
            )
        return int(n)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            return self._at(*key)
        if isinstance(key, slice):
            return self._taps[key]
        return self._tile_at(key)

    def __iter__(self) -> Iterator[TensorAccessPattern]:
        for step in range(len(self)):
            yield self._tile_at(step)

    def __contains__(self, tap: object) -> bool:
        return tap in self._taps

    def _immutable(self) -> TypeError:
        return TypeError(
            "a TileGrid cannot be edited; build a TensorAccessSequence.from_taps(list(grid)) to edit"
        )

    def __setitem__(self, idx: Any, tap: Any) -> None:
        raise self._immutable()

    def __delitem__(self, idx: Any) -> None:
        raise self._immutable()

    def insert(self, index: int, value: TensorAccessPattern) -> None:
        raise self._immutable()

    # ------------------------------------------------------------ refinements

    def _replace(self, **kw: Any) -> TileGrid:
        args: dict[str, Any] = dict(
            tensor_dims=self._tensor_dims,
            offset=self._offset,
            grid=self._grid,
            tile_sizes=self._tile_sizes,
            tile_strides=self._tile_strides,
            order=self._order,
            partial=self._partial,
            tile_axes=self._tile_axes,
        )
        args.update(kw)
        return TileGrid(**args)

    def order(self, order: str | Sequence[int]) -> TileGrid:
        """Set the step order over the grid axes.

        Args:
            order (str | Sequence[int]): `"row"` (the default order), `"col"`,
                or a permutation of the grid axes, slowest-varying first.
                `"col"` walks down a column of tiles before moving to the
                next column.
        """
        if isinstance(order, str):
            if order == "row":
                perm: Sequence[int] = range(len(self._grid))
            elif order == "col":
                perm = range(len(self._grid) - 1, -1, -1)
            else:
                raise ValueError(
                    f"order must be 'row', 'col' or a permutation, got {order!r}"
                )
        else:
            perm = order
        return self._replace(order=perm)

    def permute_tile(self, axes: Sequence[int]) -> TileGrid:
        """Reorder the dimensions inside each tile, leaving the grid alone.

        Args:
            axes (Sequence[int]): A permutation of the tile dimensions.
        """
        axes = validate_permutation(axes, len(self._tile_sizes), "axes")
        remap = {old: new for new, old in enumerate(axes)}
        grid = [
            a if a.rep_pos is None else a._replace(rep_pos=remap[a.rep_pos])
            for a in self._grid
        ]
        return self._replace(
            grid=grid,
            tile_sizes=[self._tile_sizes[a] for a in axes],
            tile_strides=[self._tile_strides[a] for a in axes],
            tile_axes=[self._tile_axes[a] for a in axes],
        )

    def repeat(self, count: IntLike) -> TileGrid:
        """Walk each tile `count` times (a stride-0 outermost tile dimension).

        Args:
            count (IntLike): Number of walks per tile; must be >= 1.
        """
        count = sint(count)
        require(count >= 1, f"repeat count must be >= 1, got {show(count)}")
        grid = [
            a if a.rep_pos is None else a._replace(rep_pos=a.rep_pos + 1)
            for a in self._grid
        ]
        return self._replace(
            grid=grid,
            tile_sizes=[count] + self._tile_sizes,
            tile_strides=[0] + self._tile_strides,
            tile_axes=(-1,) + self._tile_axes,
        )

    def group(
        self,
        repeats: Sequence[IntLike],
        steps: Sequence[IntLike] | None = None,
        order: str = "row",
        partial: bool = False,
    ) -> TileGrid:
        """Gather `repeats[i]` tiles spaced `steps[i]` tiles apart into each step.

        Along grid axis `i` with `G` tiles, a block is `S * R` tiles
        (`S = steps[i]`, `R = repeats[i]`); within a block, group `j`
        (`0 <= j < S`) takes tiles `j, j + S, ..., j + (R - 1) S`. Steps
        along the axis enumerate blocks then groups, and each step's tile
        gains a leading repeat dimension of `R` striding `S` tiles.

        Args:
            repeats (Sequence[IntLike]): Tiles per group, one entry per grid axis.
            steps (Sequence[IntLike] | None, optional): Tile distance between group
                members, one entry per grid axis. Defaults to contiguous groups.
            order (str, optional): `"row"` walks each group's repeat
                dimensions in grid-axis order; `"col"` walks them reversed,
                i.e. all repeats along the last axis before advancing the
                first, as `TileGrid.order` does for steps. Defaults to `"row"`.
            partial (bool, optional): Allow an incomplete block at the tensor
                edge. `R` is first capped at `ceildiv(G, S)`, and each step's
                repeat size is `min(R, ceildiv(G - first, S))`. Otherwise `G`
                must be divisible by `S * R`. Defaults to False.
        """
        if self.is_grouped:
            raise ValueError("this TileGrid is already grouped")
        if order not in ("row", "col"):
            raise ValueError(f"order must be 'row' or 'col', got {order!r}")
        ng = len(self._grid)
        repeats = [sint(r) for r in repeats]
        steps = [1] * ng if steps is None else [sint(s) for s in steps]
        if len(repeats) != ng or len(steps) != ng:
            raise ValueError(f"repeats and steps need {ng} entries")
        rep_sizes: list[IntLike] = []
        rep_strides: list[IntLike] = []
        grid: list[_GridAxis] = []
        for i, a in enumerate(self._grid):
            g, gs = a.tiles, a.stride
            s, r = steps[i], repeats[i]
            require(s >= 1, f"steps[{i}] must be >= 1")
            require(r >= 1, f"repeats[{i}] must be >= 1")
            # A step wider than the axis degenerates to 1.
            s = (
                sselect(s > g, 1, s)
                if (is_sym(s) or is_sym(g))
                else (1 if s > g else s)
            )
            if partial:
                r = smin(r, sceildiv(g, s))
                blocks = g // (s * r)
                left = g - blocks * s * r
                n_steps = blocks * s + smin(left, s)
            else:
                require(
                    g % (s * r) == 0,
                    f"grid axis {i} of {show(g)} tiles is not divisible by steps*repeats = "
                    f"{show(s)}*{show(r)}; pass partial=True to allow a ragged edge",
                )
                n_steps = (g // (s * r)) * s
            grid.append(_GridAxis(n_steps, gs, g, s, r, i))
            rep_sizes.append(r)
            rep_strides.append(gs * s)
        if order == "col":
            rep_sizes.reverse()
            rep_strides.reverse()
            grid = [a._replace(rep_pos=ng - 1 - i) for i, a in enumerate(grid)]
        return self._replace(
            grid=grid,
            tile_sizes=rep_sizes + self._tile_sizes,
            tile_strides=rep_strides + self._tile_strides,
            partial=partial,
            tile_axes=(-1,) * ng + self._tile_axes,
        )

    def inverse(self) -> TensorAccessPattern:
        """Return the walk that reads a tile-blocked buffer back in logical row-major order.

        If this grid tiles an `(m, n)` view into `(r, t)` tiles, a buffer
        that stores those tiles one after another, in step order and each
        tile contiguous, is read in logical row-major order by the returned
        pattern: sizes `[m//r, r, n//t, t]` with strides `[r*n, t, r*t, 1]`.
        This is the "un-blocking" walk a memtile applies to a core's blocked
        output. The pattern walks the blocked buffer, so its `tensor_dims`
        are the view's shape `(m, n)`. `order` and `permute_tile` are
        honoured.

        Raises:
            ValueError: If the grid is grouped or repeated.
        """
        ng = len(self._grid)
        if self.is_grouped or len(self._tile_sizes) != ng:
            raise ValueError(
                "inverse() is only defined on a plain tile grid (no group/repeat)"
            )
        tile_of_axis = [self._tile_axes.index(i) for i in range(ng)]
        blocked = TensorAccessPattern.full(
            [self._grid[p].tiles for p in self._order] + self._tile_sizes
        )
        interleave = [
            x for i in range(ng) for x in (self._order.index(i), ng + tile_of_axis[i])
        ]
        out = blocked.permute(interleave)
        view_dims = [
            a.tiles * self._tile_sizes[t] for a, t in zip(self._grid, tile_of_axis)
        ]
        return TensorAccessPattern._raw(view_dims, 0, out.sizes, out.strides)

    def __repr__(self) -> str:
        return (
            f"TileGrid(grid={show(self.grid_shape)}, tile={show(self.tile_shape)}, "
            f"order={self._order}, partial={self._partial}, offset={show(self._offset)})"
        )
