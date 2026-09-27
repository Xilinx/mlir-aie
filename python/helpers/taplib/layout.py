# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A layout algebra for tensor access patterns.

A :class:`Layout` is a strided view over a tensor: an element offset plus
parallel ``sizes`` and ``strides`` lists, outermost dimension first, in
elements. That is exactly what a DMA buffer descriptor executes and exactly
what :class:`~aie.helpers.taplib.TensorAccessPattern` stores, so a ``Layout``
converts to one with :meth:`Layout.tap` and to ObjectFifo ``dims_to_stream`` /
``dims_from_stream`` lists with :meth:`Layout.stream_dims`.

Every operation below is a pure function of the view's integers plus a
structural choice (which dimension, which permutation). Tiling, grouping,
transposing, slicing and repeating are therefore compositions of a few
primitives, and the same code runs on Python ints at generation time and on
staged runtime values (see :mod:`.symbolic`) inside a dynamic runtime
sequence. Only three kinds of decision ever inspect a value, and each goes
through a helper that stays branch-free when the value is staged: a minimum
(:func:`~.symbolic.smin`), a divisibility or bounds check
(:func:`~.symbolic.require`) and unit-dimension cleaning
(:meth:`Layout.coalesce`, which is generation-time only).

Conventions: all dims lists are outermost-first, all units are elements of the
viewed tensor, ranks and permutations are always concrete Python ints.
"""

from __future__ import annotations

import operator
from typing import Any, Iterator, Sequence

import numpy as np

from .symbolic import is_sym, require, sceildiv, sint, sprod, sym_any
from .tap import TensorAccessPattern

__all__ = ["Layout", "TileGrid"]

IntLike = Any  # int, np.integer, or a staged aie.ir.Value


def _c_strides(dims: Sequence[IntLike]) -> list[IntLike]:
    """Row-major strides for a tensor of shape ``dims``."""
    strides: list[IntLike] = [1] * len(dims)
    for axis in range(len(dims) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * dims[axis + 1]
    return strides


def _check_perm(axes: Sequence[int], rank: int, what: str) -> tuple[int, ...]:
    axes = tuple(int(a) for a in axes)
    if sorted(axes) != list(range(rank)):
        raise ValueError(f"{what} must be a permutation of range({rank}), got {axes}")
    return axes


class Layout:
    """A strided view: ``offset`` plus outermost-first ``sizes`` and ``strides``.

    Construct one with :meth:`full` for a whole row-major tensor, or directly
    from an offset and dims lists. Instances are immutable; every method
    returns a new ``Layout`` (or a :class:`TileGrid`).
    """

    __slots__ = ("_tensor_dims", "_offset", "_sizes", "_strides")

    def __init__(
        self,
        tensor_dims: Sequence[IntLike],
        offset: IntLike,
        sizes: Sequence[IntLike],
        strides: Sequence[IntLike],
    ):
        """Create a view.

        Args:
            tensor_dims (Sequence[IntLike]): Shape of the tensor the view indexes.
            offset (IntLike): Element offset of the first element visited.
            sizes (Sequence[IntLike]): Extent of each dimension, outermost first.
            strides (Sequence[IntLike]): Element step of each dimension, outermost first.

        Raises:
            ValueError: If the rank is zero, ``sizes`` and ``strides`` differ in
                length, or a concrete value is out of range.
        """
        tensor_dims = [sint(d) for d in tensor_dims]
        sizes = [sint(s) for s in sizes]
        strides = [sint(s) for s in strides]
        offset = sint(offset)
        if len(tensor_dims) == 0:
            raise ValueError("a Layout needs at least one tensor dimension")
        if len(sizes) == 0:
            raise ValueError("a Layout needs at least one dimension")
        if len(sizes) != len(strides):
            raise ValueError(
                f"len(sizes) ({len(sizes)}) != len(strides) ({len(strides)})"
            )
        for d in tensor_dims:
            require(d >= 1, f"tensor dimensions must be >= 1, got {tensor_dims}")
        for s in sizes:
            require(s >= 1, f"sizes must be >= 1, got {sizes}")
        for s in strides:
            require(s >= 0, f"strides must be >= 0, got {strides}")
        require(offset >= 0, f"offset must be >= 0, got {offset}")
        self._tensor_dims = tensor_dims
        self._offset = offset
        self._sizes = sizes
        self._strides = strides

    # ----------------------------------------------------------------- basics

    @classmethod
    def full(cls, tensor_dims: Sequence[IntLike]) -> Layout:
        """Return the row-major walk over a whole tensor of shape ``tensor_dims``."""
        dims = [sint(d) for d in tensor_dims]
        return cls(dims, 0, dims, _c_strides(dims))

    @classmethod
    def from_tap(cls, tap: TensorAccessPattern) -> Layout:
        """Return the view a :class:`TensorAccessPattern` describes."""
        return cls(tap.tensor_dims, tap.offset, tap.sizes, tap.strides)

    @property
    def tensor_dims(self) -> list[IntLike]:
        return list(self._tensor_dims)

    @property
    def offset(self) -> IntLike:
        return self._offset

    @property
    def sizes(self) -> list[IntLike]:
        return list(self._sizes)

    @property
    def strides(self) -> list[IntLike]:
        return list(self._strides)

    @property
    def rank(self) -> int:
        return len(self._sizes)

    @property
    def numel(self) -> IntLike:
        """Number of elements the view visits (repeats counted)."""
        return sprod(self._sizes)

    @property
    def is_symbolic(self) -> bool:
        """Whether any value of the view is staged."""
        return sym_any([self._offset, *self._sizes, *self._strides, *self._tensor_dims])

    def _with(self, offset=None, sizes=None, strides=None) -> Layout:
        return Layout(
            self._tensor_dims,
            self._offset if offset is None else offset,
            self._sizes if sizes is None else sizes,
            self._strides if strides is None else strides,
        )

    # ----------------------------------------------------------- structural ops

    def permute(self, axes: Sequence[int]) -> Layout:
        """Reorder dimensions: result dim ``i`` is this view's dim ``axes[i]``.

        This is how a transpose is expressed: ``Layout.full((M, K)).permute((1, 0))``
        walks the tensor column by column. Whether a shim DMA can execute it
        depends on the address-generation granule rule (the innermost stride
        times the element width must be a whole 32-bit word), which the DMA
        verifier and the dynamic lowering enforce.
        """
        axes = _check_perm(axes, self.rank, "axes")
        return self._with(
            sizes=[self._sizes[a] for a in axes],
            strides=[self._strides[a] for a in axes],
        )

    def split(self, dim: int, inner: IntLike) -> Layout:
        """Split dimension ``dim`` of size ``n`` into ``(n // inner, inner)``.

        The outer part strides by ``inner * stride``, the inner part keeps the
        original stride. ``n`` must be divisible by ``inner``.
        """
        dim = self._axis(dim)
        inner = sint(inner)
        n, s = self._sizes[dim], self._strides[dim]
        require(
            n % inner == 0, f"dimension {dim} of size {n} is not divisible by {inner}"
        )
        sizes = self._sizes[:dim] + [n // inner, inner] + self._sizes[dim + 1 :]
        strides = self._strides[:dim] + [s * inner, s] + self._strides[dim + 1 :]
        return self._with(sizes=sizes, strides=strides)

    def merge(self, dim: int) -> Layout:
        """Merge dimensions ``dim`` and ``dim + 1`` into one.

        Only legal when they are contiguous, i.e. ``strides[dim] ==
        sizes[dim + 1] * strides[dim + 1]``. Whether to merge is a structural
        decision, so it is only supported on concrete values.

        Raises:
            ValueError: If the two dimensions are not contiguous.
            TypeError: If either dimension is staged.
        """
        dim = self._axis(dim)
        if dim + 1 >= self.rank:
            raise ValueError(f"dimension {dim} has no successor to merge with")
        n0, s0, n1, s1 = (
            self._sizes[dim],
            self._strides[dim],
            self._sizes[dim + 1],
            self._strides[dim + 1],
        )
        if sym_any([n0, s0, n1, s1]):
            raise TypeError(
                "merge() is a structural decision; it needs concrete sizes/strides"
            )
        if n0 != 1 and s0 != n1 * s1:
            raise ValueError(
                f"dimensions {dim} and {dim + 1} are not contiguous "
                f"(stride {s0} != {n1} * {s1}); cannot merge"
            )
        sizes = self._sizes[:dim] + [n0 * n1] + self._sizes[dim + 2 :]
        strides = self._strides[:dim] + [s1] + self._strides[dim + 2 :]
        return self._with(sizes=sizes, strides=strides)

    def drop_unit_dims(self) -> Layout:
        """Remove every dimension of size 1 (keeping at least one dimension).

        Concrete values only: rank is structural.
        """
        if sym_any(self._sizes):
            raise TypeError("drop_unit_dims() needs concrete sizes")
        keep = [i for i, n in enumerate(self._sizes) if n != 1]
        if not keep:
            keep = [self.rank - 1]
        return self._with(
            sizes=[self._sizes[i] for i in keep],
            strides=[self._strides[i] for i in keep],
        )

    def coalesce(self) -> Layout:
        """Drop unit dimensions and merge every contiguous adjacent pair.

        The result visits the same elements in the same order with the fewest
        dimensions. Concrete values only.
        """
        out = self.drop_unit_dims()
        i = 0
        while i + 1 < out.rank:
            if out._strides[i] == out._sizes[i + 1] * out._strides[i + 1]:
                out = out.merge(i)
            else:
                i += 1
        return out

    def repeat(self, count: IntLike) -> Layout:
        """Walk the whole view ``count`` times: a new outermost dimension with stride 0.

        On a shim DMA the outermost dimension becomes the queue repeat; on a
        memtile it is a plain zero-stride dimension.
        """
        count = sint(count)
        require(count >= 1, f"repeat count must be >= 1, got {count}")
        return self._with(sizes=[count] + self._sizes, strides=[0] + self._strides)

    def slice(self, key: Any) -> Layout:
        """Restrict the view with NumPy basic indexing.

        ``key`` is an integer, a slice, ``Ellipsis`` or ``None``, or a tuple of
        those. An integer selects one index and removes the dimension; a slice
        keeps the dimension with a new start, extent and step; ``None`` inserts
        a size-1 dimension. Starts, stops and steps may be staged for slices
        with a positive concrete step; integer indices may be staged too.

        Raises:
            TypeError: For advanced (array or boolean) indexing.
            IndexError: For too many indices or an out-of-range concrete index.
            ValueError: For an empty slice or a non-positive step.
        """
        entries = key if isinstance(key, tuple) else (key,)
        ellipses = [i for i, k in enumerate(entries) if k is Ellipsis]
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        covered = sum(k is not Ellipsis and k is not None for k in entries)
        if covered > self.rank:
            raise IndexError(f"too many indices for a view of rank {self.rank}")
        at = ellipses[0] if ellipses else len(entries)
        fill = (slice(None),) * (self.rank - covered)
        entries = entries[:at] + fill + entries[at + 1 :]

        offset: IntLike = self._offset
        sizes: list[IntLike] = []
        strides: list[IntLike] = []
        axis = 0
        for entry in entries:
            if entry is None:
                sizes.append(1)
                strides.append(0)
                continue
            n, s = self._sizes[axis], self._strides[axis]
            axis += 1
            if isinstance(entry, slice):
                start, stop, step = entry.start, entry.stop, entry.step
                step = 1 if step is None else sint(step)
                if is_sym(step):
                    raise TypeError("a slice step must be a concrete integer")
                if step <= 0:
                    raise ValueError(f"slice step must be positive, got {step}")
                if start is None and stop is None and step == 1 and not is_sym(n):
                    sizes.append(n)
                    strides.append(s)
                    continue
                start = 0 if start is None else sint(start)
                stop = n if stop is None else sint(stop)
                if not sym_any([start, stop, n]):
                    start, stop, _ = slice(start, stop, step).indices(n)
                    extent = len(range(start, stop, step))
                    if extent <= 0:
                        raise ValueError(
                            f"slice {entry!r} selects no elements on a dimension of {n}"
                        )
                else:
                    require(
                        start >= 0, "slice start must be >= 0 on a runtime dimension"
                    )
                    require(stop <= n, "slice stop exceeds the dimension")
                    require(stop > start, "slice selects no elements")
                    extent = sceildiv(stop - start, step)
                offset = offset + start * s
                sizes.append(extent)
                strides.append(s * step)
                continue
            if isinstance(entry, (bool, np.bool_)):
                raise TypeError(
                    "boolean indexing is advanced indexing; a view is a strided walk"
                )
            if is_sym(entry):
                require(entry >= 0, "index must be >= 0 on a runtime dimension")
                require(entry < n, "index exceeds the dimension")
                offset = offset + entry * s
                continue
            try:
                i = operator.index(entry)
            except TypeError:
                raise TypeError(
                    f"index {entry!r} is advanced indexing; a view is a strided walk, "
                    "which only basic indexing describes."
                ) from None
            if is_sym(n):
                require(i >= 0, "negative indices need a concrete dimension")
                require(i < n, "index exceeds the dimension")
            elif not -n <= i < n:
                raise IndexError(f"index {i} is out of bounds for a dimension of {n}")
            elif i < 0:
                i += n
            offset = offset + i * s
        if not sizes:
            sizes, strides = [1], [1]
        return self._with(offset=offset, sizes=sizes, strides=strides)

    __getitem__ = slice

    def tile(self, tile_dims: Sequence[IntLike]) -> TileGrid:
        """Divide every dimension into tiles of ``tile_dims``.

        Dimension ``i`` of size ``n_i`` becomes a grid dimension of ``n_i //
        t_i`` tiles and a tile dimension of ``t_i`` elements; grid dimensions
        come first. The result is a :class:`TileGrid`, indexable by tile.
        """
        tile_dims = [sint(t) for t in tile_dims]
        if len(tile_dims) != self.rank:
            raise ValueError(
                f"tile_dims has {len(tile_dims)} entries for a view of rank {self.rank}"
            )
        out = self
        for dim in range(self.rank):
            out = out.split(2 * dim, tile_dims[dim])
        grid = [2 * i for i in range(self.rank)]
        inner = [2 * i + 1 for i in range(self.rank)]
        return TileGrid(out.permute(grid + inner), self.rank)

    # ------------------------------------------------------------- conversions

    def stream_dims(self) -> list[tuple[IntLike, IntLike]]:
        """``[(size, stride), ...]`` as ObjectFifo ``dims_to_stream``/``dims_from_stream`` take it."""
        return list(zip(self._sizes, self._strides))

    def tap(self, ndims: int | None = 4) -> TensorAccessPattern:
        """Return this view as a :class:`TensorAccessPattern`.

        With ``ndims`` (default 4, the shim DMA form), the view is left-padded
        with unit dimensions to that rank; a higher-rank view first drops its
        unit dimensions and is rejected if it still does not fit. ``None``
        keeps the exact rank.

        Raises:
            ValueError: If the view cannot be expressed in ``ndims`` dimensions.
        """
        out = self
        if ndims is not None:
            # Unit dimensions carry no addressing; the canonical shim form has
            # none of them except as padding. Rank is structural, so a staged
            # view keeps its dimensions where they are.
            if not sym_any(out._sizes):
                out = out.drop_unit_dims()
            if out.rank > ndims:
                raise ValueError(
                    f"view of rank {out.rank} (sizes {out._sizes}) does not fit in "
                    f"{ndims} DMA dimensions; coalesce() or re-tile it"
                )
            pad = ndims - out.rank
            if (
                pad
                and not is_sym(out._strides[0])
                and out._strides[0] == 0
                and out.rank > 1
            ):
                # Slot 0 of the shim form is the queue repeat: a pure repeat
                # stays there and the padding goes between it and the
                # addressing dimensions, so [R, th, tw] becomes [R, 1, th, tw].
                sizes = out._sizes[:1] + [1] * pad + out._sizes[1:]
                strides = out._strides[:1] + [0] * pad + out._strides[1:]
            else:
                sizes = [1] * pad + out._sizes
                strides = [0] * pad + out._strides
            out = out._with(sizes=sizes, strides=strides)
        return TensorAccessPattern(
            out._tensor_dims, out._offset, out._sizes, out._strides
        )

    # ------------------------------------------------------------------- misc

    def _axis(self, dim: int) -> int:
        dim = int(dim)
        if not -self.rank <= dim < self.rank:
            raise IndexError(f"dimension {dim} out of range for rank {self.rank}")
        return dim % self.rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        if self.is_symbolic or other.is_symbolic:
            raise TypeError("cannot compare symbolic layouts at generation time")
        return (
            self._tensor_dims == other._tensor_dims
            and self._offset == other._offset
            and self._sizes == other._sizes
            and self._strides == other._strides
        )

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self._tensor_dims),
                self._offset,
                tuple(self._sizes),
                tuple(self._strides),
            )
        )

    def __repr__(self) -> str:
        return (
            f"Layout({self._tensor_dims}, offset={self._offset}, "
            f"sizes={self._sizes}, strides={self._strides})"
        )


class TileGrid:
    """A :class:`Layout` whose leading ``n_grid`` dimensions index tiles.

    ``grid[i, j]`` (or ``grid[step]`` with a linearised step index) is the
    :class:`Layout` of one tile: the grid dimensions are fixed and the tile
    dimensions remain. Indices may be staged runtime values, in which case the
    tile's offset is staged arithmetic; that is what lets one compiled runtime
    sequence walk a runtime-sized tensor.

    Created by :meth:`Layout.tile`; refined by :meth:`group`, :meth:`order`,
    :meth:`permute_tile` and :meth:`repeat`.
    """

    __slots__ = ("_layout", "_n_grid", "_order", "_axes")

    def __init__(
        self,
        layout: Layout,
        n_grid: int,
        order: Sequence[int] | None = None,
        axes: Sequence[Sequence[int]] | None = None,
    ):
        n_grid = int(n_grid)
        if not 1 <= n_grid < layout.rank:
            raise ValueError(
                f"n_grid ({n_grid}) must leave at least one tile dimension of rank {layout.rank}"
            )
        self._layout = layout
        self._n_grid = n_grid
        self._order = (
            tuple(range(n_grid))
            if order is None
            else _check_perm(order, n_grid, "order")
        )
        # Which grid dimensions descend from each tensor axis (a group() turns
        # one axis into a (block, group) pair). "row"/"col" orders are stated
        # per tensor axis, so this is what they permute.
        self._axes = (
            tuple((i,) for i in range(n_grid))
            if axes is None
            else tuple(tuple(int(d) for d in a) for a in axes)
        )

    # --------------------------------------------------------------- shape

    @property
    def layout(self) -> Layout:
        """The underlying view (grid dimensions first, then tile dimensions)."""
        return self._layout

    @property
    def n_grid(self) -> int:
        return self._n_grid

    @property
    def grid_shape(self) -> list[IntLike]:
        return self._layout._sizes[: self._n_grid]

    @property
    def grid_strides(self) -> list[IntLike]:
        return self._layout._strides[: self._n_grid]

    @property
    def tile_shape(self) -> list[IntLike]:
        return self._layout._sizes[self._n_grid :]

    @property
    def tile_strides(self) -> list[IntLike]:
        return self._layout._strides[self._n_grid :]

    @property
    def num_steps(self) -> IntLike:
        """Number of tiles; staged when the grid is runtime-sized."""
        return sprod(self.grid_shape)

    @property
    def is_symbolic(self) -> bool:
        return self._layout.is_symbolic

    def __len__(self) -> int:
        n = self.num_steps
        if is_sym(n):
            raise TypeError(
                "len() of a runtime-sized TileGrid; use .num_steps (a staged value) instead"
            )
        return int(n)

    # ------------------------------------------------------------ indexing

    def at(self, *index: IntLike) -> Layout:
        """Return the tile at a multi-dimensional grid index."""
        if len(index) != self._n_grid:
            raise IndexError(f"expected {self._n_grid} grid indices, got {len(index)}")
        offset: IntLike = self._layout._offset
        for i, (idx, n, s) in enumerate(zip(index, self.grid_shape, self.grid_strides)):
            idx = sint(idx)
            if is_sym(idx) or is_sym(n):
                require(idx >= 0, f"grid index {i} must be >= 0")
                require(idx < n, f"grid index {i} exceeds the grid")
            elif not -n <= idx < n:
                raise IndexError(
                    f"grid index {idx} out of range for grid dimension {i} of {n}"
                )
            elif idx < 0:
                idx += n
            offset = offset + idx * s
        return self._layout._with(
            offset=offset, sizes=self.tile_shape, strides=self.tile_strides
        )

    def tile_at(self, step: IntLike) -> Layout:
        """Return the tile at linearised ``step``, following :meth:`order`.

        Delinearisation is ``//`` and ``%`` over the grid shape, so a staged
        ``step`` (a ``range_`` induction variable, for example) yields a tile
        whose offset is staged arithmetic.
        """
        step = sint(step)
        shape = self.grid_shape
        index: list[IntLike] = [0] * self._n_grid
        rest = step
        # Fastest-varying grid dimension last in ``order``.
        for pos in reversed(range(self._n_grid)):
            dim = self._order[pos]
            n = shape[dim]
            if pos == 0:
                index[dim] = rest
            else:
                index[dim] = rest % n
                rest = rest // n
        return self.at(*index)

    def __getitem__(self, key: Any) -> Layout:
        if isinstance(key, tuple):
            return self.at(*key)
        return self.tile_at(key)

    def __iter__(self) -> Iterator[Layout]:
        for step in range(len(self)):
            yield self.tile_at(step)

    def materialize(self):
        """Every tile in step order as a :class:`~aie.helpers.taplib.TensorAccessSequence`."""
        from .tas import TensorAccessSequence

        return TensorAccessSequence.from_taps([t.tap() for t in self])

    # --------------------------------------------------------- refinements

    def order(self, order: str | Sequence[int]) -> TileGrid:
        """Set the step order over the grid: ``"row"`` (default), ``"col"`` or a permutation.

        The permutation lists grid dimensions slowest-varying first. ``"col"``
        reverses the default, so for a 2-D grid steps walk down a column
        before moving to the next.
        """
        if isinstance(order, str):
            if order == "row":
                axes = self._axes
            elif order == "col":
                axes = tuple(reversed(self._axes))
            else:
                raise ValueError(
                    f"order must be 'row', 'col' or a permutation, got {order!r}"
                )
            perm: Sequence[int] = [d for a in axes for d in a]
        else:
            perm = order
        return TileGrid(self._layout, self._n_grid, perm, self._axes)

    def permute_tile(self, axes: Sequence[int]) -> TileGrid:
        """Reorder the tile dimensions (walk inside each tile), leaving the grid alone."""
        axes = _check_perm(axes, self._layout.rank - self._n_grid, "axes")
        full = list(range(self._n_grid)) + [self._n_grid + a for a in axes]
        return TileGrid(
            self._layout.permute(full), self._n_grid, self._order, self._axes
        )

    def permute_grid(self, axes: Sequence[int]) -> TileGrid:
        """Reorder the grid dimensions, leaving the tiles alone. Resets the step order."""
        axes = _check_perm(axes, self._n_grid, "axes")
        full = list(axes) + list(range(self._n_grid, self._layout.rank))
        return TileGrid(self._layout.permute(full), self._n_grid)

    def repeat(self, count: IntLike) -> TileGrid:
        """Walk each tile ``count`` times (a stride-0 outermost tile dimension)."""
        count = sint(count)
        require(count >= 1, f"repeat count must be >= 1, got {count}")
        lay = self._layout
        sizes = lay._sizes[: self._n_grid] + [count] + lay._sizes[self._n_grid :]
        strides = lay._strides[: self._n_grid] + [0] + lay._strides[self._n_grid :]
        return TileGrid(
            lay._with(sizes=sizes, strides=strides),
            self._n_grid,
            self._order,
            self._axes,
        )

    def group(
        self,
        repeats: Sequence[IntLike],
        steps: Sequence[IntLike] | None = None,
        col_major: bool = False,
    ) -> TileGrid:
        """Gather ``repeats[i]`` tiles spaced ``steps[i]`` tiles apart into each step.

        Along grid dimension ``i`` with ``G`` tiles, a block is ``S * R`` tiles
        (``S = steps[i]``, ``R = repeats[i]``); within a block, group ``j`` (for
        ``0 <= j < S``) takes tiles ``j, j + S, ..., j + (R - 1) S``. The grid
        becomes ``(G // (S R), S)`` per dimension, blocks then groups, and each
        step's tile gains a leading dimension of ``R`` repeats striding ``S``
        tiles. With ``steps`` omitted every step is a contiguous ``R``-tile
        group. ``col_major`` walks the repeat dimensions innermost-first, i.e.
        all repeats along the last grid dimension before advancing the first.

        ``G`` must be divisible by ``S * R``; ragged edges are not modelled.
        """
        ng = self._n_grid
        repeats = [sint(r) for r in repeats]
        steps = [1] * ng if steps is None else [sint(s) for s in steps]
        if len(repeats) != ng or len(steps) != ng:
            raise ValueError(f"repeats and steps need {ng} entries")
        lay = self._layout
        grid_sizes: list[IntLike] = []
        grid_strides: list[IntLike] = []
        rep_sizes: list[IntLike] = []
        rep_strides: list[IntLike] = []
        for i in range(ng):
            g, gs = lay._sizes[i], lay._strides[i]
            s, r = steps[i], repeats[i]
            require(s >= 1, f"steps[{i}] must be >= 1")
            require(r >= 1, f"repeats[{i}] must be >= 1")
            require(
                g % (s * r) == 0,
                f"grid dimension {i} of {g} tiles is not divisible by steps*repeats = {s}*{r}",
            )
            grid_sizes += [g // (s * r), s]
            grid_strides += [gs * s * r, gs]
            rep_sizes.append(r)
            rep_strides.append(gs * s)
        if col_major:
            rep_sizes.reverse()
            rep_strides.reverse()
        sizes = grid_sizes + rep_sizes + lay._sizes[ng:]
        strides = grid_strides + rep_strides + lay._strides[ng:]
        new_grid = 2 * ng
        # Keep the current step order, expanded so each old grid dimension's
        # (block, group) pair stays adjacent, block outermost.
        order = [x for d in self._order for x in (2 * d, 2 * d + 1)]
        axes = tuple(
            tuple(x for d in a for x in (2 * d, 2 * d + 1)) for a in self._axes
        )
        return TileGrid(lay._with(sizes=sizes, strides=strides), new_grid, order, axes)

    def inverse(self) -> Layout:
        """Return the walk that reads a tile-blocked buffer back in logical row-major order.

        If this grid tiles a row-major ``(m, n)`` tensor into ``(r, t)`` tiles,
        a buffer that stores those tiles one after another (each tile
        contiguous) is read in logical row-major order by the returned view:
        sizes ``[m//r, r, n//t, t]`` with strides ``[r*n, t, r*t, 1]``. This is
        the "un-blocking" layout a memtile applies to a core's blocked output,
        and the inverse of :meth:`Layout.tile` up to that buffer's storage.

        Only defined on a grid straight from :meth:`Layout.tile` (no grouping,
        no repeat).
        """
        ng = self._n_grid
        lay = self._layout
        if lay.rank != 2 * ng:
            raise ValueError(
                "inverse() is only defined on a plain tile grid (no group/repeat)"
            )
        blocked = Layout.full(lay._sizes)  # tiles stored consecutively, each contiguous
        interleave = [x for i in range(ng) for x in (i, ng + i)]
        out = blocked.permute(interleave)
        return Layout(lay._tensor_dims, 0, out._sizes, out._strides)

    def __repr__(self) -> str:
        return f"TileGrid(grid={self.grid_shape}, tile={self.tile_shape}, order={self._order}, layout={self._layout!r})"
