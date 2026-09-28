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
from typing import Any, Iterator, NamedTuple, Sequence

import numpy as np

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
from .tap import TensorAccessPattern

__all__ = ["Layout", "PaddedLayout", "TileGrid"]

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
            require(d >= 1, f"tensor dimensions must be >= 1, got {show(tensor_dims)}")
        for s in sizes:
            require(s >= 1, f"sizes must be >= 1, got {show(sizes)}")
        for s in strides:
            require(s >= 0, f"strides must be >= 0, got {show(strides)}")
        require(offset >= 0, f"offset must be >= 0, got {show(offset)}")
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
        require(count >= 1, f"repeat count must be >= 1, got {show(count)}")
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

        Dimension ``i`` of size ``n_i`` becomes a grid axis of ``n_i // t_i``
        tiles (striding ``t_i`` times this view's stride) and a tile dimension
        of ``t_i`` elements. The result is a :class:`TileGrid`, indexable by
        tile. ``n_i`` must be divisible by ``t_i``.
        """
        tile_dims = [sint(t) for t in tile_dims]
        if len(tile_dims) != self.rank:
            raise ValueError(
                f"tile_dims has {len(tile_dims)} entries for a view of rank {self.rank}"
            )
        grid: list[_GridAxis] = []
        for dim, t in enumerate(tile_dims):
            n, s = self._sizes[dim], self._strides[dim]
            require(t >= 1, f"tile_dims[{dim}] must be >= 1")
            require(
                n % t == 0,
                f"dimension {dim} of size {n} is not divisible by tile size {t}",
            )
            grid.append(_GridAxis(n // t, s * t, n // t, 1, 1, None))
        return TileGrid(
            self._tensor_dims, self._offset, grid, list(tile_dims), list(self._strides)
        )

    def partition(self, parts: IntLike, dim: int = -1) -> TileGrid:
        """Split dimension ``dim`` into ``parts`` equal contiguous pieces.

        ``Layout.full((1, N)).partition(k)[i]`` is the ``i``-th of ``k`` equal
        chunks of a flat range: the pattern every per-column or per-channel
        design writes by hand as ``[1, 1, 1, N // k]`` with offset
        ``i * N // k``. Other dimensions are kept whole, so the grid has
        exactly ``parts`` steps.
        """
        dim = self._axis(dim)
        parts = sint(parts)
        require(parts >= 1, "parts must be >= 1")
        require(
            self._sizes[dim] % parts == 0,
            f"dimension {dim} of size {self._sizes[dim]} is not divisible into {parts} parts",
        )
        tile_dims = list(self._sizes)
        tile_dims[dim] = self._sizes[dim] // parts
        return self.tile(tile_dims)

    # ------------------------------------------------------------- conversions

    def stream_dims(self) -> list[tuple[IntLike, IntLike]]:
        """``[(size, stride), ...]`` as ObjectFifo ``dims_to_stream``/``dims_from_stream`` take it."""
        return list(zip(self._sizes, self._strides))

    def pad(self, padding: Sequence[Sequence[int]]) -> PaddedLayout:
        """Surround every walk of each dimension with constant elements.

        A memtile MM2S channel can pad the stream it emits: for dimension
        ``i`` it inserts ``before`` constant elements ahead of each pass over
        the dimension and ``after`` behind it, so the padded stream is
        ``prod(before_i + size_i + after_i)`` elements long (the pad value is
        set per channel on the fifo). ``padding`` is one ``(before, after)``
        pair per dimension of this layout, outermost first, and every count
        is a compile-time int.

        Returns:
            PaddedLayout: This layout with its padding, whose
            :meth:`~PaddedLayout.stream_dims` and :meth:`~PaddedLayout.pad_dims`
            are what ``ObjectFifo(dims_to_stream=..., pad_dimensions=...)``
            take, and whose :meth:`~PaddedLayout.materialize` shows where the
            constants land.
        """
        return PaddedLayout(self, padding)

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
            if out.rank > ndims and not out.is_symbolic:
                out = out.drop_unit_dims()
            if out.rank > ndims:
                raise ValueError(
                    f"view of rank {out.rank} (sizes {out._sizes}) does not fit in "
                    f"{ndims} DMA dimensions; coalesce() or re-tile it"
                )
            pad = ndims - out.rank
            # Slot 0 of the shim form is the queue repeat. A leading stride-0
            # dimension whose size is not the constant 1 is that repeat, staged
            # or not: it stays in slot 0 and the padding goes between it and
            # the addressing dimensions, so [R, th, tw] becomes [R, 1, th, tw].
            is_repeat = (
                out.rank > 1
                and not is_sym(out._strides[0])
                and out._strides[0] == 0
                and (is_sym(out._sizes[0]) or out._sizes[0] != 1)
            )
            if pad and is_repeat:
                sizes = out._sizes[:1] + [1] * pad + out._sizes[1:]
                strides = out._strides[:1] + [0] * pad + out._strides[1:]
            else:
                sizes = [1] * pad + out._sizes
                strides = [0] * pad + out._strides
            # A unit dimension ahead of the first real one never steps, so its
            # stride is 0; the repeat slot is transparent to that scan, and a
            # staged size ends it (rank and unit-ness are structural).
            for i in range(1 if is_repeat else 0, len(sizes)):
                if is_sym(sizes[i]) or sizes[i] != 1:
                    break
                strides[i] = 0
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


class PaddedLayout:
    """A :class:`Layout` plus the constant padding a memtile emits around it.

    Built by :meth:`Layout.pad`. Padding is a property of the *emitting*
    DMA (a memtile ``dims_to_stream``), not of the layout's own elements:
    the layout still indexes the object it reads; the padded stream is what
    the consumer receives.
    """

    def __init__(self, layout: Layout, padding: Sequence[Sequence[int]]):
        pads = []
        for entry in padding:
            if len(entry) != 2:
                raise ValueError("each padding entry is a (before, after) pair")
            before, after = (sint(v) for v in entry)
            if is_sym(before) or is_sym(after):
                raise TypeError("padding counts must be compile-time ints")
            if before < 0 or after < 0:
                raise ValueError(f"padding counts must be >= 0, got {entry}")
            pads.append((before, after))
        if len(pads) != layout.rank:
            raise ValueError(
                f"padding has {len(pads)} entries for a layout of rank {layout.rank}"
            )
        self._layout = layout
        self._padding = tuple(pads)

    @property
    def layout(self) -> Layout:
        """The unpadded walk."""
        return self._layout

    @property
    def padding(self) -> tuple[tuple[int, int], ...]:
        """``(before, after)`` per dimension, outermost first."""
        return self._padding

    @property
    def padded_sizes(self) -> list[IntLike]:
        """Extent of each dimension on the padded stream."""
        return [b + s + a for (b, a), s in zip(self._padding, self._layout.sizes)]

    @property
    def numel(self) -> IntLike:
        """Elements on the padded stream (what the consuming object must hold)."""
        return sprod(self.padded_sizes)

    def stream_dims(self) -> list[tuple[IntLike, IntLike]]:
        """Return the layout's ``dims_to_stream``."""
        return self._layout.stream_dims()

    def pad_dims(self) -> list[tuple[int, int]]:
        """``pad_dimensions`` as ``ObjectFifo`` takes it (one pair per dim)."""
        return [tuple(p) for p in self._padding]

    def materialize(self, pad_value: int = -1):
        """Flat object index of every element of the padded stream.

        Returns a NumPy array shaped :attr:`padded_sizes`; padded positions
        hold ``pad_value``. Needs a concrete layout.
        """
        import numpy as np

        lay = self._layout
        if lay.is_symbolic:
            raise TypeError("materialize needs a concrete layout")
        idx = np.zeros((), dtype=np.int64) + int(lay.offset)
        for size, stride in zip(lay.sizes, lay.strides):
            idx = idx[..., None] + np.arange(int(size), dtype=np.int64) * int(stride)
        return np.pad(idx, self._padding, constant_values=pad_value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PaddedLayout)
            and self._layout == other._layout
            and self._padding == other._padding
        )

    def __hash__(self) -> int:
        return hash((self._layout, self._padding))

    def __repr__(self) -> str:
        return f"{self._layout!r}.pad({list(self._padding)!r})"


class _GridAxis(NamedTuple):
    """One grid axis of a :class:`TileGrid`.

    ``steps`` grid positions index ``tiles`` tiles that lie ``stride``
    elements apart. After :meth:`TileGrid.group`, position ``p`` names the
    group's first tile ``(p // step) * step * repeat + p % step`` and the
    tile carries a repeat dimension of nominal size ``repeat`` at index
    ``rep_pos`` among the tile dimensions.
    """

    steps: IntLike
    stride: IntLike
    tiles: IntLike
    step: IntLike
    repeat: IntLike
    rep_pos: int | None


class TileGrid:
    """A tiling: grid axes that index tiles, and the tile's own dimensions.

    ``grid[i, j]`` (or ``grid[step]`` with a linearised step index) is the
    :class:`Layout` of one tile. Indices may be staged runtime values, in which
    case the tile's offset is staged arithmetic; that is what lets one compiled
    runtime sequence walk a runtime-sized tensor.

    Created by :meth:`Layout.tile` and :meth:`Layout.partition`; refined by
    :meth:`group`, :meth:`order`, :meth:`permute_tile` and :meth:`repeat`.
    """

    __slots__ = (
        "_tensor_dims",
        "_offset",
        "_grid",
        "_tile_sizes",
        "_tile_strides",
        "_order",
        "_partial",
    )

    def __init__(
        self,
        tensor_dims: Sequence[IntLike],
        offset: IntLike,
        grid: Sequence[_GridAxis],
        tile_sizes: Sequence[IntLike],
        tile_strides: Sequence[IntLike],
        order: Sequence[int] | None = None,
        partial: bool = False,
    ):
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
            else _check_perm(order, len(grid), "order")
        )
        self._partial = bool(partial)

    # --------------------------------------------------------------- shape

    @property
    def n_grid(self) -> int:
        """Number of grid axes (one per tensor dimension)."""
        return len(self._grid)

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
        return list(self._tile_strides)

    @property
    def num_steps(self) -> IntLike:
        """Number of tiles; staged when the grid is runtime-sized."""
        return sprod(self.grid_shape)

    @property
    def is_grouped(self) -> bool:
        return any(a.rep_pos is not None for a in self._grid)

    @property
    def is_symbolic(self) -> bool:
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
    def layout(self) -> Layout:
        """The whole grid as one view: grid axes in step order, then the tile.

        Only an ungrouped grid is a single strided view. With
        ``order("col")`` the grid axes come column-major, so the view walks
        down a column of tiles before moving to the next column.
        """
        if self.is_grouped:
            raise ValueError(
                "a grouped TileGrid is not a single strided view; index it instead"
            )
        axes = [self._grid[p] for p in self._order]
        return Layout(
            self._tensor_dims,
            self._offset,
            [a.tiles for a in axes] + self._tile_sizes,
            [a.stride for a in axes] + self._tile_strides,
        )

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
        if len(index) != self.n_grid:
            raise IndexError(f"expected {self.n_grid} grid indices, got {len(index)}")
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
        # A repeat of exactly one tile is no repeat: leave it out, so the tile
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
        return Layout(self._tensor_dims, offset, sizes, strides)

    def tile_at(self, step: IntLike) -> Layout:
        """Return the tile at linearised ``step``, following :meth:`order`.

        Delinearisation is ``//`` and ``%`` over the grid shape, so a staged
        ``step`` (a ``range_`` induction variable, for example) yields a tile
        whose offset is staged arithmetic.
        """
        step = sint(step)
        shape = self.grid_shape
        index: list[IntLike] = [0] * self.n_grid
        rest = step
        # Fastest-varying grid axis last in ``order``.
        for pos in reversed(range(self.n_grid)):
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

    def _replace(self, **kw) -> TileGrid:
        args = dict(
            tensor_dims=self._tensor_dims,
            offset=self._offset,
            grid=self._grid,
            tile_sizes=self._tile_sizes,
            tile_strides=self._tile_strides,
            order=self._order,
            partial=self._partial,
        )
        args.update(kw)
        return TileGrid(**args)

    def order(self, order: str | Sequence[int]) -> TileGrid:
        """Set the step order over the grid axes: ``"row"`` (default), ``"col"`` or a permutation.

        The permutation lists grid axes slowest-varying first. ``"col"``
        reverses the default, so on a 2-D grid steps walk down a column of
        tiles before moving to the next column.
        """
        if isinstance(order, str):
            if order == "row":
                perm: Sequence[int] = range(self.n_grid)
            elif order == "col":
                perm = range(self.n_grid - 1, -1, -1)
            else:
                raise ValueError(
                    f"order must be 'row', 'col' or a permutation, got {order!r}"
                )
        else:
            perm = order
        return self._replace(order=perm)

    def permute_tile(self, axes: Sequence[int]) -> TileGrid:
        """Reorder the tile dimensions (the walk inside each tile), leaving the grid alone."""
        axes = _check_perm(axes, len(self._tile_sizes), "axes")
        remap = {old: new for new, old in enumerate(axes)}
        grid = [
            a if a.rep_pos is None else a._replace(rep_pos=remap[a.rep_pos])
            for a in self._grid
        ]
        return self._replace(
            grid=grid,
            tile_sizes=[self._tile_sizes[a] for a in axes],
            tile_strides=[self._tile_strides[a] for a in axes],
        )

    def repeat(self, count: IntLike) -> TileGrid:
        """Walk each tile ``count`` times (a stride-0 outermost tile dimension)."""
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
        )

    def group(
        self,
        repeats: Sequence[IntLike],
        steps: Sequence[IntLike] | None = None,
        col_major: bool = False,
        partial: bool = False,
    ) -> TileGrid:
        """Gather ``repeats[i]`` tiles spaced ``steps[i]`` tiles apart into each step.

        Along grid axis ``i`` with ``G`` tiles, a block is ``S * R`` tiles
        (``S = steps[i]``, ``R = repeats[i]``); within a block, group ``j``
        (``0 <= j < S``) takes tiles ``j, j + S, ..., j + (R - 1) S``. Steps
        along the axis enumerate blocks then groups, and each step's tile
        gains a leading repeat dimension of ``R`` striding ``S`` tiles. With
        ``steps`` omitted every step is a contiguous ``R``-tile group.
        ``col_major`` walks the repeat dimensions innermost-first, i.e. all
        repeats along the last axis before advancing the first.

        With ``partial=False`` (default) ``G`` must be divisible by ``S * R``.
        With ``partial=True`` the tensor edge may hold an incomplete block:
        ``R`` is first capped at ``ceildiv(G, S)``, the last block's groups
        get the tiles that remain, and each step's repeat size is
        ``min(R, ceildiv(G - first, S))``. That is what ``allow_partial``
        meant on the old tiler, and it stays branch-free on staged values.
        """
        if self.is_grouped:
            raise ValueError("this TileGrid is already grouped")
        ng = self.n_grid
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
                    f"grid axis {i} of {g} tiles is not divisible by steps*repeats = {s}*{r}; "
                    "pass partial=True to allow a ragged edge",
                )
                n_steps = (g // (s * r)) * s
            grid.append(_GridAxis(n_steps, gs, g, s, r, i))
            rep_sizes.append(r)
            rep_strides.append(gs * s)
        if col_major:
            rep_sizes.reverse()
            rep_strides.reverse()
            grid = [a._replace(rep_pos=ng - 1 - a.rep_pos) for a in grid]
        return self._replace(
            grid=grid,
            tile_sizes=rep_sizes + self._tile_sizes,
            tile_strides=rep_strides + self._tile_strides,
            partial=partial,
        )

    def inverse(self) -> Layout:
        """Return the walk that reads a tile-blocked buffer back in logical row-major order.

        If this grid tiles a row-major ``(m, n)`` tensor into ``(r, t)`` tiles,
        a buffer that stores those tiles one after another (each tile
        contiguous) is read in logical row-major order by the returned view:
        sizes ``[m//r, r, n//t, t]`` with strides ``[r*n, t, r*t, 1]``. This is
        the "un-blocking" layout a memtile applies to a core's blocked output,
        and the inverse of :meth:`Layout.tile` up to that buffer's storage.

        Only defined on a grid straight from :meth:`Layout.tile` (no grouping,
        no repeat, no tile permutation).
        """
        ng = self.n_grid
        if self.is_grouped or len(self._tile_sizes) != ng:
            raise ValueError(
                "inverse() is only defined on a plain tile grid (no group/repeat)"
            )
        blocked = Layout.full([a.tiles for a in self._grid] + self._tile_sizes)
        interleave = [x for i in range(ng) for x in (i, ng + i)]
        out = blocked.permute(interleave)
        return Layout(self._tensor_dims, 0, out._sizes, out._strides)

    def __repr__(self) -> str:
        return (
            f"TileGrid(grid={self.grid_shape}, tile={self.tile_shape}, "
            f"order={self._order}, partial={self._partial}, offset={self._offset})"
        )
