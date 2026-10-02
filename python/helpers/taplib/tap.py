# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tensor access patterns: strided walks over a tensor, and the algebra that builds them.

A `TensorAccessPattern` is an element offset plus parallel `sizes` and
`strides` lists, outermost dimension first, in elements of the tensor it
walks. That is exactly what a DMA buffer descriptor executes, so the same
object is what a runtime `fill`/`drain` takes and what an ObjectFifo takes
as `to_stream` / `from_stream`.

Start from `TensorAccessPattern.full()` (the row-major walk of a whole
tensor) and refine it: index it like a NumPy array, `permute()` or `T` it,
`repeat()` it, or `tile()` it into a grid of tiles. Every operation returns a
new pattern.

Every operation is integer arithmetic, so the same code runs on Python ints
at generation time and on staged runtime values inside a dynamic runtime
sequence. A divisibility or bounds check on a staged value becomes a
dispatch-time guard (`require()`).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import numpy as np

from ._symbolic import IntLike, is_sym, require, show, sint, sprod, sym_any
from .utils import (
    row_major_strides,
    validate_and_clean_sizes_strides,
    validate_offset,
    validate_permutation,
    validate_tensor_dims,
    zero_leading_unit_strides,
)

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation


class TensorAccessPattern:
    """A strided walk over a tensor: `offset` plus outermost-first `sizes` and `strides`.

    Build one with `full()` and refine it, or give the numbers directly.
    Instances are immutable; every method returns a new pattern.

    Two patterns are equal when they walk the same tensor the same way, i.e.
    when their `coalesce()` forms match. Staged patterns are equal only to
    themselves.
    """

    _DTYPE = np.int32

    def __init__(
        self,
        tensor_dims: Sequence[IntLike],
        offset: IntLike,
        sizes: Sequence[IntLike],
        strides: Sequence[IntLike],
    ):
        """Create an access pattern.

        Values may be Python ints or staged runtime values; a check on a staged
        value becomes a dispatch-time guard.

        Args:
            tensor_dims (Sequence[IntLike]): Shape of the tensor the pattern walks.
            offset (IntLike): Element offset of the first element visited.
            sizes (Sequence[IntLike]): Extent of each dimension, outermost first.
            strides (Sequence[IntLike]): Element step of each dimension, outermost first.

        Raises:
            TypeError: If a value is not an integer.
            ValueError: If `sizes` and `strides` differ in length or a
                concrete value is out of range.
        """
        tensor_dims = [sint(d) for d in tensor_dims]
        offset = sint(offset)
        sizes = [sint(s) for s in sizes]
        strides = [sint(s) for s in strides]
        self._tensor_dims = tuple(validate_tensor_dims(tensor_dims))
        self._offset = validate_offset(offset, tensor_dims)
        sizes, strides = validate_and_clean_sizes_strides(sizes, strides)
        self._sizes: tuple[IntLike, ...] = tuple(sizes)
        self._strides: tuple[IntLike, ...] = tuple(strides)
        self._padding: tuple[tuple[int, int], ...] | None = None

    @classmethod
    def _raw(
        cls,
        tensor_dims: Sequence[IntLike],
        offset: IntLike,
        sizes: Sequence[IntLike],
        strides: Sequence[IntLike],
        padding: tuple[tuple[int, int], ...] | None = None,
    ) -> TensorAccessPattern:
        # Build a pattern the algebra has already checked. Skipping the
        # constructor's checks keeps a staged walk from emitting the same
        # dispatch-time guards once per intermediate pattern.
        out = cls.__new__(cls)
        out._tensor_dims = tuple(tensor_dims)
        out._offset = offset
        out._sizes = tuple(sizes)
        out._strides = tuple(zero_leading_unit_strides(sizes, strides))
        out._padding = padding
        return out

    @classmethod
    def full(cls, tensor_dims: Sequence[IntLike]) -> TensorAccessPattern:
        """Return the row-major walk over a whole tensor of shape `tensor_dims`.

        Args:
            tensor_dims (Sequence[IntLike]): Shape of the tensor.

        Returns:
            TensorAccessPattern: A pattern that visits every element once, in order.
        """
        dims = [sint(d) for d in tensor_dims]
        return cls(dims, 0, dims, row_major_strides(dims))

    # ------------------------------------------------------------ properties

    @property
    def tensor_dims(self) -> tuple[IntLike, ...]:
        """Shape of the tensor the pattern walks."""
        return self._tensor_dims

    @property
    def offset(self) -> IntLike:
        """Element offset of the first element visited."""
        return self._offset

    @property
    def sizes(self) -> tuple[IntLike, ...]:
        """Extent of each dimension, outermost first."""
        return self._sizes

    @property
    def strides(self) -> tuple[IntLike, ...]:
        """Element step of each dimension, outermost first."""
        return self._strides

    @property
    def transformation_dims(self) -> tuple[tuple[IntLike, IntLike], ...]:
        """The pattern as `((size, stride), ...)`, outermost first."""
        return tuple(zip(self._sizes, self._strides))

    @property
    def padding(self) -> tuple[tuple[int, int], ...] | None:
        """`(before, after)` padding per dimension, or None if the pattern is unpadded."""
        return self._padding

    @property
    def padded_sizes(self) -> tuple[IntLike, ...]:
        """Extent of each dimension on the (possibly padded) stream."""
        if self._padding is None:
            return self._sizes
        return tuple(b + s + a for (b, a), s in zip(self._padding, self._sizes))

    @property
    def rank(self) -> int:
        """Number of dimensions of the walk."""
        return len(self._sizes)

    @property
    def numel(self) -> IntLike:
        """Number of tensor elements the walk visits (repeats counted, padding not)."""
        return sprod(self._sizes)

    @property
    def is_symbolic(self) -> bool:
        """Whether any value of the pattern is staged."""
        return sym_any([self._offset, *self._sizes, *self._strides, *self._tensor_dims])

    def _with(self, offset=None, sizes=None, strides=None) -> TensorAccessPattern:
        if self._padding is not None:
            raise ValueError(
                "a padded pattern cannot be reshaped further; apply pad() last"
            )
        return TensorAccessPattern._raw(
            self._tensor_dims,
            self._offset if offset is None else offset,
            self._sizes if sizes is None else sizes,
            self._strides if strides is None else strides,
        )

    def _axis(self, dim: int) -> int:
        dim = int(dim)
        if not -self.rank <= dim < self.rank:
            raise IndexError(f"dimension {dim} out of range for rank {self.rank}")
        return dim % self.rank

    # ------------------------------------------------------- reshaping the walk

    def permute(self, axes: Sequence[int]) -> TensorAccessPattern:
        """Reorder dimensions: result dimension `i` is this pattern's dimension `axes[i]`.

        Whether a shim DMA can execute the result depends on the
        address-generation granule rule (the innermost stride times the
        element width must be a whole 32-bit word), which the DMA verifier and
        the dynamic lowering enforce.

        Args:
            axes (Sequence[int]): A permutation of `range(rank)`.

        Returns:
            TensorAccessPattern: The reordered walk.

        Raises:
            ValueError: If `axes` is not a permutation of `range(rank)`.
        """
        axes = validate_permutation(axes, self.rank, "axes")
        return self._with(
            sizes=[self._sizes[a] for a in axes],
            strides=[self._strides[a] for a in axes],
        )

    @property
    def T(self) -> TensorAccessPattern:  # noqa: N802
        """The walk with its dimensions reversed, like `numpy.ndarray.T`.

        `TensorAccessPattern.full((M, N)).T` walks an `(M, N)` tensor
        column by column.
        """
        return self.permute(range(self.rank - 1, -1, -1))

    def split(self, dim: int, inner: IntLike) -> TensorAccessPattern:
        """Split dimension `dim` of size `n` into `(n // inner, inner)`.

        The outer part strides by `inner * stride`; the inner part keeps the
        original stride.

        Args:
            dim (int): The dimension to split.
            inner (IntLike): Size of the new inner dimension; must divide `n`.

        Returns:
            TensorAccessPattern: The walk with one more dimension.

        Raises:
            ValueError: If `inner` does not divide the dimension.
        """
        dim = self._axis(dim)
        inner = sint(inner)
        n, s = self._sizes[dim], self._strides[dim]
        require(
            n % inner == 0,
            f"dimension {dim} of size {show(n)} is not divisible by {show(inner)}",
        )
        sizes = (*self._sizes[:dim], n // inner, inner, *self._sizes[dim + 1 :])
        strides = (*self._strides[:dim], s * inner, s, *self._strides[dim + 1 :])
        return self._with(sizes=sizes, strides=strides)

    def merge(self, dim: int) -> TensorAccessPattern:
        """Merge dimensions `dim` and `dim + 1` into one.

        Only legal when they are contiguous, i.e. ``strides[dim] ==
        sizes[dim + 1] * strides[dim + 1]``. Whether to merge is a structural
        decision, so the values must be concrete.

        Args:
            dim (int): The outer of the two dimensions.

        Returns:
            TensorAccessPattern: The walk with one fewer dimension.

        Raises:
            ValueError: If the two dimensions are not contiguous.
            TypeError: If either dimension is staged.
        """
        dim = self._axis(dim)
        if dim + 1 >= self.rank:
            raise ValueError(f"dimension {dim} has no successor to merge with")
        n0, s0 = self._sizes[dim], self._strides[dim]
        n1, s1 = self._sizes[dim + 1], self._strides[dim + 1]
        if sym_any([n0, s0, n1, s1]):
            raise TypeError(
                "merge() is a structural decision; it needs concrete sizes/strides"
            )
        if n0 != 1 and s0 != n1 * s1:
            raise ValueError(
                f"dimensions {dim} and {dim + 1} are not contiguous "
                f"(stride {s0} != {n1} * {s1}); cannot merge"
            )
        sizes = (*self._sizes[:dim], n0 * n1, *self._sizes[dim + 2 :])
        strides = (*self._strides[:dim], s1, *self._strides[dim + 2 :])
        return self._with(sizes=sizes, strides=strides)

    def _drop_unit_dims(self) -> TensorAccessPattern:
        # Rank is structural, so only a size that is the constant 1 is
        # dropped; at least one dimension is kept.
        keep = [i for i, n in enumerate(self._sizes) if is_sym(n) or n != 1]
        if not keep:
            keep = [self.rank - 1]
        return self._with(
            sizes=[self._sizes[i] for i in keep],
            strides=[self._strides[i] for i in keep],
        )

    def coalesce(self) -> TensorAccessPattern:
        """Return the same walk in the fewest dimensions.

        Size-1 dimensions are dropped and every contiguous adjacent pair is
        merged; the result visits the same elements in the same order. A pair
        with a staged size or stride is kept. The DMA builders apply this
        themselves to a walk deeper than the hardware takes.

        Returns:
            TensorAccessPattern: The same walk with no size-1 or mergeable dimensions.
        """
        out = self._drop_unit_dims()
        i = 0
        while i + 1 < out.rank:
            pair = out._sizes[i : i + 2] + out._strides[i : i + 2]
            if not sym_any(pair) and (
                out._strides[i] == out._sizes[i + 1] * out._strides[i + 1]
            ):
                out = out.merge(i)
            else:
                i += 1
        return out

    def repeat(self, count: IntLike) -> TensorAccessPattern:
        """Walk the whole pattern `count` times: a new outermost dimension with stride 0.

        On a shim DMA the outermost dimension becomes the queue repeat; on a
        memtile it is a plain zero-stride dimension.

        Args:
            count (IntLike): Number of walks; must be >= 1.

        Returns:
            TensorAccessPattern: The walk with a new outermost dimension of `count`.

        Raises:
            ValueError: If `count` is below 1.
        """
        count = sint(count)
        require(count >= 1, f"repeat count must be >= 1, got {show(count)}")
        return self._with(sizes=(count, *self._sizes), strides=(0, *self._strides))

    def __getitem__(self, key: Any) -> TensorAccessPattern:
        """Restrict the walk with NumPy basic indexing.

        `key` is an integer, a slice, `Ellipsis` or `None`, or a tuple of
        those, applied to the pattern's dimensions exactly as NumPy applies it
        to an array's: an integer selects one index and removes the dimension,
        a slice keeps it with a new start, extent and step, and `None`
        inserts a size-1 dimension. `TensorAccessPattern.full(shape)[key]`
        walks exactly the elements of `np.zeros(shape)[key]`, in order.
        Starts, stops and integer indices may be staged; a slice step must be
        a positive Python int.

        Args:
            key: An integer, slice, `Ellipsis` or `None`, or a tuple of those.

        Returns:
            TensorAccessPattern: The restricted walk.

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
            raise IndexError(f"too many indices for a pattern of rank {self.rank}")
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
                if start is None and stop is None and step == 1:
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
                    extent = (stop - start - 1) // step + 1
                offset = offset + start * s
                sizes.append(extent)
                strides.append(s * step)
                continue
            if isinstance(entry, (bool, np.bool_)):
                raise TypeError(
                    "boolean indexing is advanced indexing; a pattern is a strided walk"
                )
            if is_sym(entry):
                entry = sint(entry)
                require(entry >= 0, "index must be >= 0 on a runtime dimension")
                require(entry < n, "index exceeds the dimension")
                offset = offset + entry * s
                continue
            try:
                i = operator.index(entry)
            except TypeError:
                raise TypeError(
                    f"index {entry!r} is advanced indexing; a pattern is a strided "
                    "walk, which only basic indexing -- integers, slices, Ellipsis "
                    "and None -- describes."
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

    def tile(self, tile_dims: Sequence[IntLike]) -> TensorAccessPattern:
        """Divide every dimension into tiles of `tile_dims`.

        Dimension `i` of size `n_i` becomes a grid dimension of `n_i // t_i`
        tiles and a tile dimension of `t_i` elements. The grid dimensions
        come first, so a rank-`r` pattern becomes a rank-`2r` one that walks
        the tiles in row-major order: `tiles[i, j]` is the tile at grid
        position `(i, j)` and `tiles[i]` the `i`-th row of tiles. Reorder
        the grid with `permute()` (`(1, 0, 2, 3)` walks a column of tiles at
        a time) and take strided or partial groups of tiles by slicing.

        Args:
            tile_dims (Sequence[IntLike]): One tile extent per dimension; each
                must divide the dimension.

        Returns:
            TensorAccessPattern: The grid dimensions followed by the tile dimensions.

        Raises:
            ValueError: If `tile_dims` does not have one entry per dimension,
                or an entry is below 1 or does not divide its dimension.
        """
        tile_dims = [sint(t) for t in tile_dims]
        if len(tile_dims) != self.rank:
            raise ValueError(
                f"tile_dims has {len(tile_dims)} entries for a pattern of rank {self.rank}"
            )
        grid_sizes: list[IntLike] = []
        grid_strides: list[IntLike] = []
        for dim, t in enumerate(tile_dims):
            n, s = self._sizes[dim], self._strides[dim]
            require(t >= 1, f"tile_dims[{dim}] must be >= 1")
            require(
                n % t == 0,
                f"dimension {dim} of size {show(n)} is not divisible by tile size {show(t)}",
            )
            grid_sizes.append(n // t)
            grid_strides.append(s * t)
        return self._with(
            sizes=(*grid_sizes, *tile_dims), strides=(*grid_strides, *self._strides)
        )

    def partition(self, parts: IntLike, dim: int = 0) -> TensorAccessPattern:
        """Split dimension `dim` into `parts` equal contiguous pieces.

        Like `np.array_split` on an evenly divisible axis, it splits the
        outermost dimension unless told otherwise. The result has a new
        outermost dimension of `parts`, so
        `TensorAccessPattern.full((N,)).partition(k)[i]` is the `i`-th of
        `k` equal chunks of a flat range. Other dimensions are kept whole.

        Args:
            parts (IntLike): Number of pieces; must divide the dimension.
            dim (int, optional): The dimension to split. Defaults to 0.

        Returns:
            TensorAccessPattern: The pieces, one per index of the new outermost dimension.

        Raises:
            ValueError: If `parts` is below 1 or does not divide the dimension.
        """
        dim = self._axis(dim)
        parts = sint(parts)
        n, s = self._sizes[dim], self._strides[dim]
        require(parts >= 1, "parts must be >= 1")
        require(
            n % parts == 0,
            f"dimension {dim} of size {show(n)} is not divisible into {show(parts)} parts",
        )
        chunk = n // parts
        sizes = list(self._sizes)
        sizes[dim] = chunk
        return self._with(sizes=(parts, *sizes), strides=(s * chunk, *self._strides))

    def inverse(self) -> TensorAccessPattern:
        """Return the walk that puts this walk's stream back in row-major order.

        A DMA that streams a tensor with a pattern visiting every element
        exactly once stores a buffer `B` with `B[k] = tensor.flat[walk[k]]`.
        The inverse walks `B` and emits the tensor in row-major order, so
        `p.inverse().gather(p.gather(x))` is `x.reshape(-1)`. For the
        tiling `full((m, n)).tile((r, t))` it is the "un-blocking" walk a
        memtile applies to a core's blocked output: sizes `[m//r, r, n//t, t]`
        with strides `[r*n, t, r*t, 1]`.

        Returns:
            TensorAccessPattern: The inverse walk, over a buffer of the same shape.

        Raises:
            TypeError: If the pattern is staged.
            ValueError: If the pattern is padded or does not visit every
                element of its tensor exactly once.
        """
        self._require_concrete("inverse()")
        if self._padding is not None:
            raise ValueError("a padded pattern has no inverse")
        steps = [i for i, n in enumerate(self._sizes) if n != 1]
        steps.sort(key=lambda i: self._strides[i], reverse=True)
        covered = 1
        for i in reversed(steps):
            if self._strides[i] != covered:
                covered = 0
                break
            covered *= self._sizes[i]
        if self._offset != 0 or covered != sprod(self._tensor_dims):
            raise ValueError(
                f"{self} does not visit every element exactly once, so it has no inverse"
            )
        if not steps:
            return TensorAccessPattern.full(self._tensor_dims)
        stream_strides = row_major_strides(self._sizes)
        return TensorAccessPattern._raw(
            self._tensor_dims,
            0,
            [self._sizes[i] for i in steps],
            [stream_strides[i] for i in steps],
        )

    def pad(self, padding: Sequence[Sequence[int]]) -> TensorAccessPattern:
        """Surround every walk of each dimension with constant elements.

        A memtile MM2S channel can pad the stream it emits: for dimension
        `i` it inserts `before` constant elements ahead of each pass over
        the dimension and `after` behind it, so the stream is
        `prod(padded_sizes)` elements long. Give a padded pattern to an
        ObjectFifo as `to_stream` and it sets the padding too (the pad value
        is set on the fifo). Padding is applied last: a padded pattern cannot
        be reshaped further.

        Args:
            padding (Sequence[Sequence[int]]): One `(before, after)` pair per
                dimension, outermost first, as compile-time ints.

        Returns:
            TensorAccessPattern: This walk with its padding.

        Raises:
            TypeError: If a padding count is staged.
            ValueError: If the pattern is already padded, an entry is not a
                pair of counts >= 0, or there is not one entry per dimension.
        """
        if self._padding is not None:
            raise ValueError("this pattern is already padded")
        pads = []
        for entry in padding:
            if len(entry) != 2:
                raise ValueError("each padding entry is a (before, after) pair")
            before, after = (sint(v) for v in entry)
            if is_sym(before) or is_sym(after):
                raise TypeError("padding counts must be compile-time ints")
            if before < 0 or after < 0:
                raise ValueError(f"padding counts must be >= 0, got {tuple(entry)}")
            pads.append((before, after))
        if len(pads) != self.rank:
            raise ValueError(
                f"padding has {len(pads)} entries for a pattern of rank {self.rank}"
            )
        if not any(b or a for b, a in pads):
            return self
        return TensorAccessPattern._raw(
            self._tensor_dims, self._offset, self._sizes, self._strides, tuple(pads)
        )

    def __iter__(self) -> Iterator[TensorAccessPattern]:
        """Iterate over the outermost dimension, like a NumPy array.

        `for row in TensorAccessPattern.full((M, N)).tile((m, n))` visits
        each row of tiles; iterate a row for the tiles themselves.

        Returns:
            Iterator[TensorAccessPattern]: One pattern per index of the outermost dimension.

        Raises:
            TypeError: If the outermost size is staged; index the pattern
                with a `range_` induction variable instead.
        """
        n = self._sizes[0]
        if is_sym(n):
            raise TypeError(
                "cannot iterate a runtime-sized dimension; index it with a "
                "range_ induction variable instead"
            )
        return (self[i] for i in range(n))

    # ------------------------------------------------------------ simulation

    def _require_concrete(self, what: str) -> None:
        if self.is_symbolic:
            raise TypeError(f"{what} needs a pattern with concrete values")

    def _walk(self) -> np.ndarray:
        # Flat tensor index of every stream element, shaped padded_sizes;
        # padded positions hold -1.
        self._require_concrete("simulating a walk")
        idx = np.zeros((), dtype=np.int64) + int(self._offset)
        for size, stride in zip(self._sizes, self._strides):
            idx = idx[..., None] + np.arange(int(size), dtype=np.int64) * int(stride)
        numel = int(np.prod(self._tensor_dims))
        if idx.max() >= numel:
            raise ValueError(
                f"{self} reaches element {idx.max()} of a {numel}-element tensor"
            )
        if self._padding is not None:
            idx = np.pad(idx, self._padding, constant_values=-1)
        return idx

    def gather(self, tensor: Any, pad_value: Any = 0) -> np.ndarray:
        """Return the stream a DMA walking `tensor` with this pattern emits.

        This is what an ObjectFifo given this pattern as `to_stream` sends,
        padding included.

        Args:
            tensor (array_like): A tensor of shape `tensor_dims` (any
                shape with the same number of elements is accepted).
            pad_value (optional): The value padded positions hold. Defaults to 0.

        Returns:
            np.ndarray: A 1-D array of `prod(padded_sizes)` elements.

        Raises:
            TypeError: If the pattern is staged.
            ValueError: If `tensor` does not have `prod(tensor_dims)` elements
                or the walk leaves the tensor.
        """
        arr = np.asarray(tensor)
        idx = self._walk().reshape(-1)
        self._check_numel(arr.size, "tensor")
        out = arr.reshape(-1)[np.maximum(idx, 0)]
        if self._padding is not None:
            out = np.where(idx >= 0, out, np.asarray(pad_value, dtype=arr.dtype))
        return out

    def scatter(self, stream: Any, out: np.ndarray | None = None) -> np.ndarray:
        """Write `stream` into a tensor the way a DMA walking this pattern does.

        This is what an ObjectFifo given this pattern as `from_stream` stores:
        stream element `k` lands at the `k`-th position of the walk.

        Args:
            stream (array_like): The stream, one element per step of the walk.
            out (np.ndarray | None, optional): The tensor to write into; elements the
                walk does not visit keep their values. Defaults to a zero tensor
                of shape `tensor_dims`.

        Returns:
            np.ndarray: The written tensor.

        Raises:
            ValueError: If the pattern is padded (only an emitting DMA pads) or
                the stream length does not match the walk.
        """
        if self._padding is not None:
            raise ValueError("only a to_stream pattern can pad")
        stream = np.asarray(stream).reshape(-1)
        idx = self._walk().reshape(-1)
        if stream.size != idx.size:
            raise ValueError(
                f"stream has {stream.size} elements for a walk of {idx.size}"
            )
        if out is None:
            out = np.zeros(self._tensor_dims, dtype=stream.dtype)
        self._check_numel(out.size, "out")
        flat = out.reshape(-1)
        flat[idx] = stream
        if not np.shares_memory(flat, out):
            out[...] = flat.reshape(out.shape)
        return out

    def _check_numel(self, size: int, what: str) -> None:
        numel = int(np.prod(self._tensor_dims))
        if size != numel:
            raise ValueError(
                f"{what} has {size} elements; the pattern walks a tensor of {numel}"
            )

    def accesses(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the access_order and access_count arrays.

        The access_order array numbers the accesses to each element of the
        tensor in walk order, -1 where the walk never goes; an element accessed
        more than once holds its last number. The access_count array holds the
        number of times the walk accesses each element.

        Returns:
            tuple[np.ndarray, np.ndarray]: access_order and access_count, each
                of shape `tensor_dims`.

        Raises:
            TypeError: If the pattern is staged.
        """
        idx = self._walk().reshape(-1)
        idx = idx[idx >= 0]
        numel = int(np.prod(self._tensor_dims))
        order = np.full(numel, -1, dtype=self._DTYPE)
        np.maximum.at(order, idx, np.arange(idx.size, dtype=order.dtype))
        count = np.bincount(idx, minlength=numel).astype(self._DTYPE)
        return order.reshape(self._tensor_dims), count.reshape(self._tensor_dims)

    def access_order(self) -> np.ndarray:
        """Return the access_order array of `accesses()`.

        Returns:
            np.ndarray: The walk number of each element's last access, -1 if unvisited.
        """
        return self.accesses()[0]

    def access_count(self) -> np.ndarray:
        """Return the access_count array of `accesses()`.

        Returns:
            np.ndarray: The number of times the walk accesses each element.
        """
        return self.accesses()[1]

    def compare_access_orders(self, other: TensorAccessPattern) -> bool:
        """Return whether two patterns walk the same elements in the same order.

        Patterns with different sizes and strides can still be functionally
        equivalent; this compares the walks themselves, padding included.

        Args:
            other (TensorAccessPattern): The pattern to compare to.

        Returns:
            bool: Whether the two walks visit the same elements in the same order.

        Raises:
            TypeError: If `other` is not a TensorAccessPattern.
        """
        if not isinstance(other, TensorAccessPattern):
            raise TypeError(
                "Can only compare access order against another TensorAccessPattern"
            )
        return np.array_equal(self._walk().reshape(-1), other._walk().reshape(-1))

    def visualize(
        self,
        show_arrows: bool | None = None,
        title: str | None = None,
        file_path: str | None = None,
        show_plot: bool = True,
        plot_access_count: bool = False,
    ) -> None:
        """Plot the access order (and optionally count) of the walk over its tensor.

        Args:
            show_arrows (bool | None, optional): Draw arrows between consecutively
                accessed elements. Defaults to None (only for small tensors).
            title (str | None, optional): Title of the plot. Defaults to the pattern's repr.
            file_path (str | None, optional): Path to save the plot to. Defaults to None.
            show_plot (bool, optional): Show the plot, e.g. in a Jupyter notebook.
                Defaults to True.
            plot_access_count (bool, optional): Plot the access count as well as
                the access order. Defaults to False.

        Raises:
            NotImplementedError: If the tensor has more than 2 dimensions.
        """
        from .visualization2d import visualize_from_accesses

        if len(self._tensor_dims) > 2:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )
        access_order, access_count = self.accesses()
        if title is None:
            title = str(self)
        visualize_from_accesses(
            np.atleast_2d(access_order),
            np.atleast_2d(access_count) if plot_access_count else None,
            title=title,
            show_arrows=show_arrows,
            file_path=file_path,
            show_plot=show_plot,
        )

    def animate(
        self,
        frame_dims: int = 1,
        title: str | None = None,
        animate_access_count: bool = False,
    ) -> FuncAnimation:
        """Animate the walk, one frame per index of its `frame_dims` outermost dimensions.

        For a tiling `full((M, N)).tile((m, n))`, `frame_dims=2` shows one
        tile per frame and `frame_dims=1` one row of tiles.

        Args:
            frame_dims (int, optional): Number of outermost dimensions the
                frames step through. Defaults to 1.
            title (str | None, optional): The title of the animation. Defaults to None.
            animate_access_count (bool, optional): Animate the access count
                as well as the access order. Defaults to False.

        Returns:
            FuncAnimation: A handle to the animation.

        Raises:
            NotImplementedError: If the tensor has more than 2 dimensions.
            ValueError: If `frame_dims` is not in `[0, rank]`.
        """
        from .visualization2d import animate_from_accesses

        if not 0 <= frame_dims <= self.rank:
            raise ValueError(
                f"frame_dims must be in [0, {self.rank}], got {frame_dims}"
            )
        self._require_concrete("animate()")
        if len(self._tensor_dims) > 2:
            raise NotImplementedError(
                "Visualization is only currently supported for 1- or 2-dimensional tensors"
            )
        shape = (
            self._tensor_dims
            if len(self._tensor_dims) == 2
            else (1, *self._tensor_dims)
        )
        order_frames = [np.full(shape, -1, self._DTYPE)]
        count_frames = [np.zeros(shape, self._DTYPE)] if animate_access_count else None
        for index in np.ndindex(*self._sizes[:frame_dims]):
            order, count = self[index].accesses()
            order_frames.append(order.reshape(shape))
            if count_frames is not None:
                count_frames.append(count.reshape(shape))
        return animate_from_accesses(
            order_frames,
            count_frames,
            title="TensorAccessPattern Animation" if title is None else title,
        )

    def __repr__(self) -> str:
        pad = "" if self._padding is None else f".pad({list(self._padding)})"
        return (
            f"TensorAccessPattern({show(self.tensor_dims)}, offset={show(self._offset)}, "
            f"sizes={show(self._sizes)}, strides={show(self._strides)}){pad}"
        )

    def _key(self) -> tuple:
        if self._padding is not None:
            dims = tuple(zip(self._sizes, self._strides, self._padding))
        else:
            c = self.coalesce()
            dims = tuple((n, s) for n, s in zip(c._sizes, c._strides) if n != 1)
        return (tuple(self._tensor_dims), self._offset, dims)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorAccessPattern):
            return NotImplemented
        if self.is_symbolic or other.is_symbolic:
            return self is other
        return self._key() == other._key()

    def __hash__(self) -> int:
        if self.is_symbolic:
            return object.__hash__(self)
        return hash(self._key())
