<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tensor Access Patterns (`taplib`)

`taplib` provides abstractions for describing how data is tiled and streamed
between memory and AIE compute tiles. A **Tensor Access Pattern** (TAP)
describes a multi-dimensional iteration over a buffer, generating the DMA
descriptor sequences that the NPU hardware executes.

## The layout algebra

A DMA buffer descriptor executes a strided walk: an element *offset* plus
parallel *sizes* and *strides* lists, outermost dimension first. `taplib`
models that walk as a `Layout`, a view over a tensor of a known shape.
`Layout.full(dims)` is the row-major walk over a whole tensor; every other
view is a composition of a few pure operations on the view's integers:

| Operation | Method |
| --- | --- |
| Tile a view into a grid of equal tiles | `.tile(tile_dims)` (returns a `TileGrid`) |
| Cut one dimension into `k` equal chunks | `.partition(k)` |
| Pick one tile | `grid[i]` (step order) or `grid[i, j]` (grid index) |
| Gather several tiles into one step | `grid.group(repeats, steps=, col_major=, partial=)` |
| Change the order tiles are visited | `grid.order("col")` |
| Transpose the walk inside each tile | `grid.permute_tile((1, 0))` |
| Transpose or permute a whole view | `.permute((1, 0))` |
| Restrict a view with NumPy indexing | `layout[2:6, ::2]` (`.slice()`) |
| Walk the same data again | `.repeat(n)` (a stride-0 outermost dimension) |
| Read a tile-blocked buffer back in row-major order | `grid.inverse()` |
| Fewest dimensions for the same walk | `.coalesce()` |

Because every operation is arithmetic on sizes and strides, the same code runs
on Python ints at generation time and on staged runtime values inside a
dynamic runtime sequence (see the `symbolic` helpers below).

A `Layout` converts to the two forms the rest of IRON consumes:

- `fill()`/`drain()` and an `ObjectFifo`'s `dims_to_stream`/`dims_from_stream`
  take a `Layout` directly; a `PaddedLayout` also sets `pad_dimensions`.
- `.tap()` returns the `TensorAccessPattern` shim form (four dimensions,
  left-padded with unit dimensions); `fill()` / `drain()` accept a `Layout`
  directly and call this for you.
- `.stream_dims()` returns `[(size, stride), ...]` at the view's exact rank,
  ready for an ObjectFifo's `dims_to_stream` / `dims_from_stream_per_cons`.

`grid.materialize()` turns every tile of a `TileGrid` into a
`TensorAccessSequence`, which is what the visualization tools
(`.visualize()`, `.animate()`, `.accesses()`) operate on.

```python
from aie.helpers.taplib import Layout

grid = Layout.full((16, 16)).tile((4, 4)).group((2, 2))
print(len(grid))     # 4
print(grid[0])       # Layout([16, 16], offset=0, sizes=[2, 2, 4, 4], strides=[64, 4, 16, 1])
print(grid[0].tap()) # TensorAccessPattern([16, 16] offset=0, sizes=[2, 2, 4, 4], strides=[64, 4, 16, 1])
```

::: helpers.taplib.layout.Layout
    options:
      show_root_heading: true
      heading_level: 2

::: helpers.taplib.layout.TileGrid
    options:
      show_root_heading: true
      heading_level: 2

::: helpers.taplib.layout.PaddedLayout
    options:
      show_root_heading: true
      heading_level: 2

::: helpers.taplib.tap.TensorAccessPattern
    options:
      show_root_heading: true
      heading_level: 2

::: helpers.taplib.tas.TensorAccessSequence
    options:
      show_root_heading: true
      heading_level: 2

## Symbolic helpers

::: helpers.taplib.symbolic
    options:
      show_root_heading: false

## Utilities

::: helpers.taplib.utils
    options:
      show_root_heading: false

## Migrating from TensorTiler2D

`TensorTiler2D` has been removed from `aie.helpers.taplib`. Its replacement is
the `Layout` / `TileGrid` algebra described above; `TensorAccessPattern` and
`TensorAccessSequence` are unchanged. Every old tiler configuration has a
direct spelling in the new API, built left to right by composing a few
operations on `Layout.full(dims)`.

### Mapping

The old factories took a flat list of flags:

```python
TensorTiler2D.simple_tiler(dims, tile_dims=None, tile_col_major=False,
                           iter_col_major=False, pattern_repeat=1, prune_step=True)
TensorTiler2D.group_tiler(dims, tile_dims, tile_group_dims=(1, 1), tile_col_major=False,
                          tile_group_col_major=False, iter_col_major=False,
                          pattern_repeat=1, allow_partial=False, prune_step=True)
TensorTiler2D.step_tiler(dims, tile_dims, tile_group_repeats, tile_group_steps=(1, 1),
                         tile_col_major=False, tile_group_col_major=False,
                         iter_col_major=False, allow_partial=False, pattern_repeat=1,
                         prune_step=True)
```

Each flag becomes one step in a chain. Omit any step whose flag was left at
its default:

```python
from aie.helpers.taplib import Layout

grid = Layout.full(dims).tile(tile_dims or dims)
grid = grid.permute_tile((1, 0))            # tile_col_major=True
grid = grid.order("col")                    # iter_col_major=True
grid = grid.group(tile_group_repeats,       # group_tiler: tile_group_dims; simple_tiler: skip
                  steps=tile_group_steps,   # omit if (1, 1)
                  col_major=tile_group_col_major,
                  partial=allow_partial)
grid = grid.repeat(pattern_repeat)          # pattern_repeat != 1
```

| Old | New |
| --- | --- |
| `simple_tiler(dims)` | `Layout.full(dims).tile(dims)` |
| `simple_tiler(dims, tile_dims)` | `Layout.full(dims).tile(tile_dims)` |
| `group_tiler(dims, tile_dims, tile_group_dims)` | `Layout.full(dims).tile(tile_dims).group(tile_group_dims)` |
| `step_tiler(dims, tile_dims, repeats, steps)` | `Layout.full(dims).tile(tile_dims).group(repeats, steps=steps)` |
| `tile_col_major=True` | `.permute_tile((1, 0))` (before `.group()`) |
| `iter_col_major=True` | `.order("col")` (before or after `.group()`) |
| `tile_group_col_major=True` | `.group(..., col_major=True)` |
| `allow_partial=True` | `.group(..., partial=True)` |
| `pattern_repeat=n` | `.repeat(n)` |

`iter_col_major` maps to `.order("col")` and can be applied before or after
`.group()`. `tile_col_major` maps to `.permute_tile((1, 0))` and must come
before `.group()`, because the permutation is over the plain tile
dimensions.

### Using the result

The old factories returned a `TensorAccessSequence`; the new chain returns a
`TileGrid`, and each of its tiles is a `Layout`:

- `grid[i]` is the i-th tile in step order; `grid[i, j]` picks a tile by grid
  index; `len(grid)`, iteration and `grid.num_steps` work as before.
- `fill(..., tap=grid[i])` and `drain(..., tap=grid[i])` accept a `Layout`
  directly. Any API typed on `TensorAccessPattern`, such as the dialect-level
  `shim_dma_single_bd_task(tap=...)`, needs `grid[i].tap()`.
- `.tap()` returns the four-dimensional shim form of a view (left-padded with
  unit dimensions). `.stream_dims()` returns `[(size, stride), ...]` at the
  view's exact rank, which is what an ObjectFifo's `dims_to_stream` /
  `dims_from_stream_per_cons` want; prefer it over the old
  `tiles[i].transformation_dims` (which is still available as
  `grid[i].tap().transformation_dims`).
- `grid.materialize()` returns a `TensorAccessSequence` of every tile. Use it
  where a sequence is required: `.visualize()`, `.animate()`, `.accesses()`,
  `.access_order()`, `.compare_access_orders()`, `==` against a
  `TensorAccessSequence`, and the inputs of `TensorAccessSequence.from_taps()`.
- A single-tile tiler that was indexed as `tiles[0]` becomes `grid[0]` (a
  `Layout`); add `.tap()` where a `TensorAccessPattern` is required, and call
  `.visualize()` and friends on `grid[0].tap()`.

### `prune_step` and column-major flags

With `prune_step=False` the new spelling produces byte-identical offset,
sizes and strides for every configuration, including `allow_partial=True`.
The one numeric difference is the old default `prune_step=True` combined with
a column-major flag, where the old tiler merged a repeat dimension into the
tile. The new API keeps dimensions separate; the two walks touch the same
elements in the same order, so the DMA behaviour is unchanged. If you need
the descriptor itself to stay byte-identical (for example to keep a golden
test stable), call `.coalesce()` on the tile:

```python
tap = grid[i].coalesce().tap()
```

### Other patterns

- `TensorAccessPattern.from_slice(shape, key)` becomes
  `Layout.full(shape)[key].tap(None)`. Passing `None` to `.tap()` keeps the
  view's own rank instead of padding to four dimensions.
- A hand-written pattern that cuts a flat buffer into `k` equal chunks,

  ```python
  TensorAccessPattern((1, N), i * (N // k), [1, 1, 1, N // k], [0, 0, 0, 1])
  ```

  becomes

  ```python
  Layout.full((1, N)).partition(k)[i]
  ```

## Composing layouts across hops

A design moves a tensor through up to four DMA walks before a core sees it:
the shim reads the host tensor onto the stream (`fill` / `drain`), the
memtile writes the stream into its object (`dims_from_stream`), a sub-fifo
reads its segment of that object back onto the stream (`dims_to_stream` on
`split` / `forward`) and the core writes the stream into its own object
(`dims_from_stream`). Each walk is only checked on hardware, and a mistake
in one of them shows up as scrambled data in the kernel.

`aie.helpers.taplib.pipeline` lets you check the chain at generation time.
A `Hop` is one walk over one object: its `kind` (one of `KINDS`: `"shim"`,
`"memtile_in"`, `"memtile_out"`, `"core_in"`), the object `shape`, the
`(size, stride)` list as `dims_to_stream` / `dims_from_stream` take it
(`None` for a linear walk), a segment `offset` and `length`, and the element
width in bytes. A `Pipeline` is a list of hops in stream order, built with
the fluent methods:

- `Pipeline.shim(tap)` takes a `Layout` or `TensorAccessPattern` (the
  runtime tap you pass to `fill` / `drain`). Several shim hops may be added
  for taps issued one after another.
- `Pipeline.memtile_in(shape, dims)` is the memtile consumer's
  `dims_from_stream` on the fifo the shim feeds.
- `Pipeline.memtile_out(shape, dims, offset=, length=)` is one sub-fifo's
  `dims_to_stream` over its segment of the memtile object; add one per
  `split` segment.
- `Pipeline.core_in(shape, dims)` is the core consumer's `dims_from_stream`
  (one hop, or one per `memtile_out` segment).

Two methods use the chain:

- `check()` returns a list of messages, one for every hop that the tile's
  DMA could not execute (too many addressing dimensions, a wrap or stride
  wider than the tile allows, a shim repeat beyond the queue limit, or an
  offset / stride that is not a whole 32-bit granule) and for every walk
  that does not cover its object or segment exactly. An empty list means
  the chain is legal.
- `compose()` follows every element through the chain with NumPy and
  returns one array per core object received, shaped like the core object,
  holding the host flat index of each element. `host_coords(host_shape,
  objects)` turns those into host coordinates.

`LIMITS` holds the per-tile limits (`max_dims`, `max_wrap`, `max_stride` and
the shim's `max_repeat` and `max_iter`) for the AIE2 family (NPU1 and NPU2).

The example below is the `transposes` design with `--strategy=combined`: the
memtile block-shuffles each tile so the kernel only transposes `s x s`
sub-tiles, and the check confirms that what the core receives is the
transposed tile.

```python
import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.taplib.pipeline import Pipeline

M, K, m, n, s = 64, 64, 16, 16, 8
host = np.arange(M * K).reshape(M, K)
tap_in_L3L2 = TensorAccessPattern(
    (M, K), 0, [M // m, K // n, m, n], [m * K, n, K, 1]
)
tap_in_L2L1 = TensorAccessPattern(
    (M, K), 0, [m // s, s, n // s, s], [s, m, s * m, 1]
)
pipe = (
    Pipeline()
    .shim(tap_in_L3L2)
    .memtile_in((m, n), tap_in_L2L1.transformation_dims)
    .memtile_out((m, n))
    .core_in((m, n))
)
assert pipe.check() == []
objs = pipe.compose()
assert len(objs) == (M // m) * (K // n)
t = 0
for i in range(M // m):
    for j in range(K // n):
        tile = host[i * m : (i + 1) * m, j * n : (j + 1) * n]
        core = objs[t]  # host indices, as stored in the core's (m, n) object
        # The kernel transposes every s x s block in place; the result must
        # be the transposed tile.
        blocks = core.reshape(m // s, s, n // s, s).transpose(0, 2, 3, 1)
        kernel_out = blocks.transpose(0, 2, 1, 3).reshape(m, n)
        assert (kernel_out == tile.T).all(), (i, j)
        t += 1
print("transposes chain: every tile arrives block-transposed")
```

A memtile MM2S channel can pad the stream it emits (`ObjectFifo`'s
`pad_dimensions` / `pad_value`). `Layout.pad([(before, after), ...])` attaches
that padding to a walk: passed as `dims_to_stream`, the `PaddedLayout` sets
both the walk and `pad_dimensions` (`stream_dims()` and `pad_dims()` give the
two lists if you need them), `padded_sizes` is what the consuming object must
hold, and `materialize()` shows where the constants land. A `memtile_out` hop
accepts the `PaddedLayout` the same way and `compose()` delivers padded
positions as host index `-1`, so the composed core object is exactly `np.pad`
of the tile.

```python
padded = Layout.full((rows, N))[:, :cols].pad([(1, 1), (2, 2)])
of_out = ObjectFifo(padded_ty, dims_to_stream=padded, pad_value=0)
Pipeline().shim(...).memtile_in((rows, N)).memtile_out((rows, N), padded).core_in(padded.padded_sizes)
```

::: helpers.taplib.pipeline
    options:
      show_root_heading: false

## Staged taps in a dispatch-time sequence

Every operation above also accepts staged values: an `aie.ir.Value` (the
`DispatchTime[T]` scalars a runtime sequence receives, and any arithmetic on
them) can stand in for a size, stride, offset, grid index, repeat count or
group size. The algebra then emits its arithmetic as `arith` ops at the point
of use, and every check it would have raised as a `ValueError` becomes an
`aiex.npu.require` guard: a fully static specialization folds the guard away
(or fails at generation time when it is false), while the dispatch-time C++
builder returns no stream and the host refuses the call.

```python
def sequence(A, B, C, M, K, N, A_hs, B_hs, C_hs):
    require(M % (m * n_aie_rows) == 0, "M must be a multiple of m * n_aie_rows")
    A_tiles = Layout.full((M, K)).tile((m * rows, k)).group((1, K // k))
    for step in range_(M // m // rows):        # an index counter; cast inside the tiler
        A_hs[0].fill(A, tap=A_tiles[step], group=tg)   # offset is staged arithmetic
```

The rules of thumb:

- Keep transfer counts static where the hardware needs them static. A
  `range_` over a staged bound stays rolled; a step's `TaskGroup` is finished
  in the same loop body or carried to the next iteration as a `range_`
  iter_arg (see the runtime-tasks guide), so a ragged last block is a peeled
  `with if_(rem > 0):` rather than a runtime-length inner loop.
- Sizes and strides reach `aie.dma_bd` as `i64`; the builder widens a
  narrower staged value for you, and hoists the transfer length and repeat
  count it derives from them before the task region opens.
- Messages must not quote staged values; use `symbolic.show(x)`, which renders
  them as `<runtime>`, or keep them constant.

`programming_examples/basic/matrix_multiplication/whole_array/whole_array_dyn.py`
is the whole-array GEMM written this way (dispatch-time `M`, `K`, `N`), and
`test/python/dispatch_taplib_gemm.py` shows how to check such a design
without an NPU.

### Checking a dispatch-time design

A dispatch-time builder's stream is never byte-identical to a static
specialization's: it draws buffer descriptors from a pool, polls before
reuse and assembles BD words at build time. `aie.utils.txn_trace` reduces
either stream to the DMA events the hardware acts on (queue pushes resolved
to the transfer they start, token waits) and compares those:

```python
from aie.utils.txn_trace import compare, explain
words = bridge.generate({"M": 256, "K": 128, "N": 128})
static = np.fromfile(static_design.compile()[1], dtype=np.uint32)
assert compare(words, static) == []
print(explain(words))   # one line per DMA event
```

`python -m aie.utils.txn_trace insts.bin [other.bin]` does the same from the
command line.

::: utils.txn_trace
    options:
      show_root_heading: false
