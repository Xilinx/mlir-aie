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
