<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Migrating to the `taplib` access-pattern algebra

`TensorTiler2D` has been removed. Every tiling it produced is now built by
refining `TensorAccessPattern.full(dims)`, as described in the
[taplib reference](../api/taplib.md). `TensorAccessPattern` and
`TensorAccessSequence` remain, and the `ObjectFifo` keywords that take a DMA
walk have shorter names.

## `TensorTiler2D`

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

Each flag becomes one step of a chain, in this order. Leave out any step
whose flag was at its default:

```python
from aie.helpers.taplib import TensorAccessPattern

grid = TensorAccessPattern.full(dims).tile(tile_dims or dims)
grid = grid.permute_tile((1, 0))            # tile_col_major=True
grid = grid.order("col")                    # iter_col_major=True
grid = grid.group(tile_group_repeats,       # group_tiler: tile_group_dims; simple_tiler: skip
                  steps=tile_group_steps,   # omit if (1, 1)
                  order="col",              # tile_group_col_major=True
                  partial=True)             # allow_partial=True
grid = grid.repeat(pattern_repeat)          # pattern_repeat != 1
```

| Old | New |
| --- | --- |
| `simple_tiler(dims)` | `TensorAccessPattern.full(dims).tile(dims)` |
| `simple_tiler(dims, tile_dims)` | `TensorAccessPattern.full(dims).tile(tile_dims)` |
| `group_tiler(dims, tile_dims, tile_group_dims)` | `TensorAccessPattern.full(dims).tile(tile_dims).group(tile_group_dims)` |
| `step_tiler(dims, tile_dims, repeats, steps)` | `TensorAccessPattern.full(dims).tile(tile_dims).group(repeats, steps=steps)` |
| `tile_col_major=True` | `.permute_tile((1, 0))`, before `.group()` |
| `iter_col_major=True` | `.order("col")`, before `.group()` |
| `tile_group_col_major=True` | `.group(..., order="col")` |
| `allow_partial=True` | `.group(..., partial=True)` |
| `pattern_repeat=n` | `.repeat(n)` |

The chain returns a `TileGrid` in place of a `TensorAccessSequence`. It is a
`TensorAccessSequence`, so `len()`, iteration, `grid[i]`, `visualize()`,
`animate()` and `accesses()` work as before, and each tile is a
`TensorAccessPattern`. `grid[i, j]` also picks a tile by grid position. A
`TileGrid` cannot be edited in place; use
`TensorAccessSequence.from_taps(list(grid))` for a list you can change.

`test/python/taplib/tiler_vs_legacy.py` checks this mapping against the
retired tiler for hundreds of configurations.

### `prune_step` and column-major flags

With `prune_step=False`, the chain produces the same offset, sizes and
strides as the old tiler for every configuration. The one difference is the
old default, `prune_step=True`, combined with a column-major flag: there the
old tiler merged a repeat dimension into the tile. The new chain keeps the
dimensions separate. Both walks touch the same elements in the same order,
so the DMA does the same thing. If the descriptor itself must stay
identical, for example to keep a golden test stable, coalesce the tile:

```python
tap = grid[i].coalesce()
```

## `TensorAccessPattern`

- `TensorAccessPattern.from_slice(shape, key)` becomes
  `TensorAccessPattern.full(shape)[key]`, which walks the elements of
  `np.zeros(shape)[key]` in order.
- A hand-written pattern that cuts a flat buffer into `k` equal chunks,

  ```python
  TensorAccessPattern((1, N), i * (N // k), [1, 1, 1, N // k], [0, 0, 0, 1])
  ```

  becomes `TensorAccessPattern.full((1, N)).partition(k)[i]`.
- `==` ignores size-1 dimensions, which never step, so a pattern compares
  equal to the same walk written with fewer leading unit dimensions.

## `ObjectFifo` keywords

The keywords that take a DMA walk now accept a `TensorAccessPattern` as well
as a `[(size, stride), ...]` list:

| Old | New |
| --- | --- |
| `ObjectFifo(..., dims_to_stream=)` | `ObjectFifo(..., to_stream=)` |
| `ObjectFifo(..., dims_from_stream_per_cons=)` | `ObjectFifo(..., from_stream_per_cons=)` |
| `.cons(dims_from_stream=)` | `.cons(from_stream=)` |
| `.split(..., dims_to_stream=, dims_from_stream=)` | `.split(..., to_stream=, from_stream=)` |
| `.join(..., dims_to_stream=, dims_from_stream=)` | `.join(..., to_stream=, from_stream=)` |
| `.forward(..., dims_to_stream=, dims_from_stream=)` | `.forward(..., to_stream=, from_stream=)` |

A padded pattern (`tap.pad(...)`) passed as `to_stream` also sets
`pad_dimensions`, so you don't need to pass both.
