<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tensor Access Patterns (`taplib`)

`taplib` describes how DMAs walk tensors. A DMA buffer descriptor executes a
strided walk: an element *offset* plus parallel *sizes* and *strides*,
outermost dimension first. A `TensorAccessPattern` is exactly that walk over
a tensor of known shape, and every tiling a design needs is built from
`TensorAccessPattern.full(dims)` (the row-major walk over the whole tensor)
with a few operations on those integers.

Coming from `TensorTiler2D` or the `dims_to_stream` keywords? See the
[taplib migration guide](../migration/taplib.md).

## Building patterns

| To | Use |
| --- | --- |
| Walk a whole tensor row by row | `TensorAccessPattern.full(dims)` |
| Restrict the walk with NumPy indexing | `tap[2:6, ::2]`, `tap[i]` |
| Reorder or reverse the dimensions | `tap.permute((1, 0))`, `tap.T` |
| Split or merge a dimension | `tap.split(dim, inner)`, `tap.merge(dim)` |
| Use the fewest dimensions for the same walk | `tap.coalesce()` |
| Walk the same data again | `tap.repeat(n)` (a stride-0 outermost dimension) |
| Have a memtile pad the stream | `tap.pad([(before, after), ...])` |
| Cut a tensor into equal tiles | `tap.tile(tile_dims)` (a `TileGrid`) |
| Cut one dimension into `k` equal chunks | `tap.partition(k)` (a `TileGrid`) |

A `TileGrid` is a sequence of tile patterns. `grid[i]` is the tile at step
`i` and `grid[i, j]` the tile at grid position `(i, j)`. It is refined with:

| To | Use |
| --- | --- |
| Gather several tiles into each step | `grid.group(repeats, steps=, order=, partial=)` |
| Visit tiles column by column | `grid.order("col")` |
| Transpose the walk inside each tile | `grid.permute_tile((1, 0))` |
| Walk each tile again | `grid.repeat(n)` |
| Get the whole ungrouped grid as one walk | `grid.tap` |
| Read a tile-blocked buffer back in row-major order | `grid.inverse()` |

Patterns and grids are immutable: every operation returns a new one.

```python
from aie.helpers.taplib import TensorAccessPattern

grid = TensorAccessPattern.full((16, 16)).tile((4, 4)).group((2, 2))
print(len(grid))  # 4
print(grid[0])    # TensorAccessPattern([16, 16] offset=0, sizes=[2, 2, 4, 4], strides=[64, 4, 16, 1])
print(TensorAccessPattern.full((1, 1024)).partition(4)[2])
                  # TensorAccessPattern([1, 1024] offset=512, sizes=[1, 256], strides=[0, 1])
```

## Using patterns

A pattern goes wherever IRON takes a DMA walk:

- `fill()` and `drain()` in a runtime sequence take it as `tap=`.
- An `ObjectFifo` takes it as `to_stream` (how the producer's DMA reads its
  object onto the stream) and as `from_stream` / `from_stream_per_cons` (how
  a consumer's DMA writes the stream into its object). A padded pattern given
  as `to_stream` sets the fifo's `pad_dimensions` too.
- `tap.transformation_dims` gives the `[(size, stride), ...]` list, for code
  that still wants one.

## Checking a data path on the host

A tensor crosses up to four DMA walks before a core sees it: the shim reads
the host tensor onto the stream, the memtile writes it into an object, the
memtile reads that object back out, and the core writes it into its own
object. On hardware a mistake in any of them just looks like scrambled data
in the kernel.

`to_stream(tensor)` returns the stream a DMA walking `tensor` with a pattern
emits, and `from_stream(stream)` the object a DMA writing `stream` with a
pattern stores. Chaining them with the patterns a design uses shows exactly
what each core receives, with no hardware. This is the `transposes` design's
`--strategy=combined` path: the memtile shuffles each tile into `s x s`
blocks so that the kernel only has to transpose each block in place.

```python
import numpy as np
from aie.helpers.taplib import TensorAccessPattern

M, K, m, n, s = 64, 64, 16, 16, 8
host = np.arange(M * K).reshape(M, K)
shim = TensorAccessPattern.full((M, K)).tile((m, n)).tap
memtile_in = TensorAccessPattern.full((n, m)).tile((s, s)).tap.permute((1, 2, 0, 3))

for t, tile in enumerate(shim.to_stream(host).reshape(-1, m * n)):
    i, j = divmod(t, K // n)
    obj = memtile_in.from_stream(tile)  # the memtile's (n, m) object
    blocks = obj.reshape(n // s, s, m // s, s)
    kernel_out = blocks.transpose(0, 3, 2, 1).reshape(n, m)
    assert (kernel_out == host[i * m : (i + 1) * m, j * n : (j + 1) * n].T).all()
```

For a padded pattern, `to_stream(tensor, pad_value=)` fills the padded
positions, and `padded_sizes` is the shape the receiving object must have.
To inspect a single walk, use `accesses()`, `access_order()`,
`access_count()`, `compare_access_orders()` and `visualize()`.

## Staged patterns in a dispatch-time sequence

Every operation also accepts staged values: an `aie.ir.Value` (a
`DispatchTime[T]` scalar a runtime sequence receives, or any arithmetic on
one) can stand in for a size, stride, offset, grid index, repeat count or
group size. The arithmetic is emitted as `arith` ops where it is used, and
each check that would have raised `ValueError` becomes an `aiex.npu.require`
guard. A fully static specialization folds the guard away, or fails at
generation time if it is false. The dispatch-time builder instead returns
no stream, and the host refuses the call with the guard's message.

```python
def seq(a_h, b_h, start, n, in_prod, out_cons):
    # The buffer as max_tiles equal chunks; the chunk index is staged
    # arithmetic (start + loop iv), so the tap's offset is too.
    chunks = TensorAccessPattern.full((1, max_tiles * tile_size)).partition(max_tiles)
    for tile in range_(n):
        tap = chunks[start + tile]
        tg = TaskGroup()
        out_cons.drain(b_h, tap=tap, wait=True, group=tg)
        in_prod.fill(a_h, tap=tap, group=tg)
        tg.finish()
```

`require(cond, message)` from `aie.helpers.taplib.symbolic` adds a guard of
your own, such as a shape constraint the design depends on.

Some rules of thumb:

- Keep transfer counts static where the hardware needs them static. A
  `range_` over a staged bound stays rolled. A step's `TaskGroup` is either
  finished in the same loop body or carried to the next iteration as a
  `range_` iter_arg (see the runtime-tasks guide). That is why a ragged last
  block is a peeled `with if_(rem > 0):` rather than a runtime-length inner
  loop.
- Staged values of any integer type are accepted. The compiler hoists their
  arithmetic out of the buffer-descriptor block, and a value too wide for its
  descriptor field is refused at dispatch instead of being truncated.
- Messages must not quote staged values. Use `symbolic.show(x)`, which renders
  them as `<runtime>`, or keep them constant.

`programming_examples/basic/matrix_multiplication/whole_array/whole_array_dyn.py`
is the whole-array GEMM written this way, with dispatch-time `M`, `K` and `N`.

### Checking a dispatch-time design

`instructions(**scalars)` on an `@iron.jit` design returns the instruction
words a call with those values would run, built exactly as a call builds
them, so no NPU is needed. A dispatch-time stream is never byte-identical to
a static specialization's: it draws buffer descriptors from a pool, polls
before reusing one and assembles descriptor words at build time.
`aie.utils.txn_trace` reduces either stream to the DMA events the hardware
acts on (queue pushes resolved to the transfer they start, plus token waits)
and compares those:

```python
from aie.utils.txn_trace import compare, explain

words = tiled_copy.specialize().instructions(n_tiles=3, start_tile=1)
static = tiled_copy.specialize(n_tiles=3, start_tile=1).instructions()
assert compare(words, static) == []
print(explain(words))  # one line per DMA event
```

`python -m aie.utils.txn_trace insts.bin [other.bin]` does the same from
the command line. `test/python/dispatch_taplib_copy.py` and the whole-array
GEMM's `tests/dispatch_txn.py` are complete examples.

## API reference

::: helpers.taplib.tap.TensorAccessPattern
    options:
      show_root_heading: true
      heading_level: 3

::: helpers.taplib.tas.TileGrid
    options:
      show_root_heading: true
      heading_level: 3

::: helpers.taplib.tas.TensorAccessSequence
    options:
      show_root_heading: true
      heading_level: 3

### Symbolic helpers

::: helpers.taplib.symbolic
    options:
      show_root_heading: false

### Utilities

::: helpers.taplib.utils
    options:
      show_root_heading: false

### Instruction-stream tracing

::: utils.txn_trace
    options:
      show_root_heading: false
