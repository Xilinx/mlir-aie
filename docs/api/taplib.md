<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tensor Access Patterns (`taplib`)

`taplib` describes how DMAs walk tensors. A DMA buffer descriptor executes a
strided walk: an element *offset* plus parallel *sizes* and *strides*,
outermost dimension first. A `TensorAccessPattern` is exactly that walk over
a tensor of known shape. The tilings designs use can be built from
`TensorAccessPattern.full(dims)` (the row-major walk over the whole tensor)
with a few operations on those integers.

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
| Cut a tensor into equal tiles | `tap.tile(tile_dims)` |
| Cut one dimension into `k` equal chunks | `tap.partition(k)` |
| Read a tile-blocked buffer back in row-major order | `tap.inverse()` |

`tile()` turns a rank-`r` pattern into a rank-`2r` one: the grid
dimensions first, then the tile dimensions, so the whole result walks every
tile in row-major order. Everything else is ordinary indexing and
reordering of those dimensions:

| To | Use |
| --- | --- |
| One tile | `tiles[i, j]` |
| A row of tiles | `tiles[i]` |
| A block of tiles, or every `S`-th tile | `tiles[a:b, j::S]` |
| Visit tiles column by column | `tiles.permute((1, 0, 2, 3))` |
| Walk each tile column-major | `tiles.permute((0, 1, 3, 2))` |
| Walk a group of tiles again | `tiles[i].repeat(n)` |

Patterns are immutable: every operation returns a new one, and a list of
patterns is just a Python `list`.

```python
from aie.helpers.taplib import TensorAccessPattern

tiles = TensorAccessPattern.full((16, 16)).tile((4, 4))
print(tiles)        # TensorAccessPattern([16, 16], offset=0, sizes=[4, 4, 4, 4], strides=[64, 4, 16, 1])
print(tiles[1, 2])  # TensorAccessPattern([16, 16], offset=72, sizes=[4, 4], strides=[16, 1])
print(tiles[:2, :2])
                    # TensorAccessPattern([16, 16], offset=0, sizes=[2, 2, 4, 4], strides=[64, 4, 16, 1])
print(TensorAccessPattern.full((1024,)).partition(4)[2])
                    # TensorAccessPattern([1024], offset=512, sizes=[256], strides=[1])
```

Tiling a tiled pattern tiles hierarchically. `tiles.tile((1, 1, 4, 4))`
keeps the grid and cuts each tile into `4 x 4` sub-tiles, so the walk goes
tile by tile and, inside each tile, sub-tile by sub-tile.
`tiles[i, j].tile((4, 4))` does the same for one tile. Such a walk can need
more dimensions than one buffer descriptor holds, even after `coalesce()`. A
shim `fill()` or `drain()` with a static walk is then split into several
transfers. An
`ObjectFifo` pattern has to fit in one descriptor (four dimensions on a
memtile, three on a core tile).

### What a DMA can walk

A DMA moves whole 4-byte words. taplib itself accepts any walk, and the
compiler rejects one the hardware cannot do when it lowers the design:

- Every size, and every stride other than an innermost 1, must span whole
  words. A bf16 run is an even number of elements, and an int8 run a
  multiple of four.
- The innermost dimension steps a word at a time. For elements that are not
  32 bits wide, its stride must be 1.

So for bf16 or int8 data, `.T` and `permute()` of a tile are rejected, and so
is a `::2` slice of the innermost dimension. For `tiles[0, 0].T` on a bf16
tensor, a shim `fill()` fails with:

```
error: 'aie.dma_bd' op Stride 1 is 1 elements * 2 bytes = 2 bytes, which is not divisible by 4.
```

The same walk as a memtile's `to_stream` fails with:

```
error: 'aie.dma_bd' op For <32b width datatypes, inner-most dim stride must be 1
```

To transpose sub-word data, let the DMA move `s x s` blocks whose rows are
whole words, and transpose each block in the kernel. The `transposes` example
below does this.

## Using patterns

A pattern goes wherever IRON takes a DMA walk:

- `fill()` and `drain()` in a runtime sequence take it as `tap=`.
- An `ObjectFifo` takes it as `to_stream` (how the producer's DMA reads its
  object onto the stream) and as `from_stream` / `from_stream_per_cons` (how
  a consumer's DMA writes the stream into its object). The pattern walks a
  tensor the size of what each transfer moves, from offset 0: one object, or
  one segment of it on a join's output or a distribute's input. A padded
  pattern as `to_stream` also pads the stream on a MemTile.
- `tap.transformation_dims` gives the `((size, stride), ...)` pairs, for code
  that still wants them.

## Checking a data path on the host

On its way from the host to a core, a tensor passes through up to four
DMAs, each walking it with its own pattern: the shim reads the host tensor
onto the stream, the memtile writes it into an object, the memtile reads that
object back out, and the core writes it into its own object. On hardware a mistake in any of them just looks like scrambled data
in the kernel.

`gather(tensor)` returns the stream a DMA emits when it walks `tensor` with
a pattern, and `scatter(stream)` the object a DMA stores when it writes
`stream` with a pattern. Chaining them with the patterns a design uses shows exactly
what each core receives, with no hardware. This is the `transposes` design's
`--strategy=combined` path: the memtile shuffles each tile into `s x s`
blocks so that the kernel only has to transpose each block in place.

```python
import numpy as np
from aie.helpers.taplib import TensorAccessPattern

M, K, m, n, s = 64, 64, 16, 16, 8
host = np.arange(M * K).reshape(M, K)
shim = TensorAccessPattern.full((M, K)).tile((m, n))
memtile_in = TensorAccessPattern.full((n, m)).tile((s, s)).permute((1, 2, 0, 3))

for t, tile in enumerate(shim.gather(host).reshape(-1, m * n)):
    i, j = divmod(t, K // n)
    obj = memtile_in.scatter(tile)  # the memtile's (n, m) object
    blocks = obj.reshape(n // s, s, m // s, s)
    kernel_out = blocks.transpose(0, 3, 2, 1).reshape(n, m)
    assert (kernel_out == host[i * m : (i + 1) * m, j * n : (j + 1) * n].T).all()
```

For a padded pattern, `gather(tensor, pad_value=)` fills the padded
positions, and `padded_sizes` is the shape the receiving object must have.
To inspect a single walk, use `accesses()`, `access_order()`,
`access_count()`, `compare_access_orders()` and `visualize()`.

## Staged patterns in a dispatch-time sequence

Every operation also accepts staged values: an `aie.ir.Value` (a
`DispatchTime[T]` scalar a runtime sequence receives, or any arithmetic on
one) can stand in for a size, stride, offset, index, slice bound or repeat
count. The arithmetic is emitted as `arith` ops where it is used, and
each check that would have raised `ValueError` becomes a `cf.assert`
guard. A fully static specialization folds the guard away, or fails at
generation time if it is false. The dispatch-time builder instead returns
no stream, and the host refuses the call with the guard's message.

```python
def seq(a_h, b_h, start, n, in_prod, out_cons):
    # The buffer as max_tiles equal chunks; the chunk index is staged
    # arithmetic (start + loop iv), so the tap's offset is too.
    chunks = TensorAccessPattern.full((max_tiles * tile_size,)).partition(max_tiles)
    for tile in range_(n):
        tap = chunks[start + tile]
        tg = TaskGroup()
        out_cons.drain(b_h, tap=tap, wait=True, group=tg)
        in_prod.fill(a_h, tap=tap, group=tg)
        tg.finish()
```

`require(cond, message)` from `aie.iron` adds a guard of your own, such as a
shape constraint the design depends on. `cond` can depend on dispatch-time
values. `whole_array.py` (below) has
`require(M % (m * n_aie_rows) == 0, "M must be a multiple of m * n_aie_rows")`
with a dispatch-time `M`, and a call whose `M` breaks it is refused with that
message. The message is a plain string fixed when the design is generated, so
it cannot include the value the call passed.

Some rules of thumb:

- Sizes, strides, offsets and loop bounds can all be dispatch-time values,
  but the number of transfers a loop step issues cannot. A `range_` over a
  dispatch-time bound is emitted as a loop, so its body is generated once.
  Every step issues the same transfers, waited on the same way, in the same
  order, and the `TaskGroup`s carried from one step to the next are a fixed
  set of values (see
  [Runtime tasks](../programming_guide/section-2/section-2d/RuntimeTasks.md)).
  This comes from how the sequence is generated, not from the DMA. So when the data does not divide evenly into blocks, the loop covers the
  whole blocks, and the ragged last block is one more transfer after it,
  under `with if_(rem > 0):`, with its size computed from `rem`.
- The structure of a pattern stays a Python value: its rank, slice steps,
  padding amounts, and the sizes and strides of the dimensions `merge()`
  combines.
- Staged values of any integer type are accepted. The compiler hoists their
  arithmetic out of the buffer-descriptor block, and a value too wide for its
  descriptor field is refused at dispatch instead of being truncated.

`programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`
is the whole-array GEMM written this way, with dispatch-time `M`, `K` and `N`.

### Checking a dispatch-time design

`instructions(**scalars)` on an `@iron.jit` design returns the instruction
words a call with those values would run, built exactly as a call builds
them, so no NPU is needed. A dispatch-time stream is not byte-identical to
a static specialization's: it draws buffer descriptors from a pool, polls for
room in a channel's task queue and assembles descriptor words at build time.
`aie.utils.txn_trace` reduces either stream to the events the hardware acts
on (queue pushes resolved to the chain of transfers they start, token waits,
and other register writes such as runtime parameters) and compares those:

```python
from aie.utils.txn_trace import compare, explain

words = tiled_copy.specialize().instructions(n_tiles=3, start_tile=1)
static = tiled_copy.specialize(n_tiles=3, start_tile=1).instructions()
assert compare(words, static) == []
print(explain(words))  # one line per event
```

`python -m aie.utils.txn_trace insts.bin [other.bin]` does the same from
the command line. `test/python/dispatch_taplib_copy.py` and the whole-array
GEMM's `tests/dispatch_txn.py` are complete examples.

## API reference

::: helpers.taplib.tap.TensorAccessPattern
    options:
      show_root_heading: true
      heading_level: 3

### Utilities

::: helpers.taplib.utils
    options:
      show_root_heading: false

### Instruction-stream tracing

::: utils.txn_trace
    options:
      show_root_heading: false
