<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# IRON Python API Reference

This reference tracks the current `aie.iron` API as installed. The one exception is the
`amd/IRON` operator-framework pattern in `patterns.md`/`complete_examples.md`, which
intentionally pins an older wheel and keeps that generation's `Runtime`/`Program` shape.

The docstrings in the installed `aie.iron` package are kept up to date — if something here
looks stale or you need a detail this file doesn't cover, check the installed package
directly (e.g. `python3 -c "import aie.iron as iron; help(iron.Program)"`, or read the
source under the active Python environment's `site-packages/aie/iron/`) rather than
assuming this reference is current.

All imports come from `aie.iron`, `aie.iron.device`, and `aie.iron.controlflow`.

For the prebuilt kernel library (`aie.iron.kernels`), the whole-design templates
(`aie.iron.algorithms`), and the `In`/`Out`/`CompileTime[T]` jit signature markers, see
[`builtin_kernels.md`](builtin_kernels.md) — check there before hand-building any of the
primitives documented here.

## Standard imports

```python
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import (
    Program, Runtime, Worker, ObjectFifo, Kernel, Buffer,
    WorkerRuntimeBarrier, TaskGroup, ScratchpadParameter,
)
from aie.iron.device import NPU1, NPU1Col1, NPU2, NPU2Col4, Tile, AnyComputeTile
from aie.iron.controlflow import range_
import aie.iron as iron
```

There is **no `aie.iron.placers` module and no `SequentialPlacer`** — tile placement is a
compiler pass (`--aie-place-tiles`) that runs over the `aie.logical_tile` ops
`resolve_program()` emits. If you see either name in older code or docs, it predates 1.4.0.

## Type descriptors

IRON uses NumPy-style type descriptors to declare tensor types passed between host, ObjectFifos, and kernels:

```python
N = 1024
tile_ty   = np.ndarray[(N,),       np.dtype[bfloat16]]
mat_ty    = np.ndarray[(M, N),     np.dtype[np.int8]]
scalar_ty = np.ndarray[(1,),       np.dtype[np.int32]]
```

The C++ kernel's pointer type and the element count of the buffer it receives must match.

## ObjectFifo

```python
of = ObjectFifo(elem_type, name="my_fifo", depth=2)

producer = of.prod()              # endpoint used on the producer side (or in the sequence's .fill())
consumer = of.cons()              # NEW consumer handle each call — for broadcast to N
```

`prod()`/`cons()` also take an optional `tile=` to pin the shim tile a runtime-driven endpoint binds to (`prod(tile=...)`, `cons(tile=..., dims_from_stream=...)`); leave it `None` (the default) unless you have a specific placement reason — IRON picks any available shim tile automatically.

Inside a `Worker` body:

```python
elem = of.acquire(n)              # blocks until n objects available; returns scalar if n==1, list-like if n>1
... use elem ...
of.release(n)                     # must match the acquire count
```

Topology helpers (executed at design build time, not on the core):

```python
# Distribute: split one input across N workers' private FIFOs
of_offsets = [chunk * i for i in range(N)]
sub_fifos = of.cons().split(of_offsets,
                            obj_types=[tile_ty] * N,
                            names=[f"sub{i}" for i in range(N)])

# Join: aggregate N producer FIFOs into one output
joined = of_out.prod().join(of_offsets,
                            obj_types=[tile_ty] * N,
                            names=[f"sub{i}" for i in range(N)])

# Forward: implicit copy through a mem tile
of_via_mem = of.forward(obj_type=tile_ty, name="forwarded")
```

`split()`/`join()`/`forward()` also accept `tile=` (which mem tile hosts the split/join/forward DMA; defaults to `AnyMemTile`), `depths=`/`depth=` (per-sub-fifo depth override, defaults to the parent's `depth`), `dims_to_stream=`/`dims_from_stream=` (per-sub-fifo stream-dimension reshaping), `plio=` (mark sub-fifos as PLIO), and `repeat_counts=`/`repeat_count=` (per-sub-fifo MemTile DMA repeat count). Defaults cover the common case above; reach for these only for custom placement or streaming layouts.

The plain `ObjectFifo(...)` constructor itself also accepts `dims_to_stream=`/`dims_from_stream_per_cons=` — distinct from the per-sub-fifo kwargs above, this reshapes the *whole* fifo's stream dimensions before any split/join:

```python
of_out = ObjectFifo(data_ty, name="out", dims_to_stream=dims)
```

`depth`: 2 gives ping-pong (double-buffering — the producer fills one slot while the consumer drains the other). For a pipeline where a stage holds a buffer while the next stage also holds one, set `depth ≥ producer_outstanding + consumer_outstanding`, where each term is the number of buffers that side holds acquired-but-not-released at once.

### Advanced ObjectFifo (opt-in)

These two constructor kwargs stay out of the way of simple designs; reach for them only when a plain fifo (or split/join) can't express what you need. They land on `aie.objectfifo` attributes.

**`consumer_obj_type=` — asymmetric transfer granularity.** The producer sends `obj_type`-sized chunks; the consumer receives smaller `consumer_obj_type`-sized chunks. Producer element count must be an integer multiple of the consumer's. Use it when one DMA fan-out feeds consumers that each walk a sub-slice, avoiding a second fifo + join.

```python
prod_ty = np.ndarray[(40,), np.dtype[np.int32]]
cons_ty = np.ndarray[(10,), np.dtype[np.int32]]
wts = ObjectFifo(prod_ty, depth=1, name="wts",
                 consumer_obj_type=cons_ty)   # 4:1 — one fill, four consumer acquires
```

**`aie_stream=(end, port)` — direct AIE-stream, no L1 buffer.** Marks the producer side as wire-only; the consumer reads straight off the stream. Pair with a kernel that writes per-element via `aie::stream::put_ms(value)` — the core body never acquires/releases the producer handle.

```python
of_out = ObjectFifo(dout_ty, name="out", depth=2, aie_stream=(0, 0))
```

## Worker

```python
def core_body(of_in, of_out, kernel_fn, scale):
    for _ in range_(num_iters):
        ein  = of_in.acquire(1)
        eout = of_out.acquire(1)
        kernel_fn(ein, eout, scale)
        of_in.release(1)
        of_out.release(1)

w = Worker(
    core_body,
    fn_args=[of_in.cons(), of_out.prod(), kernel_fn, 5],
    # tile=Tile(col, row),             # optional: pin to a tile; default AnyComputeTile
    # stack_size=0xD00,                # optional: bump if you get stack overflows
    # while_true=False,                # optional: run core_body once instead of forever (default True)
    # allocation_scheme="bank-aware",  # or "basic-sequential"; default bank-aware
    # trace=8192,                      # optional: per-worker trace buffer size in bytes
)
```

The `fn_args` list is positional and must match `core_body`'s parameters.

The placement kwarg is **`tile=`** (not `placement=`), and it defaults to `AnyComputeTile`,
which leaves the choice to the `--aie-place-tiles` pass. Passing a non-compute tile raises
`ValueError` at construction.

`while_true` (default `True`) wraps `core_body` in a `while(true)` loop so the core keeps servicing the fifo until reconfiguration; pass `False` for a core that should run exactly once (rare — most designs want the default forever-loop).

`stack_size` defaults to the target model's `getDefaultCoreStackSize()` (currently 1024 bytes) — bump it if a core overflows its stack. `dynamic_objfifo_lowering` is a per-core override of the ObjectFifo lowering strategy and is only honored when the global `--dynamic-objFifos` flag is false; leave it `None` unless you're specifically working around a lowering issue.

### CascadeFlow — direct cascade-stream link between two Workers

For designs that chain compute across adjacent tiles' cascade interconnect (e.g. an accumulator passed core-to-core) rather than through an ObjectFifo, connect two `Worker`s directly:

```python
from aie.iron import CascadeFlow

cf = CascadeFlow(src=worker_a, dst=worker_b)   # registers on worker_a; resolved when Program.resolve_program() places tiles
```

The kernels on both sides must drive the stream themselves via the AIE API's `put_mcd`/`get_scd` intrinsics — `CascadeFlow` only declares the topology edge, not the data movement.

Hardware constraints, all checked at lowering rather than construction (so a bad chain surfaces as an `aie.configure_cascade` verifier error, not a Python exception):

- **Direction is fixed.** On AIE2/AIE2P the cascade input must come from **North or West** and the output must go **South or East**. A chain therefore has to descend in row or advance east — you cannot run it upward or westward. In practice this means pinning tiles (e.g. head at row 5 down to tail at row 2 within a column) rather than leaving the chain to `--aie-place-tiles`.
- **Adjacency**: `src`/`dst` tiles must end up cardinally adjacent after placement.
- **Fan-in/out**: each compute tile has at most one cascade input and one cascade output.
- **No shim/mem participation**: ShimTiles and MemTiles have no cascade interface, so the last core in a chain must land its result in L1 for a normal ObjectFifo DMA — the cascade cannot itself reach the host.
- **Width**: the accumulator cascade is 512 bits on AIE2/AIE2P (384 on AIE1, where cascade isn't supported by this API anyway) — e.g. a `v16int32` payload. See `programming_examples/basic/matrix_multiplication/cascade/` for a worked example (a row-accumulator chain across 4 cores).

Unlike `Flow`/`PacketFlow` (`low_level_api.md`), constructing a `CascadeFlow` self-registers it on `src`'s outgoing-cascade list — there's no separate `rt.add_flow(...)` call needed.

## Kernel — external C++ function

```python
kernel_fn = Kernel(
    "eltwise_add_bf16_vector",           # extern "C" name in the .o file
    "add.o",                              # path to compiled object
    [tile_ty, tile_ty, tile_ty],          # arg types: must match kernel signature
)
```

For JIT designs you can use `iron.ExternalFunction` to compile a `.cc` source on-the-fly:

```python
kernel_fn = iron.ExternalFunction(
    "eltwise_add_bf16_vector",
    source_file="add.cc",
    arg_types=[tile_ty, tile_ty, tile_ty],
    compile_flags=["-DBIT_WIDTH=16"],
)
```

A kernel/`ExternalFunction` call silently inserts a `memref.collapse_shape` when an N-D contiguous memref argument (e.g. a 2-D ObjectFifo element) feeds a flat 1-D kernel signature with a matching element count and dtype — this is expected, not an error; a genuine shape/dtype mismatch still fails at MLIR verification.

## Buffer — L1-resident scalars / accumulators / RTPs

```python
acc = Buffer(np.ndarray[(64,), np.dtype[np.float32]], name="acc")
rtp = Buffer(np.ndarray[(16,), np.dtype[np.int32]], name="rtp0",
             use_write_rtp=True)         # host can write at runtime
```

Pass `Buffer`s to `Worker.fn_args` to use them inside the core; pair with `WorkerRuntimeBarrier` if the host writes them during execution.

By default a `Buffer` is pinned to whichever tile the `Worker` that uses it lands on. Pass `tile=` explicitly to pin a `Buffer` on a *different* tile than its consuming core — e.g. a lookup table that lives on a neighbor tile and is read directly through shared L1 memory:

```python
lut_buf = Buffer(tile=west_tile, type=lut_ty,
                  initial_value=np.array(lut_arr, dtype=np.int16), name="lut_buf")
```

`Worker` preserves any explicit `tile=` placement; it only auto-pins `Buffer`s that were created without one. See `programming_examples/ml/magika/group2.py` for a design that spreads four LUTs across the four neighbor tiles of a compute core this way.

## Runtime — host-side control flow

`Runtime` wraps a plain Python function — the **sequence body** — that describes the host-side data movement. Workers are *not* started from inside it; they're launched implicitly by being passed to `Program(..., workers=[...])`. Each entry in `Runtime`'s `fn_args` list is either:

- a **type** (a tensor type like `in_ty`, or a scalar type like `np.int32`) — declares a runtime input; the sequence body receives a live value for it (a `RuntimeData` handle for tensor types you can `.fill()`/`.drain()`, a bare SSA value for scalar types).
- a concrete **int** — also declares a runtime input, but folds it into a compile-time constant instead of a host-settable value (useful for `range_`/`if_` bounds that should specialize to a static path).
- anything else (an `ObjectFifoHandle` from `.prod()`/`.cons()`, a `Buffer`, a `Kernel`, a `WorkerRuntimeBarrier`, a `ScratchpadParameter`, ...) — passed through to the body unchanged, exactly like `Worker`'s `fn_args`.

`fill`/`drain` are now methods on the `ObjectFifoHandle` you get from `.prod()`/`.cons()`, called from inside the sequence body — there's no more `rt.fill`/`rt.drain`/`rt.start`:

```python
def sequence(a_in, c_out, in_h, out_h):
    in_h.fill(a_in)                       # DMA host → AIE
    out_h.drain(c_out, wait=True)         # DMA AIE → host

rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])

module = Program(NPU2Col1(), rt, workers=[w]).resolve_program()
```

`wait=True` blocks the sequence until that transfer completes; omit it (default `False`) to let it run asynchronously and rely on a `TaskGroup` (or a later `wait=True` transfer) to synchronize.

### Task groups — batch fills/drains so they pipeline

```python
def sequence(a_in, c_out, in_h, out_h):
    tg = TaskGroup()
    in_h.fill(a_in, group=tg)
    out_h.drain(c_out, wait=True, group=tg)
    tg.finish()                           # awaits waited transfers first, then frees the rest
```

Construct a `TaskGroup()` from inside the sequence body (it registers itself with the active sequence). Pass it as `group=` to `fill`/`drain`; call `.finish()` once all of the group's transfers have been issued. If you don't pass `group=` explicitly, `fill`/`drain` still enroll in the sequence's implicit default group — you don't need a `TaskGroup` at all for the common single-shot case above.

### Non-contiguous access via TensorAccessPattern

```python
from aie.helpers.taplib import TensorAccessPattern
tap = TensorAccessPattern((1, total_size),
                          offset=col * chunk,
                          sizes=[1, 1, 1, chunk],
                          strides=[0, 0, 0, 1])
in_h.fill(a_in, tap=tap)
```

`tap` and the raw `sizes=`/`strides=`/`offset=`/`transfer_len=` kwargs are mutually exclusive — pass one style or the other, not both. Omit both and `fill`/`drain` default to a full linear transfer of the whole buffer.

### Unmanaged transfers — hand-rolled software pipelining

By default (`managed=True`) a transfer is enrolled in a `TaskGroup` that frees it automatically. Pass `managed=False` to own the returned `Task` yourself — e.g. to carry it across `scf.for` iterations as an iter_arg in a manually pipelined loop:

```python
task = in_h.fill(a_in, managed=False)
...
task.await_()
task.free()
```

`managed=False` and `group=` are mutually exclusive (an unmanaged transfer isn't part of a `TaskGroup`).

### Runtime parameters (RTP) and the worker barrier

RTP writes are now plain indexing on the `Buffer` inside the sequence body — there's no `rt.inline_ops`. `WorkerRuntimeBarrier.set(value)` replaces `rt.set_barrier`:

```python
def sequence(a_in, c_out, in_h, out_h):
    rtp[0] = 50                           # write directly into the Buffer
    barrier.set(1)                        # unblocks a worker's wait_for_value(1)
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)
```

`rtp` and `barrier` don't need to appear in `Runtime`'s `fn_args` — they're captured directly from the enclosing Python scope. `fn_args` only needs entries for values that must become sequence-body parameters (host-supplied types) or objects the body doesn't already close over.

### Scratchpad parameters — an alternative to RTP Buffers

`ScratchpadParameter` is a newer, simpler way to pass a named scalar from host to core, without a dedicated `Buffer`/lock pair:

```python
seq_len = ScratchpadParameter("seq_len", np.int32)

def core_body(p, of_in, of_out, k):
    v = p.read()                          # inside the Worker core
    ...

w = Worker(core_body, [seq_len, of_in.cons(), of_out.prod(), k])

def sequence(a_in, c_out, in_h, out_h):
    # write seq_len via offset_parameter= on a fill/drain call, then:
    iron.sync_parameters()                # emits the host→scratchpad sync; call before workers read it
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)
```

Reach for `Buffer(..., use_write_rtp=True)` + `WorkerRuntimeBarrier` for the common case (matches most existing examples); use `ScratchpadParameter` when you specifically want to avoid a dedicated RTP buffer/lock.

Note that `ScratchpadParameter`s depend on the full-ELF flow (`aiecc --get-full-elf`) and are not available on the Phoenix/Hawk Point architectures.

## Program — finalize the design

Workers are now passed explicitly to `Program`, not discovered from the runtime sequence:

```python
prog = Program(NPU2Col1(), rt, workers=[w1, w2])
module = prog.resolve_program()   # MLIR module
```

`resolve_program(device_name="main")` takes no placer argument. It emits each tile as an
`aie.logical_tile` op and runs MLIR verification; the `--aie-place-tiles` compilation pass
later turns those into concrete `aie.tile` ops. To constrain placement, pin the individual
`Worker`/`Buffer` with `tile=Tile(col, row)` rather than swapping a placer object. The
`device_name` argument only names the emitted `aie.device` symbol — relevant for
multi-device modules, not for placement.

If the default placement genuinely fails to route, the escape hatch is a build-time flag
(`--placer=sa_placer`), not a Python object — see
[`programming_guide/section-1/README.md`](../../../programming_guide/section-1/README.md) (SA placer section).

### Tracing

Trace configuration now lives on `Program`, not `Runtime`:

```python
prog = Program(NPU2Col1(), rt, workers=[w])
prog.enable_trace(trace_size=8192, workers=[w])   # workers=None traces every worker with `trace` set
module = prog.resolve_program()
```

`enable_trace` also takes `reuse_output_buffer=` (write trace data into the tail of the last output buffer instead of appending a dedicated trace argument), `coretile_events=`/`coremem_events=`/`memtile_events=`/`shimtile_events=` (up to 8 trace events per tile type), and `egress_shim_col=` (which shim column egresses trace packets to DDR).

## iron.jit — compile and run from Python

```python
@iron.jit
def my_design(input_t: In, output_t: Out, *, N: CompileTime[int] = 4096):
    # ... build of, workers, runtime exactly as above ...
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()

# First call compiles to xclbin, caches; subsequent calls reuse it.
my_design(input_array, output_array, N=4096)
```

`use_cache` defaults to `True`, so `@iron.jit` bare is the normal form — pass
`use_cache=False` only to force a rebuild. Other config keys: `source_files`,
`object_files`, `compile_flags`, `include_paths`, `aiecc_flags`, `trace_config`, `full_elf`. Any other
keyword is matched against a `CompileTime[T]` parameter name and raises `TypeError` at
decoration time if it doesn't match one.

Annotate tensor parameters with `In` / `Out` / `InOut` and specialization knobs as
keyword-only `CompileTime[T]`; see [`builtin_kernels.md`](builtin_kernels.md) for the full
signature rules and worked examples.

If your inputs are already `iron.tensor(..., device="npu")` arrays, `jit` will route them through XRT automatically.

## WorkerRuntimeBarrier — gate workers on host-set RTPs

```python
barrier = WorkerRuntimeBarrier()
rtp = Buffer(np.ndarray[(8,), np.dtype[np.int32]], name="rtp", use_write_rtp=True)

def core(rtp, barrier, of_in, of_out):
    barrier.wait_for_value(1)             # block until host signals
    scale = rtp[0]                        # now safe to read
    # ... use scale ...

w = Worker(core, [rtp, barrier, of_in.cons(), of_out.prod()])

def sequence(a_in, c_out, in_h, out_h):
    rtp[0] = 5
    barrier.set(1)
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)

rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
module = Program(NPU2Col1(), rt, workers=[w]).resolve_program()
```
