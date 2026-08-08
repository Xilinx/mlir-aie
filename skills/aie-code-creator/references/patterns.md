<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# IRON Design Patterns

Copy a skeleton, adapt the dtype / size / kernel name. All examples use the high-level API with `bfloat16` and target `NPU2` unless noted.

## Topology cheatsheet

```
        Element-wise (single core)            Element-wise (multi-core, distribute/join)

   host ──fill──▶ of_in ──▶ Worker ──▶ of_out ──drain──▶ host

                                              ┌─▶ sub_in0 ─▶ Worker0 ─▶ sub_out0 ─┐
   host ──fill──▶ of_in (split) ──────────────┼─▶ sub_in1 ─▶ Worker1 ─▶ sub_out1 ─┤(join)──▶ of_out ──drain──▶ host
                                              └─▶ sub_inN ─▶ WorkerN ─▶ sub_outN ─┘

        Broadcast (1-to-N, same data)         Producer-consumer pipeline

                  ┌─▶ Worker0 ─▶ of_out0 ─┐
   host ─▶ of_in ─┼─▶ Worker1 ─▶ of_out1 ─┤(join)─▶ host
                  └─▶ WorkerN ─▶ of_outN ─┘
                                              host ─▶ of_in ─▶ Worker_s1 ─▶ of_mid ─▶ Worker_s2 ─▶ of_out ─▶ host
        Reduction (multi-core then combine)

   host ─▶ of_in (split) ─▶ Worker_red[i] ─▶ partials ─▶ Worker_combine ─▶ of_out ─▶ host
```

---

## Element-wise, single core

```python
import numpy as np
from ml_dtypes import bfloat16
from aie.iron import Program, Runtime, Worker, ObjectFifo, Kernel
from aie.iron.device import NPU2Col1
from aie.iron.controlflow import range_

N = 4096
TILE = 1024
tensor_ty = np.ndarray[(N,),    np.dtype[bfloat16]]
tile_ty   = np.ndarray[(TILE,), np.dtype[bfloat16]]

of_in  = ObjectFifo(tile_ty, name="in")
of_out = ObjectFifo(tile_ty, name="out")

add_k = Kernel("eltwise_add_bf16_vector", "add.o",
               [tile_ty, tile_ty, tile_ty])     # signature: (in0, in1, out) — this demo passes the same input twice

def core(of_in, of_out, k):
    for _ in range_(N // TILE):
        ein  = of_in.acquire(1)
        eout = of_out.acquire(1)
        k(ein, ein, eout)                       # in-place self-add demo
        of_in.release(1)
        of_out.release(1)

w = Worker(core, [of_in.cons(), of_out.prod(), add_k])

def sequence(a_in, c_out, in_h, out_h):
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)

rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])

module = Program(NPU2Col1(), rt, workers=[w]).resolve_program()
```

---

## Element-wise, multi-core data-parallel (distribute/join)

```python
N_WORKERS = 4
N = 8192
TILE = N // N_WORKERS
tensor_ty = np.ndarray[(N,),    np.dtype[bfloat16]]
tile_ty   = np.ndarray[(TILE,), np.dtype[bfloat16]]

of_in_top  = ObjectFifo(tensor_ty, name="in")
of_out_top = ObjectFifo(tensor_ty, name="out")

offsets = [TILE * i for i in range(N_WORKERS)]
sub_ins  = of_in_top.cons().split(offsets,
                                  obj_types=[tile_ty]*N_WORKERS,
                                  names=[f"in{i}"  for i in range(N_WORKERS)])
sub_outs = of_out_top.prod().join(offsets,
                                  obj_types=[tile_ty]*N_WORKERS,
                                  names=[f"out{i}" for i in range(N_WORKERS)])

scale_k = Kernel("scale_bf16_vector", "scale.o",
                 [tile_ty, tile_ty, np.int32, np.int32])

def core(of_in, of_out, k):
    ein  = of_in.acquire(1)
    eout = of_out.acquire(1)
    k(ein, eout, 3, TILE)
    of_in.release(1); of_out.release(1)

workers = [Worker(core, [sub_ins[i].cons(), sub_outs[i].prod(), scale_k])
           for i in range(N_WORKERS)]

def sequence(a_in, c_out, in_h, out_h):
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)

rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in_top.prod(), of_out_top.cons()])

module = Program(NPU2Col1(), rt, workers=workers).resolve_program()
```

---

## Broadcast (1 producer → N consumers, same data)

Each call to `of.cons()` returns a NEW consumer handle. The compiler inserts the broadcast routing.

```python
of_in  = ObjectFifo(tile_ty, name="in")
of_outs = [ObjectFifo(tile_ty, name=f"out{i}") for i in range(N_WORKERS)]

def core(of_in, of_out, k):
    for _ in range_(num_iters):
        ein  = of_in.acquire(1)
        eout = of_out.acquire(1)
        k(ein, eout, TILE)
        of_in.release(1); of_out.release(1)

workers = [Worker(core, [of_in.cons(), of_outs[i].prod(), k]) for i in range(N_WORKERS)]
```

---

## Producer-consumer pipeline (two stages on two cores)

```python
of_in  = ObjectFifo(tile_ty, name="in",  depth=2)
of_mid = ObjectFifo(tile_ty, name="mid", depth=2)
of_out = ObjectFifo(tile_ty, name="out", depth=2)

k_stage1 = Kernel("relu_bf16",    "relu.o",    [tile_ty, tile_ty, np.int32])
k_stage2 = Kernel("scale_bf16_vector", "scale.o", [tile_ty, tile_ty, np.int32, np.int32])

def stage1(of_in, of_mid, k):
    for _ in range_(num_iters):
        ein = of_in.acquire(1); emid = of_mid.acquire(1)
        k(ein, emid, TILE)
        of_in.release(1); of_mid.release(1)

def stage2(of_mid, of_out, k):
    for _ in range_(num_iters):
        emid = of_mid.acquire(1); eout = of_out.acquire(1)
        k(emid, eout, 3, TILE)
        of_mid.release(1); of_out.release(1)

w1 = Worker(stage1, [of_in.cons(),  of_mid.prod(), k_stage1])
w2 = Worker(stage2, [of_mid.cons(), of_out.prod(), k_stage2])
```

Both workers run concurrently; the `of_mid` ObjectFifo handles the inter-stage synchronization.

---

## Reduction (per-core partial + final combine)

```python
N = 8192
N_RED = 4
CHUNK = N // N_RED
in_ty      = np.ndarray[(N,),     np.dtype[np.int32]]
chunk_ty   = np.ndarray[(CHUNK,), np.dtype[np.int32]]
partial_ty = np.ndarray[(1,),     np.dtype[np.int32]]
out_ty     = np.ndarray[(1,),     np.dtype[np.int32]]

of_in       = ObjectFifo(in_ty, name="in")
offsets     = [CHUNK * i for i in range(N_RED)]
sub_ins     = of_in.cons().split(offsets, obj_types=[chunk_ty]*N_RED,
                                 names=[f"in{i}" for i in range(N_RED)])
of_partials = [ObjectFifo(partial_ty, name=f"p{i}") for i in range(N_RED)]
of_out      = ObjectFifo(out_ty, name="out")

red_k     = Kernel("reduce_add_vector", "reduce.o", [chunk_ty, partial_ty, np.int32])
combine_k = Kernel("combine_add",       "reduce.o", [partial_ty, partial_ty, partial_ty, partial_ty, out_ty])

def red(of_in, of_p, k):
    ein = of_in.acquire(1); ep = of_p.acquire(1)
    k(ein, ep, CHUNK)
    of_in.release(1); of_p.release(1)

reducers = [Worker(red, [sub_ins[i].cons(), of_partials[i].prod(), red_k])
            for i in range(N_RED)]

def combine(p0, p1, p2, p3, of_out, k):
    e0 = p0.acquire(1); e1 = p1.acquire(1); e2 = p2.acquire(1); e3 = p3.acquire(1)
    eo = of_out.acquire(1)
    k(e0, e1, e2, e3, eo)
    p0.release(1); p1.release(1); p2.release(1); p3.release(1); of_out.release(1)

combiner = Worker(combine, [*[p.cons() for p in of_partials], of_out.prod(), combine_k])

def sequence(a_in, c_out, in_h, out_h):
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)

rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])

module = Program(NPU2Col1(), rt, workers=[*reducers, combiner]).resolve_program()
```

---

## GEMM-style (sketch — see `mlir-aie/programming_examples/basic/matrix_multiplication/` for full impl)

```
A (m × k)   B (k × n)
  │            │
  └─split rows ┴─split cols ─▶ N×M Workers each compute one (tm × tn) tile of C
                                using MMUL inner loop over k tiles.
                              Per-tile partial C is held in L1 accumulator,
                              streamed out only when k-loop completes.
```

Key points:

- The C++ kernel is `matmul_vectorized_*` (see `kernel_intrinsics.md` §MMUL).
- Tile sizes must satisfy MMUL divisibility (see `architecture.md`).
- Each worker accumulates over its k-tiles in L1; the partial output FIFO has `depth=2` so the next k-batch can start while the previous tile drains.

---

## Runtime parameters (RTP) with barrier

```python
from aie.iron import Buffer, WorkerRuntimeBarrier

rtp     = Buffer(np.ndarray[(8,), np.dtype[np.int32]], name="rtp",
                 use_write_rtp=True)
barrier = WorkerRuntimeBarrier()

def core(rtp, barrier, of_in, of_out, k):
    barrier.wait_for_value(1)
    scale = rtp[0]
    for _ in range_(num_iters):
        ein = of_in.acquire(1); eout = of_out.acquire(1)
        k(ein, eout, scale, TILE)
        of_in.release(1); of_out.release(1)

w = Worker(core, [rtp, barrier, of_in.cons(), of_out.prod(), scale_k])

def sequence(a_in, c_out, in_h, out_h):
    rtp[0] = 7                    # RTP writes: index the Buffer directly in the sequence body
    barrier.set(1)                # unblocks core()'s wait_for_value(1) above
    in_h.fill(a_in)
    out_h.drain(c_out, wait=True)

rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])

module = Program(NPU2Col1(), rt, workers=[w]).resolve_program()
```

`rtp` and `barrier` don't need to appear in `Runtime`'s `fn_args` list — they're captured directly from the enclosing scope. `fn_args` only needs entries for values that must become body parameters (types → host buffers) or objects the sequence body doesn't already close over.
