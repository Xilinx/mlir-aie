<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Pitfalls and Anti-Patterns

Each entry: **the bad pattern, why it breaks, and the fix.**

---

## ❌ Forgetting the trailing `release` after a sliding-window loop → deadlock

`acquire(n)` means "after this call, ensure I hold `n` objects" — **not** "acquire `n`
additional objects." So `acquire(2)` + `release(1)` every iteration is a legitimate
cyclostatic/sliding-window pattern: on the first iteration it acquires 2 new objects: on
every subsequent iteration it acquires only 1 new one (the other was already held from
the previous iteration's un-released slot), then releases 1 to advance the window:

```python
# GOOD — sliding window over pairs of elements
def core(of_in, of_out, k):
    for _ in range_(N):
        e = of_in.acquire(2)     # 2 new on iter 0, then 1 new + 1 carried per iter after
        k(e[0], e[1], out_tmp)
        of_in.release(1)          # advance the window by 1
    of_in.release(1)              # BAD if omitted: release the still-held final object
```

The actual bug is **omitting that trailing `release(1)` after the loop** — the last
acquired-but-unreleased object is never returned to the producer, so after `depth`
launches the FIFO is permanently short one free slot and the producer blocks forever.

**Fix**: track how many objects are still held when a loop like this exits, and release
them explicitly afterward. This isn't the same rule as "every `acquire(n)` must be
matched by a same-call `release(n)`" — sliding-window code intentionally holds objects
across iterations; the invariant to check is that the *total* acquired count equals the
*total* released count once the handle is done being used (including after the loop, and
on every early-return path).

---

## ❌ ObjectFifo `depth` too small for the pipeline → deadlock or starvation

If a downstream worker holds a buffer while waiting for another input, and depth is 2, the producer may block before the downstream can release.

**Fix**: set `depth ≥ producer_outstanding + consumer_outstanding` — each term is the count of buffers that side holds acquired-but-not-released at once. For deep pipelines, route through a mem-tile ObjectFifo with `depth=4` or more.

---

## ❌ Using Python `range` instead of `range_` in a core body

```python
# BAD
def core(of_in, of_out, k):
    for i in range(N // TILE):       # unrolls at design build time — explodes code size
        ...

# GOOD
from aie.iron.controlflow import range_
def core(of_in, of_out, k):
    for i in range_(N // TILE):      # emits a hardware loop
        ...
```

Using plain `range` for non-trivial counts will either OOM your build or generate a giant flattened kernel that doesn't fit in L1.

---

## ❌ Wrong `pass_size_to_kernel` on an `algorithms.transform*` template

```python
# BAD — kernels.add is (in0, in1, out); the template appends a trailing size arg
transform_parallel_binary(kernels.add(tile_size=1024), tensor_ty, tile_size=1024)
```

`pass_size_to_kernel` defaults to `True`, appending `tile_size` as a trailing `int` to every kernel call. That's right for `(in, out, n)` kernels like `kernels.passthrough`, and wrong for bare `(in, out)` kernels like `kernels.add`/`kernels.mul`/`kernels.relu`.

**Fix**: set `pass_size_to_kernel=False` for bare-signature kernels. The failure surfaces as an MLIR verification error about argument count — annoying to read, but it does fail loudly rather than computing garbage.

---

## ❌ Hand-writing a kernel that `aie.iron.kernels` already ships

Writing your own `relu.cc` or `mm.cc` means maintaining a second, probably slower copy of a kernel that upstream already vectorized and tests. It also costs you `.mac_dims`, the `_ref` NumPy oracles, and the `.zero` companion kernel that `kernels.mm` hands you for free.

**Fix**: check [`builtin_kernels.md`](builtin_kernels.md) first. Hand-write when the dtype/tile isn't supported, the op fuses stages the library only exposes separately, or the access pattern isn't tile-in/tile-out — and say so explicitly when you do, so the choice is visible.

---

## ❌ `split`/`join` sub-object size doesn't match the offset stride → 75% of your data silently vanishes

```python
N = 4096; n_workers = 4; chunk = N // n_workers      # chunk = 1024
tile_ty = np.ndarray[(256,), np.dtype[bfloat16]]

# BAD: offsets stride by 1024, but each sub-object is only 256 elements
offsets = [chunk * i for i in range(n_workers)]      # [0, 1024, 2048, 3072]
subs = of.cons().split(offsets, obj_types=[tile_ty] * n_workers, names=[...])
```

Offsets are positions **within the parent object**, so the sub-objects have to tile the parent exactly. The example above covers 4 × 256 = 1024 of 4096 elements. It resolves, verifies, and compiles without a single warning — then computes on a quarter of your data.

```python
# GOOD: sub-object size == offset stride
chunk_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]
subs = of.cons().split(offsets, obj_types=[chunk_ty] * n_workers, names=[...])
```

**Fix**: assert `len(offsets) * sub_object_elems == parent_elems`. If you want tile-granular L1 buffers beneath a coarser split, split at chunk granularity and `forward()` each sub-fifo down to tile-sized objects — don't shrink `obj_types` under a coarse stride.

---

## ❌ Mismatched ObjectFifo type vs. kernel signature → garbage data

```python
# Design declares 32-bit ints
of = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]], name="of")

# Kernel signature expects bfloat16
k = Kernel("my_k", "my.o", [np.ndarray[(1024,), np.dtype[bfloat16]], ...])  # WRONG
```

Type mismatch is not caught — the kernel reads/writes raw bytes with the wrong interpretation. **Fix**: make the IRON `tile_ty`, the `Kernel` arg-type list, and the C++ pointer type agree element-by-element.

---

## ❌ Missing `__restrict` on kernel pointers → no pipelining

```cpp
// BAD: compiler must assume a, b, c may alias
void kernel(bfloat16 *a, bfloat16 *b, bfloat16 *c, int N) { ... }
```

The modulo scheduler can't pipeline reads from `a` ahead of writes to `c`. Throughput drops 5–20×.

```cpp
// GOOD
void kernel(const bfloat16 *__restrict a,
            const bfloat16 *__restrict b,
                  bfloat16 *__restrict c, int N) { ... }
```

---

## ❌ Relying on `AIE_PREPARE_FOR_PIPELINING` alone → scalar-rate loops under Peano

`AIE_PREPARE_FOR_PIPELINING` expands to `[[chess::prepare_for_pipelining]]` under Chess but to **nothing at all** under Peano/AIECC — which is the default backend. A hot loop annotated only with it gets no pipelining hint whatsoever on a default build, and you see one vector op per several cycles instead of one per cycle.

`AIE_LOOP_MIN_ITERATION_COUNT(n)` is the one that carries the information under both backends (it becomes `clang loop min_iteration_count(n)` under Peano): the scheduler needs to know the body runs at least `n` times to justify a prologue/epilogue.

```cpp
// GOOD — the MIN_ITERATION_COUNT is doing the real work on a default build
AIE_PREPARE_FOR_PIPELINING          // free under Chess, no-op under Peano
AIE_LOOP_MIN_ITERATION_COUNT(16)    // real under both
for (int i = 0; i < F; ++i) { ... }
```

Keep both — `AIE_PREPARE_FOR_PIPELINING` costs nothing and helps if someone builds with Chess — but never treat it as sufficient. Same caveat applies to `AIE_LOOP_FLATTEN` (Chess-only). In the other direction, `AIE_TRY_INITIATION_INTERVAL(n)` and `AIE_PREPARE_FOR_POSTPIPELINING` are real under **Peano only** and no-ops under Chess.

---

## ❌ Branch/switch on the loop index in a small fixed-count inner loop → scheduler refuses to pipeline

```cpp
// BAD: 3x3 conv window, branches on i inside the loop
AIE_LOOP_RANGE(3, 3)
for (int i = 0; i < 3; ++i) {
    bfloat16 w = (i == 0) ? w0 : (i == 1) ? w1 : w2;   // branch on loop index
    acc = aie::mac(acc, load_v<VEC>(row[i]), w);
}
```

A live branch/switch on the loop variable blocks back-to-back vector issue even when the trip count is a compile-time constant — `AIE_LOOP_RANGE`/`AIE_LOOP_MIN_ITERATION_COUNT` only bound the trip count, they don't remove the branch. For small, fixed-count loops like this, fully unroll instead so each iteration's branch resolves at compile time:

```cpp
// GOOD
AIE_LOOP_UNROLL_FULL
for (int i = 0; i < 3; ++i) {
    bfloat16 w = (i == 0) ? w0 : (i == 1) ? w1 : w2;
    acc = aie::mac(acc, load_v<VEC>(row[i]), w);
}
```

`AIE_LOOP_UNROLL_FULL` is real under both Chess and Peano/AIECC (unlike `AIE_PREPARE_FOR_PIPELINING`), so this is a safe default for small fixed-trip-count loops with any data-dependent branching in the body.

---

## ❌ Vector size doesn't divide the tile size → wrong output / OOB

```cpp
constexpr int VEC = 16;
const int F = N / VEC;               // 1000 / 16 = 62 → last 8 elements skipped
for (int i = 0; i < F; ++i) { ... }
```

```cpp
// GOOD
static_assert(N % VEC == 0, "tile size must be divisible by vector width");
```

---

## ❌ MMUL dimensions don't satisfy divisibility → compile-time failure or silent wrong results

```cpp
using MMUL = aie::mmul<4, 8, 4, bfloat16, bfloat16, accauto>;   // r=4, s=8, t=4
// Used inside a 4x4 outer expansion:
static_assert(m % (4*r) == 0);   // m % 16 == 0
static_assert(k % s     == 0);   // k % 8  == 0
static_assert(n % (4*t) == 0);   // n % 16 == 0
```

Pad up to the next valid shape rather than skipping the assert.

---

## ❌ Using compiler-specific intrinsics instead of the AIE API → not portable to AIE2P

```cpp
// BAD — chess-specific names
v32bfloat16 zeros = broadcast_zero_bfloat16();
v32bfloat16 out   = max(in, zeros);

// GOOD — portable AIE API
auto zeros = aie::zeros<bfloat16, 32>();
auto out   = aie::max(in, zeros);
```

Stick to `aie::...` unless the AIE API genuinely lacks an equivalent (rare).

---

## ❌ Reading an RTP before the host has written it → race

```python
def core(rtp, of_in, of_out, k):
    scale = rtp[0]                   # may read garbage if host hasn't written yet
```

**Fix**: gate with a `WorkerRuntimeBarrier`.

```python
def core(rtp, barrier, of_in, of_out, k):
    barrier.wait_for_value(1)        # blocks until barrier.set(1) in the sequence body
    scale = rtp[0]
```

---

## ❌ Forgetting `wait=True` on the final drain → host returns before NPU finishes

```python
out_h.drain(c_out)                   # non-blocking (wait defaults to False); host may read stale data
```

For the final output you almost always want `wait=True`. Use `wait=False` only when you're explicitly batching and will synchronize later.

---

## ❌ Hand-pinned tiles that violate column routing (lower-level API)

```python
# BAD — broadcasting across distant columns may exceed switchbox capacity
of = object_fifo("of", tile(0, 2), [tile(3, 2), tile(5, 2), tile(7, 2)], 2, ty)
```

Either prefer the high-level API (placer respects routing), or insert mem-tile forwarders, or use per-column FIFOs.

---

## ❌ Single worker for embarrassingly-parallel work

Spawning one `Worker` on NPU2 uses 1 of 32 cores. Always check if your problem trivially splits (element-wise, batch dimension of a matmul, independent reductions) and use the distribute/join pattern.

---

## ❌ ObjectFifo created in a function-local scope of `core_body`

```python
# BAD — ObjectFifos are design-time objects, not runtime
def core_body(of_in):
    of_temp = ObjectFifo(...)        # error: design topology must be declared outside core_body
```

ObjectFifos, Workers, Kernels, and Buffers are all built at design-construction time. Only `acquire`/`release`/kernel calls and `range_` loops belong inside `core_body`.

---

## ❌ Confusing "argument order" between `core_body` parameters and `fn_args`

```python
def core(of_in, of_out, k, scale): ...

# BAD: scale before kernel
w = Worker(core, [of_in.cons(), of_out.prod(), 5, k])
# GOOD
w = Worker(core, [of_in.cons(), of_out.prod(), k, 5])
```

Positional. There's no kwarg form. Type errors won't be caught until you try to run.

---

## ❌ Loading more than fits in L1

A `bfloat16[4096]` tile is 8 KB. Holding several of those plus the kernel stack can blow L1 (~64 KB). Symptoms: stack overflow at runtime, mysterious crashes.

**Fix**: shrink `TILE`, reduce `depth`, or move large temporaries to a mem-tile ObjectFifo.

When the overflow is because the *kernel* genuinely needs more simultaneous state than fits — several activations alive at once for a wide skip/concat, say — **split the work across two tiles and cascade the partial result** rather than squeezing the existing kernel by dropping precision or unrolling less. Those cuts trade away exactly the correctness or performance you're trying to establish; a tile split costs neither.

---

## ❌ Device name doesn't match the physical hardware → silent all-zero output or timeout

Running an `npu1` xclbin on a Strix (`npu2`) board can return an all-zero buffer **without erroring**. If your output is uniformly zero and the design looks right, check the target before debugging the design:

```bash
xrt-smi examine        # confirm the device matches NPU1 vs NPU2 in your Program(...)
```

`iron.get_current_device()` avoids this entirely — prefer it over hardcoding `NPU1()`/`NPU2()` unless you have a reason to pin.

---

## ❌ Trusting a stale JIT/xclbin cache → your fix appears to do nothing

The JIT cache under `$NPU_CACHE_HOME` (default `~/.npu/cache`) is keyed on a module hash, and incremental builds elsewhere track file mtimes rather than semantic changes. Both can serve yesterday's binary. The symptom is the worst kind: a correct fix that changes nothing, sending you off to debug code that never ran.

**Fix**: when a change doesn't move behavior at all, confirm the artifact actually rebuilt before concluding the fix was wrong — clear the cache directory, or point `NPU_CACHE_HOME` at a fresh path, and re-run. Do this before comparing two configurations, not after they disagree.

---

## ❌ Expecting the DMA to convert dtypes or do arithmetic

Buffer descriptors move addresses, strides, and lengths — and on hardware that supports it, apply compression. They do **not** perform arithmetic or dtype casts. A design that assumes "the DMA will widen these int8s to int16 on the way to L1" has an unbudgeted compute-tile cost hiding in it.

**Fix**: if the dataflow needs a type conversion between two hops, put it in a kernel on a compute tile and account for it in the cost model.

---

## ❌ Missing `event0()` / `event1()` → can't trace performance

You can run trace and get nothing useful unless you bracket the hot region. Add `event0()` before the loop and `event1()` after — both are zero-overhead in release builds and become trace markers when trace is enabled.

---

## ❌ Forgetting `set_saturation` on integer kernels that can overflow

Without it, overflow wraps. Symptoms: huge negative outputs where you expected large positives.

```cpp
::aie::set_saturation(aie::saturation_mode::saturate);
::aie::set_rounding  (aie::rounding_mode::symmetric_inf);
```

Call once at the top of the kernel.

---

## ❌ Signed power-of-2 divide/modulo on the hot path → `__divsi3` + no vectorization

A **signed** `a / 8` is not `a >> 3` (rounding toward zero differs when `a` is negative), so it lowers to a `__divsi3` software call instead of a shift. That external call also acts as a **vectorization barrier for the whole enclosing function** — so a single stray signed divide can leave the entire kernel scalar. Signed `%` by a power of 2 hits the same `__divsi3` helper. Signed `*` by a power of 2 doesn't call `__divsi3`, but a possibly-negative operand still blocks folding `a * 8` into `a << 3`, so hoist it off the hot path too.

```cpp
// BAD: signed operand → __divsi3, blocks vectorization
int n_tiles = channel_count / 8;

// GOOD: unsigned (or provably non-negative) → folds to a shift
const int n_tiles = (uint32_t)channel_count / 8u;
```

Also prefer **`constexpr`** (not `const`) for shapes/strides seeded from the design, and thread sizes in as template params or `-D` defines: only a compile-time literal lets Peano fold divides and address math to shifts. Confirm the call is gone with `llvm-nm build/X.o | grep __div` (should print nothing).

---

## ❌ Constructing IRON primitives outside the `@iron.jit` body → "no active location"

`@iron.jit` runs the function body inside an implicit MLIR context (a thread-local `Location`/`InsertionPoint`); `ObjectFifo`, `Buffer`, `Lock`, `.acquire()`, and kernel calls all read that context when they run. Call one of these from outside an active `@iron.jit` body — a module-level helper, a `@func` pykernel declared inside another function instead of at module scope, or one design calling another design's body directly — and you get:

```
RuntimeError: no active location
```

**Fix**: construct IRON primitives only inside the `@iron.jit` body (or another active MLIR context). `@func` pykernels specifically must be declared at module top level, closing over any shape/dtype constants they need, so they inherit the import-time context:

```python
# GOOD: @func at module scope, closes over VECTOR_SIZE
VECTOR_SIZE = 4096
_LINE_TY = np.ndarray[(VECTOR_SIZE // 4,), np.dtype[np.uint8]]

@func
def passthrough_fn(input: _LINE_TY, output: _LINE_TY, line_width: np.int32):
    for i in range_(line_width):
        output[i] = input[i]

@iron.jit
def passthrough_pykernel(a_in: In, b_out: Out):
    ...
```

Note that `Worker`, `Runtime`, `Program`, and `ObjectFifo` constructors are pure Python and don't touch the context — they're only registered with it later, when `Program.resolve_program()` walks the design. So passing these objects around outside a JIT body is fine; it's the primitives that emit MLIR immediately (`Buffer`, `Lock`, `.acquire()`/`.release()`, kernel calls) that need to run inside one.
