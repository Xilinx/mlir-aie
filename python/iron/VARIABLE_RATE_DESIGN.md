# DESIGN.md — `VariableRateFifo`

> **What it adds**: a *single-producer / conditional-forward* primitive
> for variable-rate dataflow that ``ObjectFifo``'s static-rate lowering
> cannot express.
>
> **Sibling primitive**: `PacketFifo` covers the *N:1 multi-producer
> fan-in* side of variable-rate dataflow via the AXI stream-switch
> packet-routing fabric. The two are complementary, not alternatives —
> see "Choice of mechanism" below.

## Problem statement

`ObjectFifo` is a *static-rate* dataflow channel: the lowering pass
(`AIEObjectFifoStatefulTransform.cpp::unrollForLoops`) inspects every
producer / consumer loop, computes
`unrollFactor = LCM(objfifo.size for each acquire op)`, and unrolls
the loop body by that factor so the BD-chain length matches the
FIFO depth. This unroll is correct *if and only if* the
acquire/release pair on the fifo is unconditional within the loop
body — every iteration releases the same number of slots.

The CRISPR PAM-filter wants the producer to inspect the input window,
check the PAM byte, and **forward only ~12.5 % of windows**:

```python
def filter_kernel(in_handle, out_handle):
    in_view = in_handle.acquire(1)
    win = in_view[0]
    if pam_check(win):       # passes ~12.5 % of the time
        out_view = out_handle.acquire(1)
        copy(win, out_view[0])
        out_handle.release(1)
    in_handle.release(1)
```

With vanilla `ObjectFifo`, the lowering pass's LCM-unroll
(`unrollFactor = LCM({2, 2}) = 2`) duplicates the loop body and
expects the duplicated `acquire(1) + release(1)` pair to fire on
every iteration. The conditional skip breaks the assumption: the
producer's lock counter advances asymmetrically, and at runtime the
consumer either deadlocks (waiting for slots that never arrive) or
sees stale data.

windows so the rate stays constant — **the producer always forwards
N=1 slot per iteration**. The DMA-volume saving on the sparse-emit
path is real, but the match-tile cycle saving is forfeit (~7-8x
wasted cycles on zero-multiplication).

## Design space

the static-rate contract:

| | **(a) `release(0)` semantics** | **(b) explicit `discard(n)` API** |
|---|---|---|
| User surface | `release(0)` on producer means "skip" | `discard(n)` on producer marks skip |
| Dialect change | `aie.objectfifo.release` constraint relaxed from `IntMinValue<1>` to `IntMinValue<0>` + lowering interprets size==0 as no-op | None (no dialect change) |
| Lowering change | `release(0)` becomes a no-op in the BD-emit pass; lock counters skip the increment | Loop unroller skips variable-rate fifos from LCM set; existing `dynamicGlobalObjectFifos` runtime-counter machinery handles asymmetric rates |
| Pass complexity | Per-op: every `release` consumer must guard against size==0 | Per-fifo: one boolean discardable attr |
| Backward compat risk | Touches a load-bearing dialect op constraint (every existing release op gets re-verified) | Vanilla ObjectFifo unchanged; new attr ignored by legacy passes |
| Auditability | "Where do we skip?" → grep for `release(0)` (mixed in with regular releases) | Grep for `.discard(` — distinct from regular releases |

**Choice: (b) explicit `discard(n)` API.** Rationale:

1. **No dialect-op constraint change.** `aie.objectfifo.release`'s
   `IntMinValue<1>` is enforced by every consumer of the op
   (verifier, lowering BD-emit, runtime-counter pass). Relaxing it
   means re-validating every code path that branches on
   `release.getSize()`. The `discard(n)` API leaves the dialect
   surface untouched — `discard()` is a Python-only no-op marker;
   no MLIR is emitted for skipped iterations at all.

2. **Auditability.** A reader of the producer kernel sees
   `discard(1)` in the skip-branch and knows the static-rate contract
   is intentionally relaxed. With `release(0)` the skip looks like a
   typo (release-zero is a noisy way to say "don't release").

3. **Composability with existing dynamic-lowering machinery.** The
   pass already has `dynamicGlobalObjectFifos` (runtime-counter-
   based access patterns) for designs that don't fit the LCM-unroll
   model. The variable-rate marker simply opts the fifo into that
   path instead of the LCM path; no new lowering machinery needed.

4. **Failure mode is loud.** If a legacy aie-opt build doesn't know
   the `aie.variable_rate` attr, the pass falls through to LCM-unroll
   and trips on the asymmetric acquire/release count — pass crash or
   silicon wedge, NOT a silent precision drift. (Compare with
   SparseFifo's silent fallback to dense weights when
   `aie.compress_mm2s` is ignored.)

## Choice of mechanism: VariableRateFifo vs PacketFifo

Both `VariableRateFifo` (this primitive) and `PacketFifo`
Choose based on the topology:

| Topology | Use |
|---|---|
| **1 producer**, 1 or more consumers, producer wants to skip slots | `VariableRateFifo` (this primitive) |
| **N producers** (each on a different tile) fanning into 1 consumer; each producer's rate is data-dependent | `PacketFifo` (AXI stream-switch + packet headers) |
| Producer's rate is static; you want compression on the wire | `SparseFifo` |
| Producer's rate is static; cascade BM register transfer | `CascadeFifo` / `AccumFifo` |

The CRISPR PAM-filter early-out path uses **VariableRateFifo** for
the producer-side filter (Tile A → match tiles), and **PacketFifo**
in scenarios where multiple match tiles fan their sparse-emit
records into the joiner tile via packet-header routing. Both
patterns coexist in the same kernel.

## API surface

```python
from aie.iron import VariableRateFifo, ObjectFifo, Worker
from aie.iron.device import Tile

# Construction is identical to ObjectFifo plus the marker.
out_fifo = VariableRateFifo(window_type, name="out", depth=2)

# Producer's kernel function: the discard() marker is the
# auditable counterpart to "just don't call acquire/release in the
# skip branch".
def filter_kernel(in_handle, out_handle):
    in_view = in_handle.acquire(1)
    win = in_view[0]
    if pam_check(win):
        out_view = out_handle.acquire(1)
        copy(win, out_view[0])
        out_handle.release(1)        # forward
    else:
        out_handle.discard(1)        # skip, no MLIR emitted
    in_handle.release(1)

w = Worker(filter_kernel, fn_args=[in_fifo.cons(), out_fifo.prod()],
           tile=Tile(0, 2))
```

## Lowering model

At construction time, `VariableRateFifo` builds a vanilla
`ObjectFifo`. At `resolve()` time, one boolean discardable attribute
is pinned on the lowered `aie.objectfifo` op:

```mlir
aie.objectfifo @vr (%tile12, {%tile33}, 2 : i32) {
    aie.variable_rate = true
} : !aie.objectfifo<memref<16xi32>>
```

The downstream `AIEObjectFifoStatefulTransformPass` consumes this
attr in two places:

### 1. Loop-unroll exclusion (`unrollForLoops`)

The LCM computation walks every `ObjectFifoAcquireOp` in each
producer / consumer `scf.for` loop and adds the target ObjectFifo's
target ObjectFifoCreateOp carries `aie.variable_rate = true`. The
producer's loop is therefore not unrolled on the variable-rate
fifo's account.

If the loop has *other* (fixed-rate) fifo accesses, the LCM-unroll
still runs against those. The variable-rate fifo's accesses fall
through to the runtime-counter path.

### 2. Split-fifo attr propagation

Mirrors the SparseFifo discardable-attr propagation slot
 at line ~2030. When the pass splits an ObjectFifo with
a remote consumer into `(producerFifo, consumerFifo)`, the
`aie.variable_rate` attr is copied to the consumer-side fifo so:

- Diagnostic dumps see the marker on both halves.
- The runtime-counter machinery on the consumer side knows the rate
  is variable.

### 3. Runtime counters (existing machinery — `dynamicGlobalObjectFifos`)

Already present in the pass; routes accesses through
`updateGlobalNextIndex` (a per-fifo per-port runtime counter that
increments on each release). No new machinery added; the variable-
rate marker just opts the fifo INTO this path rather than the LCM
path.

## Tradeoffs

### Memory back-pressure

`VariableRateFifo` keeps `ObjectFifo`'s shared-memory + lock
runtime, so back-pressure is the same as ObjectFifo:

- Producer blocks on `acquire(Produce, n)` if all `n` slots are
  still owned by the consumer.
- Consumer blocks on `acquire(Consume, n)` if no slot has been
  released yet.

Variable-rate semantics interact with back-pressure as follows:
**the producer never back-pressures itself by skipping**. A
`discard(n)` is free at the lock layer (no acquire happens, so no
slot is claimed and no lock counter advances). The consumer's
back-pressure wait is unchanged — it just sees fewer slots.

The opposite-direction risk: if the producer `discards` 100 % of
iterations for a long stretch (no forward at all), the consumer
deadlocks waiting for a slot that never arrives. This is **the
user's correctness obligation**: at the topology level, the
producer's discard rate must not equal 100 % over any window the
consumer waits on. The same obligation exists for
`PacketFifo` (a producer that never sends a packet starves the
consumer).

### Lock semantics

ObjectFifo uses two locks per slot:

- Producer-side write lock: producer acquires for write, fills,
  releases (transferring to consumer's read pool).
- Consumer-side read lock: consumer acquires for read, drains,
  releases (transferring back to producer's write pool).

`VariableRateFifo` does not change the lock topology. `discard(n)`
emits zero `aie.use_lock` ops; the producer simply doesn't acquire
the slot at all. Compared to a hypothetical "release-without-write"
semantic (which WOULD require new lock ops), `discard(n)` is
strictly less invasive.

### Interaction with split-fifo + memtile paths

Tested: the `aie.variable_rate` marker is propagated through the
split-fifo path so consumer-side fifos also
carry it. This is symmetric with the SparseFifo attr propagation
shipped today.

For memtile-mediated paths (consumer through a memtile relay),
the same split-fifo propagation runs at the memtile boundary; the
consumer-side fifo on the memtile carries the marker, and the
runtime-counter machinery handles the asymmetric rate end-to-end.

Note: the marker does NOT propagate through `ObjectFifoLink`
joins / forks (the `link` ops use a different lowering path that
doesn't touch the variable_rate attr). For variable-rate flows
through joins / forks, design at the topology level so each
sub-fifo carries its own marker explicitly.

### Interaction with SparseFifo

`SparseFifo` and `VariableRateFifo` are orthogonal: SparseFifo
flips compression bits on the BD; VariableRateFifo opts out of
LCM-unrolling. A future "variable-rate sparse" combination is a
trivial subclass:

```python
class VariableRateSparseFifo(SparseFifo, VariableRateFifo):
    # MRO ensures resolve() runs both attribute-pinning paths.
    pass
```

future work.

### Interaction with PacketFifo

PacketFifo lowers to `aie.packetflow` ops on the AXI stream switch
fabric. VariableRateFifo lowers to vanilla `aie.objectfifo` (with
the marker attr). The two coexist in the same `aie.device` body
without interference.

### Interaction with the broadcast pattern

`VariableRateFifo` permits multiple consumers (broadcast). All
consumers see the same forwarded subset — the producer's discard
applies symmetrically to every consumer. This matches `ObjectFifo`'s
broadcast semantics (one producer, N consumers all see every slot).

If different consumers should see different subsets of the
producer's output, use multiple `VariableRateFifo`s with separate
producer logic per fifo, or `PacketFifo` (which routes per-packet
to specific consumers via the packet header).

## Validation

Lit tests (in `test/objectFifo-stateful-transform/`):

- `variable_rate_fifo_attr_propagation.mlir` — the
  `aie.variable_rate = true` marker survives split-fifo, lands on
  both producer-side and consumer-side `aie.objectfifo` ops.
- `variable_rate_fifo_skip_unroll.mlir` — a producer loop with a
  variable-rate fifo + a vanilla ObjectFifo: the LCM unroll runs
  against the vanilla fifo only; the variable-rate fifo is
  excluded from the `objFifoSizes` set.

Worked example (in `programming_examples/basic/variable_rate_filter/`):

- End-to-end IRON design with a producer that filters input by
  even/odd index (deterministic predicate, easy to verify on host).
- Producer forwards only even-indexed elements; consumer's expected
  output = first half of input.
- Compiles to xclbin via the same Makefile shape as
  `passthrough_kernel`.

Regression-protect (existing lit tests):

- `non_adjacency_test_AIE2.mlir` — vanilla ObjectFifo
- `plio_test.mlir` — PLIO ObjectFifo (orthogonal feature).
- `repeat_count_test.mlir` — BD chain iteration count
  (orthogonal feature).
- `sparse_fifo_split_attr_propagation.mlir` — today's
  baseline).

(the LCM-unroll exclusion only fires on fifos with the
`aie.variable_rate` attr).

## Future work

1. **Pass-side audit of producer invariant.** Add a
   pre-lowering analysis pass that walks each variable-rate-fifo
   producer's loop body and verifies the
   `acquires_total == releases_total + discards_total` invariant
   per iteration. Today this is the user's obligation; a static
   check would catch typos.
2. **Variable-rate sparse combination.** As noted above; trivial
   subclass once a consumer surfaces.
3. **Memtile-aggregated variable-rate join.** Combining
   N variable-rate producers into a single memtile-buffered
   consumer (with offsets) — currently the marker doesn't propagate
   through `ObjectFifoLink`; would need extending if the topology
   surfaces.
4. **Upstream PR.** Once `T9` validates the primitive end-to-end
   on silicon (CRISPR PAM-filter cycle improvement vs
   filter-late v1), prepare a clean upstream PR. The lowering
   pass change is small; the IRON Python class is well-isolated.

## Cross-references

- `python/iron/variable_rate.py` — primitive class.
- `python/iron/sparse.py` — pattern source (the discardable-attr
- `lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` —
  `unrollForLoops` and the split-fifo attr-propagation block at
  ~line 2030).
- `test/objectFifo-stateful-transform/variable_rate_fifo_*.mlir` —
  lit tests.
- `programming_examples/basic/variable_rate_filter/` — worked
  example.
.
