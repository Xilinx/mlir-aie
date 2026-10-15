<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# IRON → Lean: proving a design's data movement for every shape

*A proof of concept.*

IRON designs are checked today by running them: pick a shape, run it on the
NPU, compare with numpy. That's a good check, but it covers one shape at a time.
The dynamic whole-array GEMM
([`whole_array.py`](../../programming_examples/basic/matrix_multiplication/whole_array/whole_array.py))
compiles once and then dispatches any `M`, `K`, `N` its `require(...)` guards
accept, so there are many more shapes it can run than we will ever test.

This directory takes the design's **live IRON objects**, exports their data
movement to [Lean 4](https://lean-lang.org), and proves, **for every `M`, `K`,
`N` that pass the design's own guards**, that every core receives the right
elements of `A` and `B` in the layout its kernel expects, and that every
element of `C` is written exactly once, by the core that saw the matching row of
`A` and column of `B`.

## What is proven

The theorems are about `whole_array` with `m = 64`, `k = 64`, `n = 32`, `i16`
operands and a fixed column count (1 and 4 are proven). They hold for every
`M = 256·Mt`, `K = 64·Kt`, `N = 32·cols·Nt`: exactly the shapes the design's
own guards accept (`M`, `K`, `N` positive, `M % (4m) = 0`, `K % k = 0`,
`N % (n · cols) = 0`). Stating that equivalence in Lean is a one-line
lemma we have not written yet.

| | In words | Lean |
|---|---|---|
| **T-A** | Core `(row, col)`'s `j`-th `A` object is the `m × k` tile at row block `i0` and column block `kk` (`j = τ · K/k + kk`), each element in the `(r × s)` micro-tile slot the matmul kernel expects. The core receives exactly `tiles · K/k` such objects. | `tA`, `countA` |
| **T-B** | The same for `B`: the `k × n` tile at column block `j0`, for `kk = 0 … K/k − 1`, in kernel layout. | `tB`, `countB` |
| **T-C** | Every `C[i, j]` (`i < M`, `j < N`) has exactly one writer: core `(row, col)`, output tile `τ`, slot `(a, b)`, with `(i, j) = (i0 + a, j0 + b)` for the **same** `i0`/`j0` as in T-A and T-B. No write lands out of bounds. | `tC`, `tC_writes` |

Together these say the core that produces `C[i, j]` is exactly the core that saw
row `i` of `A` and column `j` of `B`, all `K/k` pieces of them, in order. With
the kernel's contract (`C_tile += A_tile · B_tile` in its blocked layouts), that
is `C = A @ B`; [`matmul_check.py`](matmul_check.py) checks that last step
concretely, and proving it in Lean is the natural next step.

The theorems live in
[`Proofs/WholeArrayC1.lean`](IronVerify/IronVerify/Proofs/WholeArrayC1.lean)
(one column of four cores) and
[`Proofs/WholeArrayC4.lean`](IronVerify/IronVerify/Proofs/WholeArrayC4.lean)
(four columns, sixteen cores, where T-C's uniqueness spans all four drains).
The specification they are checked against, the tile each core owns and the
kernel's operand layouts, is hand-written in
[`Spec/WholeArray.lean`](IronVerify/IronVerify/Spec/WholeArray.lean). The
two-column design is exported and differentially tested but not proven.

`python check.py --steps proofs` builds every proof module and audits the axioms
of every theorem in the `IronVerify` namespace: only Lean's standard `propext`,
`Classical.choice` and `Quot.sound` are allowed, so a `sorry` or a
`native_decide` anywhere fails the check.

## How it works

```
whole_array.py ──@iron.jit──▶ live Program ──export.py──▶ IronVerify/Designs/WholeArrayC*.lean
  (ObjectFifos, links, taps,      (Runtime.transfers,          net : Net        seq M K N : Seq
   Workers, runtime sequence)      ObjectFifo/link accessors)        │
                                                                     ▼
                                             Walk / Seq / Net semantics (IronVerify/*.lean)
                                                                     │
                                         ┌───────────────────────────┴──────────────────────┐
                                         ▼                                                  ▼
                              Proofs/*.lean: ∀ M K N, guards →           ironeval: evaluate at one shape
                              T-A, T-B, T-C                               ◀── difftest ──▶ sim.py (numpy oracle)
```

1. **Live objects, not source text.** The generator returns its `Program`
   unresolved; `@iron.jit` resolves it and keeps it, so
   `design.specialize(...).compilable.program` hands back the fifo graph, every
   tap and the runtime sequence the body emitted (`Runtime.transfers`). Nothing
   parses Python.
2. **Export** ([`export.py`](export.py)). Each fifo becomes a `FifoD` (object
   sizes, `to_stream` / `from_stream` walks, repeat count, endpoints), each
   split/join/forward a `LinkD`, and the runtime sequence a `Seq`: shim
   transfers under the `scf.if` / `scf.for` structure that issued them, with
   `cf.assert` guards as `.check`s. `M`, `K` and `N` stay symbolic: the staged
   `arith` behind every tap becomes a Lean `Int` term.
3. **Semantics** ([`IronVerify/IronVerify/`](IronVerify/IronVerify)).
   - `Walk.lean`: a tensor access pattern; stream position `p` reads
     `offset + Σ iᵈ · strideᵈ`, as taplib's `TensorAccessPattern` does.
   - `Seq.lean`: the host elements each shim channel moves, in issue order.
   - `Net.lean`: where every stream element comes from. A producer DMA gathers
     an object onto its stream, a consumer DMA scatters the stream into
     consecutive objects, links cut a shared object into segments, a join's
     output walks each input's segment in turn.
   - `Lemmas.lean`: walk indexing, mixed-radix digits, uniform concatenations.
4. **Proofs** ([`IronVerify/IronVerify/Proofs/`](IronVerify/IronVerify/Proofs)):
   the theorems above, about the exported `Design`, by unfolding the semantics
   and closing the index arithmetic.

### Why trust the model

The Lean semantics are definitions, so they are checked against independent
code rather than assumed:

- [`difftest_walk.py`](difftest_walk.py): `Walk` against taplib's own
  `gather` on ~2000 random access patterns plus every `whole_array` tap.
- [`difftest_design.py`](difftest_design.py): the exported **symbolic** design,
  evaluated in Lean at a concrete shape, against [`sim.py`](sim.py), a numpy
  oracle that interprets the live runtime sequence of the design *specialized*
  to that shape and chains taplib gathers and scatters along the fifo graph (the
  convention of `test/python/taplib/stream_simulation.py`). Every element's
  provenance must match: which host element or core object slot it came from.
  The oracle already earned its keep: it caught an early model that walked a
  join's whole shared object instead of each input's segment.
- [`matmul_check.py`](matmul_check.py): the oracle's data movement plus the
  kernel contract computes `A @ B` on random integers.
- [`mutations.py`](mutations.py): real bugs injected into copies of
  `whole_array.py` must be caught (see below).
- Hardware: the same design passes its numpy check on a Strix Halo NPU
  (npu2) with this branch's hooks, one compiled design per column count:
  `whole_array.py --n-aie-cols 1 --dynamic 256x128x64 512x128x64 768x192x96 1280x64x64`
  and `--n-aie-cols 4 --dynamic 512x512x512 256x768x512 512x128x256 1280x64x128`
  print `PASS!` for every shape.

## Trusted base and assumptions

The theorems are only as strong as what they rest on. Stated plainly:

- **IRON's resolve and everything below it** (the objectfifo passes, BD
  lowering, aiecc) are trusted: the model describes what IRON *asks* the
  hardware to do.
- **The kernel** is a contract: it reads its operands in the
  `(r, s)` / `(s, t)` / `(r, t)` blocked layouts and accumulates.
- **Per-channel FIFO order**: tasks on one shim channel run in issue order, and
  a core consumes objects in the order its DMA delivers them.
- **No overflow**: the exporter drops `arith` casts (`extsi`, `index_cast`, …),
  so terms are exact integers; a design whose index arithmetic overflows `i32`
  / `i64` is outside the model.
- **Semantics vs silicon**: the model is validated against the oracle and a
  hardware run, not proven against the hardware.
- **Not covered**: deadlock freedom, timing, kernel correctness, padding,
  packet routing.

## Reproduce

You need an mlir-aie environment built from this branch, where
`python -c "import aie.iron"` works (the exporter relies on the small IRON
accessors this branch adds), plus network access the first time.

```bash
# 1. Lean 4 + Mathlib (pinned by lean-toolchain and lake-manifest.json).
#    Installs elan into utils/iron_lean/IronVerify/.elan unless ELAN_HOME is set;
#    downloads the Mathlib build cache (~5 GB) and builds the project.
utils/iron_lean/setup.sh

# 2. Everything: export staleness, differential tests, proofs, mutations.
python utils/iron_lean/check.py
```

`check.py --steps export difftest proofs mutations` runs a subset;
`--elan-home` points at an existing elan install. Expected tail:

```
=== summary ===
PASS export
PASS difftest
PASS proofs
PASS mutations
```

The `difftest` step should print `PASS` for 11 shapes across 1, 2 and 4
columns, for example `PASS WholeArrayC2 1280x64x192: 246000 lines, ok true, 0
missing, 0 extra`, plus `2042/2042 walks agree`. Approximate times on a Strix Halo
workstation: setup a few minutes (dominated by the Mathlib cache download),
difftest ~10 s, proofs ~20 s from scratch (seconds when cached), mutations
~1 min.

The committed exports in `IronVerify/IronVerify/Designs/` target npu2
(`export.py --device`, default `npu2`, pinned so no NPU is needed to export);
`check.py --steps export` regenerates them and fails if they differ.

## Experiment

```bash
cd utils/iron_lean
export ELAN_HOME=$PWD/IronVerify/.elan PATH=$PWD/IronVerify/.elan/bin:$PATH

# Export another configuration (any n_aie_cols, or a specialized --shape MxKxN).
python export.py --name WholeArrayC8 --n-aie-cols 8
(cd IronVerify && lake build ironeval)

# Evaluate it in Lean at one shape: every object each core receives, every C write.
IronVerify/.lake/build/bin/ironeval design WholeArrayC8 1024x64x256 | head

# Compare it with the numpy oracle.
python difftest_design.py --name WholeArrayC8 --n-aie-cols 8 --shapes 1024x64x256

# Inject a bug and watch the checks catch it (or add your own to MUTATIONS).
python mutations.py --only a_permute
```

To prove something new, open `IronVerify/` in VS Code with the Lean 4
extension (after `export ELAN_HOME=...`), start from the existing proof file,
and use `lake env lean IronVerify/Proofs/<File>.lean` for quick checks.

### Mutation results

Each mutant is one real bug in a copy of `whole_array.py` (one column, shapes
`256x128x64` and `512x128x64`). The difftest is expected to *pass*: it shows
the model faithfully describes the buggy design. The concrete `A @ B` check and
the proofs are what must catch it.

| Mutant | Bug | difftest | `A @ B` | proofs |
|---|---|---|---|---|
| (none) | `WholeArrayC1` as shipped | PASS | PASS | build |
| `a_layout` | A reaches cores row-major, not in `r × s` blocks | PASS | FAIL | fail |
| `a_permute` | A's repeat dim left outermost, so each row block selects a repeat | PASS | FAIL | fail |
| `a_repeat` | each A row sent one time fewer than the cores consume it | PASS | FAIL | fail |
| `a_row_wiring` | memtile split feeds the wrong core row | PASS | FAIL | fail |
| `b_permute` | B walked a row of tiles first instead of a column of tiles | PASS | FAIL | fail |
| `c_double_write` | C tap ignores the row block (caught at 2 row blocks) | PASS | FAIL | fail |
| `split_offset` | memtile split offsets rotated | rejected by IRON at resolve | | |

## Layout

| Path | Role |
|---|---|
| `setup.sh` | install the pinned Lean toolchain and Mathlib, build |
| `check.py` | one driver for every reproduction step |
| `export.py` | live IRON `Program` → `IronVerify/Designs/<Name>.lean` |
| `sim.py` | numpy oracle over the specialized design |
| `difftest_walk.py`, `difftest_design.py` | Lean evaluation vs taplib / the oracle |
| `matmul_check.py` | oracle data movement + kernel contract = `A @ B` |
| `mutations.py` | injected-bug suite |
| `IronVerify/IronVerify/{Walk,Seq,Net}.lean` | semantics |
| `IronVerify/IronVerify/Lemmas.lean` | generic lemmas |
| `IronVerify/IronVerify/Designs/` | generated exports (do not edit) |
| `IronVerify/IronVerify/Spec/WholeArray.lean` | hand-written spec: tile ownership, kernel layouts |
| `IronVerify/IronVerify/Proofs/` | the theorems |
| `IronVerify/IronEval.lean` | `ironeval`: evaluate walks and designs |

## Limitations and next steps

- **`C = A @ B` in Lean**: state the kernel contract and derive the matmul from
  T-A/B/C.
- **More of `whole_array`**: the column-major `B`/`C` variants, more column
  counts, other dtypes (other `mac_dims`).
- **Other designs**: the exporter is generic over fifos, links and runtime
  sequences, but only `whole_array` has a loader and proofs; padding, packet
  flows and core-to-core streams are not modeled.
- **Overflow obligations** for the dropped casts.
- **Lowering**: relate the model to the BD chains the objectfifo passes emit,
  shrinking the trusted base.
- **CI**: a job that runs `setup.sh` and `check.py` (Lean + Mathlib cache is the
  main cost).
