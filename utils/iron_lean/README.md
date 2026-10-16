<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# IRON → Lean: proving whole_array for every shape and configuration

*A proof of concept.*

IRON designs are checked today by running them. You pick a shape, run it on
the NPU and compare against numpy. That's a good check, but it covers one shape
of one configuration at a time.

The dynamic whole-array GEMM
([`whole_array.py`](../../programming_examples/basic/matrix_multiplication/whole_array/whole_array.py))
compiles once, then dispatches any `M`, `K`, `N` its `require(...)` guards
accept. On top of that it has a family of compile-time configurations:
- 1 to 8 columns;
- tile sizes `m, k, n`;
- dtypes, which set the kernel's micro-tile `r × s × t`;
- column-major `B` and `C`.

This directory proves, in [Lean 4](https://lean-lang.org), that **every**
member of that family is correct.
- **Data movement:** every core receives exactly the right elements of `A` and
  `B`, in the layout its kernel reads, and every element of `C` is written
  exactly once, by the core that saw the matching row and column.
- **Arithmetic:** with the kernel's declared contract, the written value is
  `(A @ B)[i, j]`, exactly in `ZMod 2^w` for integer dtypes.

The proofs hold for every shape the guards accept and every parameter value
the design allows. Each configuration the examples ship is then tied to them,
by proving that its export from the **live IRON objects** is an instance of the
proven model.

## What is proven

### One schema, symbolic in everything

[`Schema/WholeArray.lean`](IronVerify/IronVerify/Schema/WholeArray.lean)
defines `schema p : Design`, the whole_array design written once as a
function of its parameters:

```lean
structure Params where
  rows cols m k n r s t : Nat
  bColMaj cColMaj : Bool
```

`p.Valid` asks only that every size is positive and that `r ∣ m`, `s ∣ k` and
`t ∣ n`. The theorems in
[`Schema/Theorems.lean`](IronVerify/IronVerify/Schema/Theorems.lean) hold for
**every** valid `p` and every shape the guards accept:

| | In words | Lean |
|---|---|---|
| Guards | The design's `require`s accept exactly `M = rows·m·Mt`, `K = k·Kt`, `N = n·cols·Nt` with the runtime-parameter product `(rows·Mt)·(cols·Nt)` inside `i32` | `guards_iff` |
| T-A, T-B | Core `(row, col)`'s `j`-th A (B) object is the right tile, each element in the micro-tile slot the kernel reads, across both B layouts and the uneven shim split; each core receives exactly `tiles · K/k` objects | `tA`, `tB`, `countA`, `countB` |
| T-C | Every `C[i, j]` has exactly one writer across all `cols` drains: the core whose T-A/T-B tiles cover row `i` and column `j`. No write lands out of bounds | `tC`, `tC_writes` |
| C = A @ B | The drained value equals `Σ_k A[i, k] · B[k, j]` over any commutative semiring | `tMM` |
| Integer dtypes | Exact in `ZMod 2^w` when stores wrap; exact over ℤ if no partial sum leaves the output range when they saturate | `tMM_wrap`, `tMM_sat` |
| Floats | The value is the `kk`-ordered accumulation of the kernel calls (rounding left abstract) | `orderedAcc` |
| Overflow | The compiler's host-buffer guard accepts every transfer exactly when `M·K`, `K·N`, `M·N` fit the buffers; no index arithmetic wraps i64 for `i32` dispatch values | `hostFits_iff`, `noWrap64` |

### Tied to the real designs

A theorem about `schema p` only matters if `schema p` is what IRON builds.
[`instances.py`](instances.py) exports every configuration the examples ship
from its live `Program`: dtypes i8, i16 and bf16, every tile size, both
column-major switches, 1–8 columns on npu2 and 1–4 on npu1. That's 92
designs. Each one comes with a Lean proof that it is an `Instance` of
`schema p`:
- **Fifo graph:** the export's fifo graph equals the schema's, up to walk
  normalization, proved with `normalize_sound`.
- **Streams:** every shim channel moves the same host elements for every
  shape.
- **Guards:** the export's guards accept only those shapes.
- **Kernel layouts:** the kernel layouts match.
- **Widths:** every integer `arith` result of the export's runtime sequence
  fits its type (`i32`, `i64` or `index`), for every shape its guards accept
  and every `i32` dispatch value (`wraps_ok`, from `export.py --wraps`). So
  the exact integer terms the model uses are what the machine computes.

[`Schema/Exports.lean`](IronVerify/IronVerify/Schema/Exports.lean) then
restates T-A, T-B, T-C and C = A @ B (`tMM`, `tMM_wrap`, `tMM_sat`,
`orderedAcc`) for **any** instance. They are
quantified over `M K N : Int`, and the only hypothesis is "the export's own
`cf.assert`s pass". Two worked corollaries in
[`Schema/Examples.lean`](IronVerify/IronVerify/Schema/Examples.lean) are about
real IRON output:

```lean
theorem matmul_3col (hA hB : Nat → ℤ) {M K N : Int}
    (hok : (Instances.npu2_i16_i16_64x64x32_3col.design.seq (M, K, N)).ok = true)
    {i j c q : Nat} (hiM : i < M.toNat) (hjN : j < N.toNat) (hc : c < 3)
    (hq : (… design.host (M, K, N) (cL3 c))[q]? = some (2, ((i * N.toNat + j : Nat) : Int))) :
    ∃ w, (∀ e ∈ Instances.npu2_i16_i16_64x64x32_3col.kernels, accDomain? e.2 = some (.wrap w)) ∧
      ((ValueDomain.read (.wrap w)
          (∑ x ∈ Finset.range K.toNat, hA (i * K.toNat + x) * hB (x * N.toNat + j)) : ℤ) : Wrap w) =
        drainVal … (M, K, N) … c q
```

In words: for the 3-column i16 design IRON generates, and every M, K, N it
accepts, the value drained into `C[i, j]` is the dot product of row `i` of A
and column `j` of B, read in the value domain every core's kernel contract
selects. That domain is not chosen by hand: `domain_3col` proves, by `decide`
on the exported contracts, that it is 16-bit wraparound. `matmul_8col` says
the same for 8 columns.

`check.py` audits every theorem. 334 schema theorems and the 92 instance proofs
use only Lean's standard `propext`, `Classical.choice` and `Quot.sound`. So
there's no `sorry` and no `native_decide` anywhere.

### Where the parameters come from

Nothing about the kernel is restated by hand. The matmul kernel's
[`KernelContract`](../../python/iron/kernels/_common.py) already declares, per
operand, a `TensorLayout` with its micro-tile `block`, its `stream` walk and
its transpose. It also declares the dtypes, the accumulator, the rounding
`setup` and, new in this PR, the saturation behaviour. The exporter reads the
contract off the live `Worker` and emits it.

The Lean value domain follows from it:
- **Integer outputs** with no saturating kernel on the core use
  `ZMod 2^w_out`. We confirmed this on the NPU: i16→i16 overflow wraps,
  element-exact against numpy, and does not saturate.
- **Floats** get the ordered-accumulation statement.

## How it works

```
whole_array.py ─@iron.jit─▶ live Program ─export.py─▶ Lean Design (net, seq M K N, kernel contracts)
                                │                          │
                                │                instances.py: export = schema p  (Instance)
                                ▼                          │
                         sim.py numpy oracle      Schema/Theorems.lean: ∀ p valid, ∀ M K N guarded
                                ▲                          │
                        difftests ──── ironeval ──────────────┘   Schema/Exports.lean: ∀ instance
```

1. **Live objects, not source text.** The generator returns its `Program`
   unresolved, and `@iron.jit` resolves it and keeps it. The fifo graph,
   every tap, the runtime transfers and each core's kernel contracts are read
   through small read-only accessors. Nothing parses Python.
2. **Export** ([`export.py`](export.py)). Fifos become `FifoD`s and links
   `LinkD`s. The runtime sequence becomes a `Seq`: the shim transfers under the
   `scf.if`/`scf.for` structure that `TaskGroup.pipelined` emits, with
   `cf.assert` guards as `.check`s. `M`, `K` and `N` stay symbolic: the staged
   `arith` behind every tap becomes a Lean `Int` term. `--wraps` additionally
   emits a width check for every `arith` result.
3. **Semantics** ([`Walk`, `Seq`, `Net`](IronVerify/IronVerify), about 260
   lines). They say where every stream element comes from:
   - a producer DMA gathers an object onto its stream;
   - a consumer DMA scatters the stream into consecutive objects;
   - links cut a shared object into segments, and a join walks each input's
     segment in turn.
4. **The schema proof** ([`Schema/Proofs/`](IronVerify/IronVerify/Schema/Proofs))
   is split into layers with small interfaces, each its own file:
   - fifo names with symbolic indices (`Names`);
   - per-hop semantics (`NetHops`);
   - mixed-radix and layout arithmetic over symbolic radices (`Arith`);
   - closed-form host streams (`Seq`);
   - the three operand paths (`PathA`, `PathB`, `PathC`);
   - the arithmetic (`Matmul`, `Kernel`) and `Overflow`.
5. **Instances** ([`Schema/Inst/`](IronVerify/IronVerify/Schema/Inst)):
   - **One tactic, `wa_stream`,** proves each export's channels equal to the
     schema's for every shape. It folds the pipelined peels back into one loop
     and closes the walk arithmetic.
   - **Fifo graphs and kernel layouts** are checked by `decide`.

## Why trust the model

The Lean semantics are definitions, so they are checked against independent
code rather than assumed:

| Check | What it compares | Result |
|---|---|---|
| [`difftest_walk.py`](difftest_walk.py) | `Walk` against taplib's own `gather` | 2042 walks |
| [`difftest_design.py`](difftest_design.py) | The exported **symbolic** design, evaluated in Lean, against [`sim.py`](sim.py). The oracle interprets the live runtime sequence of the *specialized* design and chains taplib gathers and scatters | 26 shapes, 1–8 columns, up to 328k elements each, every element's provenance identical |
| [`schema_check.py`](schema_check.py) | `schema p` evaluated against the oracle on the live design | 1920 configurations, npu1 and npu2, every dtype, tile and column-major combination |
| [`difftest_layout.py`](difftest_layout.py) | Lean operand layouts against the contract's `TensorLayout.encode` | 960 operands |
| [`difftest_matmul.py`](difftest_matmul.py) | Lean's per-call matmul in its value domain against the contract's `mm_tile_ref`, with overflowing integer inputs | 156 calls |
| [`matmul_check.py`](matmul_check.py) | The oracle's data movement plus the kernel contract against numpy `A @ B` on random data | used by the mutation suite |
| [`mutations.py`](mutations.py) | Real bugs injected into copies of `whole_array.py` | every mutant caught (below) |
| NPU (Strix Halo) | whole_array `--dynamic` on 34 shapes × 21 configurations: 1–8 columns, col-major, i8, bf16; plus all 11 `run_strix_*` lit tests | all `PASS!` |

The oracle earned its keep early. It caught a model that walked a join's whole
shared object instead of each input's segment.

### Mutation results

Each mutant is one real bug in a copy of `whole_array.py`.
- **The difftest is expected to pass:** it shows the model faithfully
  describes the buggy design.
- **The concrete `A @ B` check and the schema-instance proof must catch it.**
  The report names the part of the instance proof that fails.

| Mutant | Bug | difftest | `A @ B` | schema instance |
|---|---|---|---|---|
| (none) | 1 col, 3 cols, B col-major, as shipped | PASS | PASS | instance |
| `a_layout` | A reaches cores row-major, not in `r × s` blocks | PASS | FAIL | fails: fifo graph |
| `a_permute` | A's repeat dim left outermost, so each row block selects a repeat | PASS | FAIL | fails: stream A |
| `a_repeat` | each A row sent one time fewer than the cores consume it | PASS | FAIL | fails: stream A |
| `a_row_wiring` | memtile split feeds the wrong core row | PASS | FAIL | fails: fifo graph |
| `a_uneven_tap` | A tap assumes the rows split evenly over the shims (wrong only at 3 columns) | PASS | FAIL | fails: stream A |
| `b_col_maj_tap` | column-major B walked k-tile first (wrong only with `b_col_maj`) | PASS | FAIL | fails: stream B |
| `b_permute` | B walked a row of tiles first instead of a column | PASS | FAIL | fails: stream B |
| `c_double_write` | C tap ignores the row block | PASS | FAIL | fails: stream C |
| `split_offset` | memtile split offsets rotated | rejected by IRON at resolve | | |

## Trusted base and assumptions

- **IRON's resolve and everything below it** are trusted: the objectfifo
  passes, BD lowering and aiecc. The model describes what IRON *asks* the
  hardware to do.
- **The kernel's per-call arithmetic** is its declared contract, mirroring
  `mm_tile_ref`, and is checked numerically, not proven against the C++. Its
  layouts, dtypes and saturation are read from the `KernelContract`.
- **The core loop** is assumed: each output tile is zeroed, then accumulated
  over its `K/k` objects in delivery order. It lives in the worker body, which
  isn't exported yet.
- **Per-channel FIFO order:** shim tasks on one channel run in issue order.
- **Host-side facts** are hypotheses, not cf.asserts:
  - the buffer sizes behind `hostFits_iff`;
  - the dispatch values being `i32` (the host ABI) behind `noWrap64` and the
    per-instance `wraps_ok`.
- **Not covered:** deadlock freedom, timing, padding, packet flows. The
  theorem for a parameter value is about `schema p`. It is tied to IRON for the
  92 configurations proved to be instances.

## Found along the way

- **whole_array crashed at 3 columns**, with row 3 left without an A fifo.
  This PR fixes it, with each shim feeding a contiguous run of rows.
  - 3, 5, 6 and 7 columns now work and are NPU-checked.
  - At 1, 2, 4 and 8 columns the data movement is unchanged, element for
    element.
- **The runtime-parameter write `(M/m)·(N/n)` could wrap i32** for shapes
  the guards accepted (e.g. M = 2147483392, N = 2147483616). Before this PR
  only the compiler's dispatch-time host-buffer check prevented it. whole_array
  now has its own `require` (a division, so the check itself cannot wrap). The
  per-instance width proofs then follow from the design's guards alone.
- **Several kernels leave saturation on the core:** filter2d, add_weighted
  and the conv/bn kernels set it and never restore it, so later kernels on that
  core inherit it. Their contracts now say so (`leaves_saturation`).
- **4-byte outputs:** a C buffer near 2³¹ elements gets an address offset of
  about 1.5 · 2³² bytes. The compiler carries it in i64 and accepts it.
  Whether the firmware honours offsets above 4 GiB is untested.

## Reproduce

You need an mlir-aie environment built from this branch, where
`python -c "import aie.iron"` works, plus network access the first time.

```bash
# 1. Lean 4 + Mathlib (pinned by lean-toolchain and lake-manifest.json).
#    Installs elan into utils/iron_lean/IronVerify/.elan unless ELAN_HOME is set,
#    downloads the Mathlib build cache (~5 GB) and builds the project.
utils/iron_lean/setup.sh

# 2. Everything.
python utils/iron_lean/check.py
```

The steps, in order (`--steps` runs a subset):

| Step | What it does | Time |
|---|---|---|
| `export` | re-exports the difftest designs from the live objects, fails if stale | seconds |
| `difftest` | walks and 26 design shapes against the oracle | ~30 s |
| `kernels` | layout codec and matmul arithmetic against the contract | ~5 min |
| `sweep` | `schema p` against the oracle on 1920 configurations | ~16 min |
| `proofs` | builds the schema theorems, audits every axiom | ~2 min cold |
| `instances` | exports the 92 configurations from their live objects, builds their `Instance` and width proofs and the worked examples | ~3 min cold on 32 cores, peak ~45 GB |
| `mutations` | the injected-bug suite | ~2.5 min |

Expected tail:

```
=== summary ===
PASS export (2 s)
PASS difftest (24 s)
PASS kernels (305 s)
PASS sweep (941 s)
PASS proofs (6 s)
PASS instances (11 s)
PASS mutations (150 s)
```

(`proofs` and `instances` are seconds once built; about 2 and 1.5 minutes cold
on 32 cores.)

## Experiment

```bash
cd utils/iron_lean
export ELAN_HOME=$PWD/IronVerify/.elan PATH=$PWD/IronVerify/.elan/bin:$PATH

# Export a configuration and evaluate it in Lean at one shape. This adds
# IronVerify/IronVerify/Designs/MyDesign.lean to the registry; delete it afterwards.
python export.py --name MyDesign --n-aie-cols 5
(cd IronVerify && lake build ironeval)
IronVerify/.lake/build/bin/ironeval design MyDesign 256x64x160 | head

# Compare it with the numpy oracle.
python difftest_design.py --name MyDesign --n-aie-cols 5 --shapes 256x64x160

# Inject a bug and watch the checks catch it (or add your own to MUTATIONS).
python mutations.py --only a_uneven_tap
```

Run `python <script> --help` for each script's flags. To prove something new,
open `IronVerify/` in VS Code with the Lean 4 extension (after
`export ELAN_HOME=...`). Use `lake env lean <file>` for quick checks.

## Layout

| Path | Role |
|---|---|
| `setup.sh`, `check.py` | install the pinned toolchain; run every reproduction step |
| `export.py`, `instances.py` | live IRON `Program` → Lean designs; the 92 instance exports |
| `sim.py`, `difftest_*.py`, `schema_check.py`, `matmul_check.py` | the numpy oracle and the differential checks |
| `mutations.py` | injected-bug suite |
| `IronVerify/IronVerify/{Walk,Seq,Net,Kernel}.lean` | semantics |
| `IronVerify/IronVerify/Schema/WholeArray.lean`, `Spec.lean`, `Defs.lean` | the parametric design and its spec |
| `IronVerify/IronVerify/Schema/Theorems.lean` | the theorems |
| `IronVerify/IronVerify/Schema/Proofs/` | their proofs, one layer per file; `Domain.lean` picks each core's value domain from its exported contract |
| `IronVerify/IronVerify/Schema/Inst/`, `Normalize.lean` | instance checking: `Instance`, `wa_stream`, walk normalization, the width obligations (`Wraps.lean`) |
| `IronVerify/IronVerify/Schema/Instances/` | instance proofs, generated by `check.py --steps instances` (not committed) |
| `IronVerify/IronVerify/Schema/Exports.lean`, `Examples.lean` | the theorems for any instance; the 3- and 8-column corollaries (built after generation) |
| `IronVerify/IronVerify/Designs/` | generated exports the difftests evaluate (do not edit) |
| `IronVerify/IronEval.lean` | `ironeval`: evaluate walks, designs, the schema, layouts |

## Next steps

- **Export the worker body** (its `scf` loops, acquires and the
  runtime-parameter writes), so the core-loop contract is derived rather than
  assumed.
- **Relate the model to the BD lowering**, shrinking the trusted base from
  "IRON's resolve" to the hardware.
- **Other designs.** The semantics and the exporter are design-agnostic. Each
  new design needs a schema and its proof layers, while the instance machinery
  carries over.
- **CI.** A job running `setup.sh` and `check.py`; the Mathlib cache is the
  main cost.
