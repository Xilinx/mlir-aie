# aiecc: unabsorbed `main` changes since the fork

Tracking of changes made to the `aiecc` driver (`tools/aiecc/`) and its test
suite (`test/aiecc/`) on `main` **after** our fork point that are **not yet**
reflected in the declarative rewrite on `simplify-aiecc`. Prepared in advance of
a rebase.

- Fork point (merge-base): `8da13e91d8a7b273c096fe87fad50fc2304f6105`
- `main` at time of analysis: `9f58dcd1697`
- Scope: the 14 commits in `MB..origin/main` that touch `tools/aiecc/`, plus one
  test-only dialect change.

Diff of branch-only work: `git diff $(git merge-base main HEAD)..HEAD`.

---

## 1. Rebase blockers — our code will not build/run after rebase

These commits delete symbols/options that our branch still references.

### #3179 — Remove dead AIE/AIEX dialect ops, passes (`72e91b426cf`)
- Deletes the entire `AIEObjectFifoRegisterProcess` pass (239-line
  `lib/Dialect/AIE/Transforms/AIEObjectFifoRegisterProcess.cpp`, its `.td`
  definition, and the `createAIEObjectFifoRegisterProcessPass()` factory).
- Our branch still calls `createAIEObjectFifoRegisterProcessPass()` in
  `tools/aiecc/IRTransforms.h` (~line 331).
- **Action:** remove that call (the monolithic driver dropped it too; the pass
  is dead).

### #3217 — Remove the AIEVec to C++ backend (`78f5a4390c8`)
- Removes the AIEVec C++ backend and, with it, the `target-backend` option on
  `convert-vector-to-aievec`. `target-backend` no longer exists anywhere in
  `include/`/`lib/` on `main`.
- Our branch still emits `convert-vector-to-aievec{... target-backend=llvmir ...}`
  in `tools/aiecc/IRTransforms.h` (~line 316); the pipeline string will fail to
  parse after rebase.
- **Action:** drop the `target-backend=llvmir` token from the pipeline string.

---

## 2. New Peano IR-downgrade logic (each ships a NEW test that must pass)

Our `downgradeIRForPeano` (`tools/aiecc/IRTransforms.h` ~line 179) is at the
fork baseline (handles `getelementptr inbounds nuw`, typed `inf`/`nan`,
`nocreateundeforpoison`). The following four post-fork additions are missing.
Each adds a new test under `test/aiecc/` that will arrive on rebase and must
pass.

### #3214 — Restore align-attribute stripping (`14da22a7d85`)
- Strips `, align <N>` attributes. Retaining them makes Peano's capped-O1 `opt`
  skip vectorizing the matmul reduction loop, ~10x program-memory blowup →
  AIE core-memory overflow.
- New test: `test/aiecc/peano_compat_align_strip.mlir`.

### #3232 — Downgrade LLVM 23 `f0x` float literals (`0a69477883a`)
- Rewrites `f0x<8hex>` 32-bit float literals to the double-widened `0x<16hex>`
  form Peano's LLVM 21 `opt` accepts (token-boundary matched).
- New test: `test/aiecc/peano_compat_f0x_float.mlir`.

### #3241 — Downgrade decimal bfloat16 literals (`afc887fcc9e`)
- Rewrites decimal `bfloat N.NNe+NN` literals to the bit-exact `bfloat 0xR<4hex>`
  form (round-to-nearest-even) that Peano's LLVM 21 can parse.
- New test: `test/aiecc/peano_compat_bfloat_decimal.mlir`.

### #3247 — Downgrade bare `inf`/`nan` in phi instructions (`92e9475ceb4`)
- LLVM 23 omits the type prefix for inf/NaN constants appearing as phi operands
  (e.g. `phi float [ -inf, %entry ]`). Rewrites the bare `-inf`/`inf`/`nan`
  keywords to double-widened hex, using token-boundary checks (the existing
  `replaceTypedLiteral` cannot match these).
- New test: `test/aiecc/peano_compat_phi_inf_nan.mlir`.

---

## 3. Behavioral pass change

### #3243 — Make scf→cf lowering runtime-sequence-aware (`fe8f8c22993`)
- Switches the SCF-to-CF step from the generic `createSCFToControlFlowPass()` to
  `xilinx::AIEX::createAIESCFToControlFlowPass()`, which leaves `scf` inside an
  `aie.runtime_sequence` intact (lowered to a flat NPU instruction stream by its
  own path) while lowering core/host `scf` as usual.
- Our branch still uses the generic `mlir::createSCFToControlFlowPass()` in
  `tools/aiecc/IRTransforms.h` (~line 356).
- **Action:** switch to the AIEX pass.

---

## 4. Performance optimizations (output byte-identical per commit messages)

Not correctness blockers; our declarative engine handles these differently.

### #3211 — Lower NPU instructions once for all devices on the full-ELF path (`7fb70606c97`)
- On the full-ELF / no-expand-load-pdis path, lower NPU instructions once for
  the whole module instead of per device (O(devices) → O(1) lowering). Byte
  identical output.

### #3216 — Build per-core compile slices from a stripped base module (`a2841cb2d22`)
- Instead of serializing the whole multi-core/multi-device module to every
  worker, build a stripped base once (drop runtime sequences and other devices),
  then derive small per-core slices. Byte-identical objects/ELFs; bounded peak
  memory.
- New test: `test/aiecc/per_core_slice_lowering_equivalence.mlir` plus three
  `test/aiecc/Inputs/per_core_slice_lowering_equivalence.*.mlir` fixtures.
- **Action:** confirm our per-core split produces equivalent slices so the new
  equivalence test passes.

### #3250 — Replace `std::async` + busy-wait with `DefaultThreadPool` in `compileCores` (`dc50ead8fb8`)
- Threadpool-based parallel core compilation. **N/A architecturally**: our engine
  executes edges sequentially, so there is no `compileCores` fan-out yet. Related
  gap: `-j` is currently a no-op because the engine does not parallelize per-core
  edges (see §6).

---

## 5. Already matched / effectively no-op on our branch

### #3240 — Make dynamic-objFifos the consistent default (pass + driver) (`0ca8da13438`)
- Driver flag default: **already `true`** on our branch
  (`tools/aiecc/CommandLineOptions.h` ~line 89) — matches.
- Remaining divergence: the pass-level default in
  `include/aie/Dialect/AIE/Transforms/AIEPasses.td` still reads `false` on our
  branch (affects standalone `aie-opt` only; the driver always passes an explicit
  value, so driver output is unaffected).

### #3261 — Default `-j` to 0 (auto-detect CPU count) (`1ba5b5cfb6d`)
- `-j` default change: cosmetic for us — our `-j` is an explicit no-op
  (`tools/aiecc/CommandLineOptions.h` ~line 161).
- Also carries a chesshack path bugfix (`col_col_row` → `col_row`) that does not
  apply: our Chess-downgrade intermediate is the `chess-compat_{0}.ll` edge whose
  `{0}` key (`<dev>_core_<col>_<row>`) is already correct. (There is no separate
  "chesshack" file in the rewrite; it is the same downgrade step, just renamed.)

---

## 6. Standing architectural gap (not a single commit)

- **Parallel core compilation / meaningful `-j`.** The monolithic driver's `-j`
  (`numThreads`) drives an internal thread pool over cores in `compileCores`
  (it is *not* forwarded to `xchesscc`/`clang`/`llc`/`lld`). Our engine is
  sequential, so `-j` currently does nothing. Honoring it requires fanning the
  independent per-core edges (`objects_{0}.o` / `elfs_{0}.elf`) out across a
  thread pool (thread-safe materialization, per-worker `MLIRContext`), analogous
  to #3250.

---

## 7. Mechanical

- **#3231 — Harmonize license headers + mechanical header checks (`cf89594b718`).**
- **#3210 — Add AMD copyright notices for compliance (`a0b5f47372`).**
- Our rewrite's new files carry their own headers; expect trivial header
  conflicts only.

---

## 8. Test-only dialect change (no `tools/aiecc` edit)

### #3225 — [dyn-seq P1.2] Carry npu scalar op fields as SSA operands (`627e3ef673f`)
- Modifies `test/aiecc/` only, reflecting an NPU op-signature change in the
  dialect. Verify our NPU lowering still matches after the dialect change lands.

---

## Rebase heads-up: test-suite conflicts

Besides the 5 clean **new** test files above (which simply appear on rebase),
~47 files under `test/aiecc/` were **modified** on `main` since the fork. Because
our branch also rewrote/adjusted many of the same tests, expect merge conflicts
across the test suite during the rebase.
