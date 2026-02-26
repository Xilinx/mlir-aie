# C++ aiecc Feature Parity Plan

## Goal
Build on top of `aiecc-aiesim-host-support` to create a feature-complete C++ aiecc that
effectively deprecates `aiecc.py`. All existing tests must pass using the new thin Python
wrapper (`aiecc.py`) that delegates to the C++ `aiecc` binary.

## Features Added to C++ aiecc

- Parallel core compilation (`-j N` / `--nthreads N`); `-j 0` auto-detects, `-j 1` default
- Unified compilation (`--unified` / `--no-unified`)
- AIEBU library integration (direct ELF generation)
- Bootgen library integration (direct PDI generation)
- Host compilation (`--compile-host` via `aie-translate --aie-generate-xaie`)
- AIE simulation support (`--aiesim`)

## Current Test Status (28 tests)

### Passing (24)
All cpp_* tests, library_integration, buffers_xclbin, only_insts, fallback_*, simple_aie2,
simple_xclbin, cpp_xchesscc_*, cpp_aiesim

### Failing (2)
- [ ] `generate_pdi.mlir` — PDI written to tmpDir instead of CWD; bootgen library output missing `bootgen` keyword
- [ ] `simple.mlir` — AIE1 target missing `chess_intrinsic_wrapper.ll`

### Unsupported (2)
- `repeater_generation.mlir` — unconditionally disabled
- `cpp_npu_and_xclbin.mlir` — requires xrt LIT feature

## Fixes Needed

### Fix 1: PDI output path (generate_pdi.mlir)
**Problem:** C++ aiecc writes PDI to `tmpDirName/<pdi-name>` but the old Python aiecc.py
writes to `<pdi-name>` relative to CWD. Test `generate_pdi.mlir` checks `ls | grep MlirAie`
in CWD and expects `bootgen {{.*}} MlirAie0.pdi` in output.

**Fix:** Change `pdiPath` construction to use `pdiFileName` directly (CWD-relative) instead
of prepending `tmpDirName`. When using bootgen library, still print the equivalent bootgen
command for verbose output compatibility.

**Files:** `tools/aiecc/aiecc.cpp` lines 3864-3967

### Fix 2: AIE1 chess compilation (simple.mlir)
**Problem:** `simple.mlir` uses `xcvc1902` (AIE1). The chess compilation flow looks for
`aie_runtime_lib/AIE/chess_intrinsic_wrapper.ll` which doesn't exist (only AIE2/AIE2P built).
The old Python aiecc.py didn't use `chess_intrinsic_wrapper.ll` — it had a different chess
compilation flow.

**Fix:** Make `chess_intrinsic_wrapper.ll` linking optional — skip it with a warning when
the file doesn't exist for the target, matching old Python behavior.

**Files:** `tools/aiecc/aiecc.cpp` lines 900-911

### Fix 3: Verbose output format parity
**Problem:** Some tests check for specific verbose output patterns like bare command names
at line start. The C++ aiecc sometimes prefixes with `Executing:` or uses different format.

**Fix:** Ensure verbose output matches Python aiecc.py format — print commands without
prefix, print `bootgen` command when generating PDI (even via library).

## Python Wrapper Changes

Replace 2,164-line `aiecc.py` with ~180-line thin wrapper that:
- Delegates all work to C++ `aiecc` binary
- Preserves deprecated `run()` API with warning
- Host compilation flags emit deprecation warnings
- Filters `-I`, `-L`, `-l`, `-o` flags

## Next Steps

1. Fix PDI output path (CWD-relative)
2. Fix bootgen verbose output (print command even when using library)
3. Fix AIE1 chess_intrinsic_wrapper.ll (make optional)
4. Rebuild + ninja install
5. Run full test suite with Vitis 9999 + XRT
6. Verify all 28 tests pass (except pre-existing unsupported)
