<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Windows CI example-test parity — design exploration

Status: exploration / pre-PR. Branch `explore/windows-ci-parity` off `main`.

**Relationship to #3453** (Deprecate WSL-based Windows setup): that issue owns
the *policy/docs* deprecation (mark `buildHostWin.md` deprecated, update README,
set a timeline — no code/CI changes). This work owns the *code* side: the WSL
example-scaffolding branches #3453 lists for removal (its step 3) are deleted
here, atomically with the CMake-native path that replaces them — since no CI
covers WSL, they can't be safely removed until that replacement lands. #3453 =
intent; this PR = executes the removal.

**TL;DR:** ~126 `programming_examples`/`programming_guide` tests are skipped on
Windows because they run via GNU `make`. Recommended direction: converge the
example build/run flow on **CMake+Ninja** (idiomatic on both OSes; the host
build is already CMake), retire the `make` orchestrator and the WSL shims, and
keep the Makefiles as thin CMake-delegating wrappers so they stay tested.
Reachable parity: 112 of 126 (14 are Chess/Vitis, never portable to native
Windows). Installing `make` is documented as an interim stopgap only.

## Problem

The native-Windows Ryzen AI CI job (`buildAndTestRyzenAIWindows.yml`) builds
mlir-aie from source and runs the lit suite on real NPU hardware, but it
**skips ~126 `programming_examples` / `programming_guide` tests** gated behind
`REQUIRES: makefile_examples`. That feature is only enabled when GNU `make` is
on PATH **and** `os.name != "nt"` (`python/aie_lit_utils/lit_config_helpers.py`,
`add_makefile_examples_feature`). These `run_makefile*.lit` files invoke
`make -f Makefile` to build and run each design; with no `make` on the Windows
runner they are all unsupported. This is the main Windows↔Linux coverage gap.

## Key finding: the host build is already CMake

Each example `Makefile` just `include`s `programming_examples/makefile-common`
and instantiates two macros:
- `jit_xclbin` — builds the xclbin/insts by running the example's `@iron.jit`
  Python design directly (no host toolchain).
- `build_host_exe` — **already shells out to `cmake` + `cmake --build`**,
  wrapping the example's own `CMakeLists.txt` (39 exist today). So the host
  `test.cpp` is a CMake build on Linux *already*; `make` is only the
  orchestrator, not the compiler.

Consequence: reaching parity does **not** require migrating examples to CMake —
that plumbing exists. It only requires letting the existing Makefiles run on
Windows.

## Inventory (verified)

**126** make-gated lit files (`REQUIRES: makefile_examples`); 124 call `make`
directly (the 2 `run_aot_byo.lit` carry the gate but call no make). By dir:
`programming_examples` 123, `programming_guide` 3, `test/` 0. Across 59 example
dirs.

| Bucket | Count | Portability |
|--------|-------|-------------|
| **A. Pure-Python/JIT** (make just runs a `.py`) | 14 | Trivial — a make-free `%python %S/test.py` variant already works on Windows |
| **B. Host-C++ compile** (make builds `test.cpp` via cmake `build_host_exe`, runs on NPU) | 98 | The bulk. Needs make to run so the existing cmake host-build fires on Windows |
| **C. Chess/Vitis** (`xchesscc`) | 14 | **Never portable** — `AIE_BUILD_CHESS_CLANG` force-disabled on WIN32 |
| D. Other | 0 | — |

- **Reachable parity ceiling: 112** (14 A + 98 B). Hard ceiling of 14 Chess.
- 31 of the 59 gated dirs already ship a non-make `run*.lit`; **28 are
  make-only** (all matmul `tests/` param sweeps, vision, resnet, block_datatypes,
  packet_switch, event_trace, programming_guide sections).
- The non-make variants are **lossy**: they run the design on NPU via the Python
  script and skip the host-C++ `test.cpp` verification the Makefile does. So
  "parity" means keeping the C++ host-runtime coverage (bucket B), not just
  running more `.py` files — which is exactly why enabling `make` (rather than
  converting to make-free lit) is the right lever.
- **Bucket B host builds** link `xrt_coreutil` + `test_utils` (static lib);
  ~5-6 vision/resnet examples also need OpenCV. Centralized in
  `programming_examples/common.cmake` (`target_link_test_utils`). No cxxopts.

## Options considered

| # | Option | Effort | Native Windows? | Verdict |
|---|--------|--------|-----------------|---------|
| 1 | Native Windows `make` under **cmd** | L | no (needs Unix userland) | Dead end — cmd lacks `realpath`/`rm`/`mkdir -p`/`which` |
| 2 | Install `make` + **Git-for-Windows bash** | S–M | partial (bolts on a Unix userland) | Works for CI, but not idiomatic Windows; keeps make+bash dependency |
| 3 | **CMake-native example build** (extend `common.cmake`) | M–L | **yes — cmake+ninja, no make/bash** | **Recommended** |
| 4 | Rewrite lit RUN lines make-free | M/test | partial | Duplicates build logic into 126 lit files; orphans Makefiles |

### Why CMake over "install make"
Installing `make` + Git-bash is the cheapest way to green CI, but it bolts a
Unix userland onto Windows — philosophically the same kind of shim as the WSL
hack we're deprecating, just more native. CMake+Ninja is the **idiomatic,
first-class build system on both Windows and Linux**: `docs/buildHostWinNative.md`
already sanctions it as the Windows prerequisite (`winget Kitware.CMake`), the
main mlir-aie build already uses it, and the example **host builds already run
through CMake today** (`build_host_exe` calls `cmake --build`). Aligning the
whole example flow on CMake removes the make/bash dependency entirely and gives
one build system both OSes exercise natively.

## Recommendation: CMake-native example builds, deprecate make **and** WSL on Windows

Align the example build/run flow on **CMake + Ninja** so a design builds and
runs end-to-end with no `make`, no bash, and no WSL — the idiomatic native
experience on both Windows and Linux. This is a larger change than "install
make," but it is the durable direction and removes two shims at once (make-as-
orchestrator and the WSL `powershell`/`wslpath` hack).

### What already exists vs. the gap
- **Host C++ build: already CMake.** `build_host_exe` runs `cmake --build`
  wrapping each example's `CMakeLists.txt`; `programming_examples/common.cmake`
  centralizes `test_utils`/XRT/HRX/OpenCV linking. No new per-example authoring
  for the host side.
- **The only gap is xclbin/insts generation**, which today lives in make's
  `jit_xclbin` macro — a trivial two lines:
  `python <design>.py --xclbin-path=... --insts-path=...`.

### Plan
1. **Add xclbin/insts generation + a run step to CMake.** Extend `common.cmake`
   with a helper (e.g. `add_aie_design`) that emits an `add_custom_command`
   running the example's `@iron.jit` Python with `--xclbin-path`/`--insts-path`
   (mirroring `jit_xclbin`), wires it as a dependency of the host `test.cpp`
   target, and registers a `ctest` test that runs the host exe through
   `run_on_npu.py`. This makes each example fully buildable+runnable via
   `cmake --build` + `ctest` alone.
2. **Strip the WSL branches** (implements step 3 of issue #3453). Remove the
   WSL-detection forks across the example scaffolding:
   - `programming_examples/mlir_aie_init.cmake:32` — `find_program(WSL NAMES powershell.exe)`
   - `programming_examples/common.cmake:26,102-103` — WSL XRT path fallback + `powershell.exe cmake` comment
   - `programming_examples/makefile-common:36-43, 81, 135-137` — `powershell`/`getwslpath`/`wslpath -w`
   - per-example CMakeLists `if(NOT WSL)` blocks (e.g. vision OpenCV_DIR)
   Native cmake is on PATH and accepts forward-slash paths — no wrapper, no path
   translation.
3. **Point lit at cmake/ctest, not make.** Replace `make -f Makefile` RUN lines
   with a small cmake configure + build + `ctest` (or `%run_on_npu% <exe>`)
   sequence, via a shared lit substitution so the 126 tests don't each duplicate
   it. Enable the gate on Windows (`lit_config_helpers.py:108`) keyed on cmake
   being present (it always is).
4. **Keep the Makefiles as thin wrappers (so they stay tested).** Rather than
   delete the Makefiles (which would break the documented `make run` developer
   UX and the "keep Makefiles tested" ask), reduce each to delegate to the CMake
   flow (`cmake --build` + `ctest`). The Makefile then just calls CMake, Linux
   CI still exercises `make`, and the *actual* build logic lives once in CMake —
   no duplication, no orphaning.
5. **Workflow**: the Windows job's `check-reference-designs` /
   `check-programming-guide` steps pick up the newly-enabled tests; cmake+ninja
   are already provisioned for the main build.

### Why this over "install make"
Installing make greens CI fastest but keeps a Unix-userland dependency on
Windows and leaves the WSL-shaped shims in place. CMake-native removes the
make/bash/WSL dependencies entirely, uses the build system Windows supports
first-class (and the docs already require), and keeps **one** build description
(CMake) that both OSes test — the Makefiles become thin delegating wrappers, so
they remain exercised on Linux without holding duplicate logic.

### Cost / risk
Bigger than install-make: touches `common.cmake` + ~33 per-example CMakeLists
(mostly mechanical — add the design-gen helper, drop WSL branches) and the lit
RUN lines. Do it incrementally (helper first, migrate a few examples, prove on
both OSes, then sweep). The xclbin-gen logic is small and centralizable, so this
is M–L, not the XL it would be if the host build weren't already CMake.

## Parity ceiling / out of scope
- Chess/Vitis (`xchesscc`) examples cannot run on native Windows — permanently
  skipped (14). Toolchain limitation, not a CI gap. Reachable ceiling: 112.
- **WSL support is being removed**, not extended (per #3453) — the
  `powershell.exe`/`wslpath` branches in `makefile-common`,
  `mlir_aie_init.cmake`, `common.cmake`, and per-example CMakeLists are retired
  by this work.

## Phasing for the PR (CMake-native)
1. **Add a design-gen + run helper to `common.cmake`** (`add_aie_design`-style):
   `add_custom_command` runs the example's `@iron.jit` `.py` to produce
   xclbin/insts; wire it as a dep of the host target; register a `ctest` that
   runs the exe via `run_on_npu.py`. Prove it on one example on both OSes.
2. **Migrate examples** to the helper and **strip WSL branches** (per-example
   `if(NOT WSL)`, `common.cmake:26`, `makefile-common:36-43/81/135-137`). Mostly
   mechanical across ~33 CMakeLists.
3. **Reduce Makefiles to thin CMake wrappers** so `make run` still works and
   Linux keeps exercising them, but the logic lives once in CMake.
4. **Point lit at cmake/ctest** via a shared substitution; enable the gate on
   Windows (`lit_config_helpers.py:108`) keyed on cmake presence.
5. **Enable in the Windows workflow**; triage the first run — 14 Chess tests
   stay unsupported; watch OpenCV (vision/resnet) and `run_on_npu.py` quoting.

## Open questions
- `run_on_npu.py` invocation from `ctest` on Windows — confirm quoting.
- Vision/resnet OpenCV dependency on the Windows runner
  (`find_package(OpenCV)`); is it provisioned?
- Exact shared-lit-substitution shape for the cmake configure+build+ctest
  sequence so the 126 tests don't each duplicate it.

## Appendix — interim stopgap: install `make` (if fast CI is needed before the CMake work)
If green Windows CI is wanted *before* the CMake migration lands, `make` can be
installed as a temporary bridge. Minimal changes: provision `make` + bash +
`python3` from **one** unix-like distro (MSYS2, or Git-for-Windows + `make`;
avoid GnuWin32-make + Git-bash, since 25 Makefiles pin `SHELL := /bin/bash`);
add a native-Windows branch to `makefile-common` (lines 36-43: `powershell=`
empty, `getwslpath=echo`; line 81: key `host_exe_suffix` off OS); relax the lit
gate (`lit_config_helpers.py:108`). This is ~6-10 shared lines + provisioning,
but it retains the make/bash dependency and the WSL-shaped shims — so treat it
strictly as a bridge to the CMake-native end state, not the destination.
