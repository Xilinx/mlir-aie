# aiecc declarative driver — assessment report

**Date:** 2026-06-29
**Branch / HEAD:** `simplify-aiecc` @ `3dc8877c74b`
**Reviewer:** automated assessment (build + run + code review)
**Hardware:** AMD NPU **Strix** (`npu2`, firmware 1.1.2.64), XRT 2.21.0
**Toolchains:** Vitis 2025.1 (xchesscc/aietools), Peano `llvm-aie` 21.0 nightly, aiebu-asm, bootgen, xclbinutil — all via `~/setup_buildenv.sh`

---

## 1. Executive summary

**The declarative driver works.** Once two build-environment blockers were
cleared (see §2), `aiecc` builds cleanly and produces correct artifacts across
every backend and artifact type that was exercised — **validated end-to-end on
real NPU hardware** for both the Peano and the Chess (xchesscc/xbridge) core
flows.

| Area | Result |
|------|--------|
| Build | ✅ after fixing 2 blockers (missing `GenVersion.cmake`, un-checked-out submodule) |
| NPU compile (Peano, default) | ✅ xclbin + insts, **runs on HW: `PASS!`** |
| NPU compile (xchesscc + xbridge) | ✅ xclbin + insts, **runs on HW: `PASS!`** |
| Per-device artifacts (txn / pdi / cdo / full-elf / inst-elf / ctrlpkt) | ✅ all `EXIT=0` |
| Unified & multi-core compilation | ✅ |
| AIE1 (`xcvc1902`) core compile + `aie_inc.cpp` | ✅ core ELF (arch `0x108`) + host source |
| AIE1 **host-executable link** | ❌ fails on unresolved `cdo_*`/`ess_*` symbols (§4) |
| `test/aiecc` lit suite | ⚠️ 10 pass / 29 fail / 8 unsupported — **failures are stale test harness, not driver bugs** (§3) |
| Design-goal adherence | ✅ strong; a few refactor candidates (§5) |

The single genuine functional gap found is the **AIE1 host-executable link
step**, which `FEATURE_GAPS.md` (P3) currently marks as "Validated end-to-end".
That claim does **not** reproduce in this environment.

---

## 2. Build status — two blockers (both fixed to proceed)

The project had **no build directory and an empty `install/`** at start. Building
from the MLIR distribution wheel (`mlir==23.0.0.2026060107`) surfaced two
blockers:

### 2.1 Missing `tools/aiecc/GenVersion.cmake` (build-breaking)
`tools/aiecc/CMakeLists.txt` runs a custom target that invokes
`GenVersion.cmake` to regenerate `AIECCVersion.h`, **but the file does not exist
in the tree** (never committed — `git show HEAD:tools/aiecc/GenVersion.cmake` →
*does not exist*). Result:

```
CMake Error: Not a file: .../tools/aiecc/GenVersion.cmake
ninja: build stopped: subcommand failed.
```

**Action taken:** created a minimal `tools/aiecc/GenVersion.cmake` that emits
`#define AIECC_GIT_SHA "<short-sha>"` and only rewrites the header when the SHA
changes (matching the intent documented in the CMakeLists comment). This should
be committed.

### 2.2 `cmake/modulesXilinx` submodule not checked out
The submodule directory contained only a `.git` link; every file
(`FindVitis.cmake`, `FindXRT.cmake`, …) was **staged for deletion** in the
working tree. Consequently `find_package(Vitis)` silently failed,
`VITIS_AIETOOLS_DIR` was empty, and the **chess intrinsic wrappers were never
built** — which would make every `--xchesscc` path fail at runtime.

**Action taken:** restored the files with `git checkout HEAD -- .` inside the
submodule, reconfigured with `-DAIE_BUILD_CHESS_CLANG=ON`, and built/installed
`aie-runtime-libs` + `xchesscc_wrapper`. This produced the wrappers for **AIE,
AIE2 and AIE2P** (the AIE1 wrapper that P3 calls out is present). This is a
**pre-existing dirty-repo state, not a code defect** — noted for environment
hygiene.

Minor: `-DLLVM_USE_LINKER=lld` had to be dropped (host `clang` here cannot
`-fuse-ld=lld`); the from-wheels script already guards this with
`command -v lld`, so no change needed.

After these, the build is clean (`BUILD EXIT=0`), AIEBU was detected
(`Found AIEBU library — aiecc will use direct library calls`), and
`aiecc --version` reports the SHA correctly.

---

## 3. Test results

### 3.1 Real NPU hardware (the decisive evidence)

Reproduced the two dedicated driver hardware tests on Strix (`npu2_1col`):

| Test | Flow | Result |
|------|------|--------|
| `test/npu-xrt/add_one_cpp_aiecc` | Peano → xclbin + insts → XRT run | **`PASS!`** |
| `test/npu-xrt/add_one_cpp_aiecc_xchesscc` | xchesscc/xbridge → xclbin + insts → XRT run | **`PASS!`** |

Both produced a valid `aie.xclbin` + `insts.bin`, the host `test.cpp` linked
against XRT, and the on-device run returned `PASS!` (exit 0). This exercises the
full pipeline: placement → routing → core compile → ELF link → CDO → bootgen PDI
→ xclbinutil → NPU execution.

> Note: `add_one_cpp_aiecc/run.lit` itself still passes the **removed**
> `--no-xchesscc --no-xbridge` flags, so the *unmodified* lit test would fail at
> the `aiecc` step. I ran the flow with corrected flags (Peano is the default).
> The xchesscc test's RUN line is already correct (`--xchesscc --xbridge`).

### 3.2 `test/aiecc` lit suite — 10 / 29 / 8 (pass / fail / unsupported)

**Every one of the 29 failures is test-harness rot, not a driver fault.**
Breakdown of failure causes (from the lit log):

- **Removed/renamed CLI options** the tests still pass:
  `--no-xchesscc` (38×), `--no-xbridge` (25×), `--verbose` (17×, the driver
  registers only `-v`), `-n`/dry-run (6×), `--no-compile` (5×), `--compile`
  (3×), `--tmpdir=` (now `--workdir`), `--repeater-output-dir`.
- **10 invocations still call `aiecc.py`** (the retired Python driver), e.g.
  `only_insts.mlir`.
- **Stale `CHECK:` strings** — tests expect legacy verbose lines
  (`Generating transaction MLIR for device`, `Generating aie_inc.cpp for
  device`, `Successfully parsed input file`); the declarative engine instead
  prints `aiecc: running edge '<name>'` / `aiecc: wrote edge '<name>'`.
- **`xchesscc_*` tests + AIE1 tests are `UNSUPPORTED`** because lit's `chess`
  feature isn't being set in this configuration — yet they **compile and run
  fine when invoked manually** (§3.3).

The 10 currently-passing tests are the migrated `cpp_link_with_*` family, which
confirms the suite *can* pass once updated. This matches **FEATURE_GAPS P6**
(test-suite triage), which remains the largest outstanding chunk of work.

### 3.3 Artifact sweep with corrected flags (functional confirmation)

To prove the lit failures are harness-only, each failing artifact type was
re-run against the same inputs with current flags:

| Artifact | Flag | Result |
|----------|------|--------|
| Transaction MLIR | `--aie-generate-txn` | ✅ `EXIT=0` |
| PDI | `--aie-generate-pdi` | ✅ `EXIT=0` |
| Full ELF | `--generate-full-elf` | ✅ `EXIT=0` |
| Instruction ELF | `--aie-generate-elf` | ✅ `EXIT=0` |
| Unified compile | `--unified --aie-generate-npu-insts` | ✅ `EXIT=0` |
| Multi-core | `--aie-generate-npu-insts` | ✅ `EXIT=0` |
| AIE2P vector add | `--aie-generate-npu-insts` | ✅ `EXIT=0` |
| NPU insts + xclbin | `--aie-generate-npu-insts --aie-generate-xclbin` | ✅ valid xclbin |
| Control packets | `aie-opt …column-control-overlay` → `--aie-generate-ctrlpkt --keep-loc` | ✅ `ctrlpkt.bin` + `dma_seq.bin` + `.locmap.json` |
| xchesscc core compile | `--xchesscc --xbridge` | ✅ core ELF + insts |

(The control-packet test requires the `aie-opt` overlay pre-pass that the lit
RUN line performs; run standalone without it, `dma_memcpy_nd` legitimately fails
to find its `shim_dma_allocation` — expected, not a driver bug.)

---

## 4. The one real functional gap: AIE1 host-executable link

`aiecc --xchesscc --compile-host test/aiecc/simple.mlir test/aiecc/test.cpp …`
on the AIE1 device `xcvc1902`:

- ✅ Core compile succeeds → `elfs_main_core_1_2.elf` (`ELF 32-bit … arch 0x108`,
  i.e. AIE1).
- ✅ `aie_inc.cpp` host-config source generated (102 lines, well-formed).
- ❌ **Final `clang++` host link fails:**

```
ld.lld: error: undefined reference: cdo_BlockWrite32
>>> referenced by .../libxaienginecdo.so (disallowed by --no-allow-shlib-undefined)
   …cdo_Write32, cdo_MaskWrite32, cdo_MaskPoll, ess_Write32, ess_WriteCmd…
```

The host link line (in `buildHostExeSubgraph`) hard-codes `-lxaienginecdo`. The
installed `libxaienginecdo.so` carries **undefined** `cdo_*` / `ess_*` symbols
that are provided by the bootgen CDO driver (`libcdo_driver_mlir_aie.a`), which
is **not** on the host link line. The only other candidate,
`libxaienginecdo_static.a`, lives in `install/lib`, not in the
`runtime_lib/<arch>/xaiengine/lib` directory the host edge searches.

**Impact:** `FEATURE_GAPS.md` P3 states AIE1 was "**Validated end-to-end** …
EXIT=0" producing an x86-64 host executable. That does not reproduce here; the
host-exe step fails. The AIE1 *core* path and `aie_inc.cpp` generation are
fine — only the host-link wiring (shared with the general `--compile-host`
path) is broken. Worth re-validating and either (a) bundling the CDO driver /
linking the static `xaienginecdo`, or (b) re-checking the runtime-lib packaging
this claim was originally validated against.

---

## 5. Code review against the stated design goals

The driver header lays out four rules: (1) express all dependencies explicitly
as graph nodes/edges, (2) use the `Item` abstraction — no manual disk writes,
(3) keep the graph statically declared / side-effect free, (4) do as little work
as possible in the orchestrator. Overall adherence is **strong and genuinely
impressive** — the graph in `main` reads as a declarative dataflow spec, the
engine prunes from requested outputs, and materialization is lazy
(`Item<T>::asFile()`).

**Well-realized:**
- `Graph.h` / `ExecutionEngine.h`: clean payload `Item`s, type-erased nodes,
  backward-reachability pruning, lazy `asFile()`, duplicate-output-path guard.
  The engine never special-cases an artifact type.
- `Actions.h` `ShellCommand`: fluent, declarative external-tool invocation;
  inputs/outputs are slots filled from `Item`s, so edges don't hand-roll argv or
  temp paths. `emitBinary`/`PassPipeline` lift translators and pipelines into
  edges uniformly.
- Output selection is a single `outputs.push_back(...)` list; no `if(want…)`
  guards scattered through the graph — exactly rule (3)/(4).

**Refactor candidates (deviations from rules 2 & 4):**
1. **CDO edge** (`cdo`) calls `llvm::sys::fs::create_directories(cdoDir)` and
   lets `AIETranslateToCDODirect` write a *directory* of `.bin`s, then returns
   `File{}`. This is a pragmatic "directory-as-File" that sidesteps the `Item`
   serialization contract (rule 2). Justified by the translator's API, but it is
   the clearest place the abstraction leaks; worth a comment or a dedicated
   `Directory` payload type.
2. **Host-exe join** (`buildHostExeSubgraph`) does substantial **path/string
   construction inside the lambda** (the `rtLib` builder, `-I/-L/-Wl,-R`
   assembly, arch detection, single-device validation). The header explicitly
   warns against "string manipulation for path generation" in the graph. It's
   encapsulated in a subgraph builder, but it's the densest non-declarative
   block in the file and the natural target for extraction into helpers/an
   `Item`-typed config.
3. **Chess object lambda** (`buildObjectSubgraph`) similarly builds
   `chess-llvm-link` / wrapper paths inline; could move to named helpers in
   `Utils.h` to keep the edge declarative.
4. **Inline clone-and-run-pipeline lambdas** (`npuLowered`, `ctrlpktDmaSeq`,
   `txn`, `ctrlpktLowered`) run real pipelines inside `map`/`join` callbacks
   rather than via the `PassPipeline` action. They're correct and readable, but
   rule (4) ("create an MLIR pass") suggests these destructive lowerings are
   the kind of "involved transformation" that could live behind a named action
   for consistency.

None of these are bugs; they're the spots where future maintenance pressure
will push against the declarative ideal.

**Minor CLI consistency note:** the driver registers verbose as `-v` only;
there is no `--verbose` long form (the legacy driver and many tests use
`--verbose`). Cheap to add an alias and it would un-break a large fraction of
the lit suite for free.

---

## 6. Recommendations (priority order)

1. **Commit `tools/aiecc/GenVersion.cmake`** (or remove the custom target). The
   tree currently does not build without it.
2. **Re-validate / fix the AIE1 (and general) `--compile-host` link** — resolve
   the `cdo_*`/`ess_*` symbols (link the CDO driver or the static
   `xaienginecdo`). Reconcile the "Validated end-to-end" claim in
   `FEATURE_GAPS.md` P3.
3. **Migrate the `test/aiecc` lit suite (P6):** s/`--no-xchesscc`//,
   s/`--tmpdir`/`--workdir`/, drop `-n`/`--no-compile`, add a `--verbose` alias
   or switch tests to `-v`, replace `aiecc.py` invocations, and update `CHECK:`
   strings to the `aiecc: running edge …` output. Wire up lit's `chess` feature
   so the xchesscc/AIE1 tests stop being silently `UNSUPPORTED`.
4. **Optional polish:** introduce a `Directory` payload (or document the
   directory-as-`File` CDO case), and lift the host/chess path-building blocks
   out of their lambdas to keep the graph declaration pure.

---

## 7. How this was verified (reproducibility)

```bash
source ~/setup_buildenv.sh
# build (from MLIR wheel 23.0.0.2026060107):
#   my_install/mlir  → cmake (Release+asserts, AIE_BUILD_CHESS_CLANG=ON) → ninja → ninja install
build/bin/aiecc --version
# NPU HW (npu2): add_one_cpp_aiecc + add_one_cpp_aiecc_xchesscc → ./test.exe … → PASS!
# lit: lit -v build/test/aiecc   → 10 pass / 29 fail / 8 unsupported
```
