# aiecc declarative reimplementation — feature-parity gaps

Tracking parity between the declarative driver (`aiecc.cpp`, ~1,000 lines) and the
previous monolithic driver on `main` (~5,700 lines).

> The `README.md` parity table describes the **old** monolithic C++ driver.
> It is **not** an accurate description of the declarative driver and
> should be reconciled once parity work lands.

Goal: **functional parity** with the base implementation, accepting justified
differences. In-driver **parallelization** of core compilation is explicitly
**deferred** as a future addition (`-j` / `--nthreads` are accepted and passed
through to sub-tools, but the driver itself stays single-threaded).

---

## P1 — Missing compilation artifacts (START NOW)

Each is a standalone final artifact the old driver could emit but the declarative
driver cannot yet. Several are *already built internally* and only need to be
exposed as graph outputs.

| Flag | Artifact | Status / Notes |
|------|----------|----------------|
| `--aie-generate-pdi` / `--pdi-name` | per-device `.pdi` | **Done.** PDI edge exposed as an output. |
| `--aie-generate-cdo` | per-device CDO `.bin` set | **Done.** CDO refactored into its own edge (directory-as-File); feeds the BIF/PDI and is exposable standalone. |
| `--aie-generate-txn` / `--txn-name` | per-device transaction `.mlir` | **Done.** New edge runs `convert-aie-to-transaction` on the ELF-patched module. |
| `--aie-generate-ctrlpkt` + `--ctrlpkt-name` / `--ctrlpkt-dma-seq-name` | ctrl-packet `.bin`, DMA-seq `.bin` | **Done.** Shared `ctrlpkt_lowered` edge branches into both binaries (pipeline runs once). |
| `--ctrlpkt-elf-name` (combined ctrl-packet ELF) | combined `.elf` | **Done** — `ctrlpktElf` edge bundles the DMA-seq + ctrl-packet binaries with an `external_buffers.json` patch and assembles them in-memory via aiebu (`blob_instr_transaction`). In-memory only: hard-fails if aiebu is not linked. Emitted with `--aie-generate-ctrlpkt`. Default `{0}_ctrlpkt.elf`. |
| `--aie-generate-elf` / `--elf-name` | instruction ELF (via aiebu) | **Done** — `instElf` edge: split NPU module per device → `AIETranslateNpuToBinary` → in-memory aiebu (`blob_instr_transaction`). In-memory only: hard-fails if aiebu is not linked (no subprocess fallback). Default `design.elf`. |
| `--generate-full-elf` / `--full-elf-name` | combined PDIs + NPU insts ELF | **Done** — `fullElf` edge builds the `aie2_config` JSON then assembles it via a declarative `ShellCommand` edge (`aiebu-asm -t aie2_config`). TODO: move in-memory once the library's `aie2_config` path is fixed (it is a no-op in this XRT build). Default `aie.elf`. |


## P2 — In-memory tool calls (functional parity)

The old driver had in-process library paths; the declarative driver shells out.
Both must be restored. A simple callback-function edge is preferred over a new
subprocess edge where the library is linkable.

- [x] **aiebu in-memory** — **Done.** Links `AIEBU::aiebu_static` (CMake does
  when `aiebu_FOUND`); a shared `assembleElf` helper calls
  `aiebu_assembler_get_elf` in-memory (`blob_instr_transaction`) for the
  instruction ELF and combined control-packet ELF. Guarded by
  `AIECC_HAS_AIEBU_LIBRARY`; in-memory only with no subprocess fallback, so it
  hard-fails if the library is unavailable. The full ELF (`aie2_config`) still
  shells out via a declarative `ShellCommand` edge because the library entry
  point is a no-op in some XRT packages — see the TODO at the `fullElf` edge.
- [x] **bootgen in-memory** — **Done.** Links `bootgen-lib` (CMake configures
  `tools/bootgen` before `tools/aiecc` so the `if(TARGET bootgen-lib)` check
  succeeds). The `pdi` edge calls the bootgen C API `bootgen_generate_pdi`
  (`bootgen_c_api.h`, `BOOTGEN_ARCH_VERSAL`) in-process via the `assemblePdi`
  helper instead of forking the `bootgen` CLI. Guarded by
  `AIECC_HAS_BOOTGEN_LIBRARY`; builds without the library fall back to a
  declarative `bootgen` ShellCommand edge (no ad-hoc subprocess/temp-file
  machinery). Verified byte-identical to the CLI output.

## P3 — Core-compile backend swappability (functional parity)

- [x] **Chess backend (`--xchesscc` / `--xbridge`)** — **Done.** `buildObject`
  branches on `--xchesscc`: downgrade IR for Chess → `chess-llvm-link` against
  `chess_intrinsic_wrapper.ll` → `xchesscc_wrapper -c`. The link step branches on
  `--xbridge`: per-core BCF (`AIETranslateToBCF`) → `xchesscc_wrapper +l`. The two
  flags are coupled (selecting either enables both), mirroring the legacy driver;
  Peano objects can't be xbridge-linked and vice versa. `--aietools` auto-
  discovered from `$AIETOOLS_ROOT` / `xchesscc` on PATH. Same path serves unified
  and per-core. `link_with` external objects are forwarded on the xchesscc link
  command line: a `linkWithObjs` edge parses the BCF's `_include _file` entries
  and resolves them (BCF includes keep their relative spelling — the chess parser
  rejects bare absolute paths; the resolved objects are passed via ShellCommand's
  variadic `inputs()` part).
- [x] **AIE1 support** — **Done + validated.** AIE1 (Versal AI Engine) is
  **not** an NPU target: it emits *no* `npu-insts`/`cdo`/`pdi`/`xclbin` (the
  in-tree `translateToCDODirect` asserts `IsNPU`, so `--aie-generate-cdo` & co.
  intentionally don't apply to AIE1). Its deliverable is the set of per-core
  ELFs plus a generated `aie_inc.cpp` host-control file, linked into a host
  executable that drives the array through `libxaiengine` — i.e. the
  **Versal/host-compile path** (see *Host compilation*), not new core-compile
  logic. What makes it work:
  - *Core compile (already in place):* `getChessTarget` maps `aie`/`aie1`
    → `target`; `getInputWithAddressesPipeline` already skips
    `convert-vector-to-aievec` for AIE1; `getCoreLLVMLoweringPipeline` forwards
    the target to `ConvertAIEVecToLLVM`. AIE1 requires Chess (`--xchesscc`/
    `--xbridge`); Peano's AIE1 backend is not exercised. **Build-config
    prerequisite (fixed):** `chess-llvm-link` *is* present for AIE1
    (`tps/lnx64/target/bin/LNa64bin/`), but the build must include AIE1 in
    `AIE_VITIS_COMPONENTS` so `aie_runtime_lib/AIE/chess_intrinsic_wrapper.ll`
    is generated/installed — `utils/build-mlir-aie-from-wheels.sh` now passes
    `-DAIE_VITIS_COMPONENTS=AIE;AIE2;AIE2P` (was `AIE2;AIE2P`). Without the AIE1
    wrapper, chess-llvm-link fails on that missing input (reported confusingly
    as `chess-llvm-link: No such file or directory`).
  - *`aie_inc.cpp` edge:* `AIETranslateToXAIEV2` on the ELF-patched physical
    module → host-side array-configuration source (shared with host compilation
    and aiesim).
  - *Final-artifact selection:* AIE1 routes through the host-compile path (core
    ELFs + `aie_inc.cpp` + host exe), not the NPU artifact graph.
  - *Runtime libs:* the host edge wires `runtime_lib/<arch>/xaiengine`
    (include+lib) and the `test_lib` memory allocator.
  - **Validated end-to-end** (`test/aiecc/simple.mlir` = `xcvc1902`, with
    `test.cpp`): `aiecc --xchesscc --compile-host …` → chess core ELF (AIE1 arch
    `0x108`) + `aie_inc.cpp` (3534 B) + x86-64 host executable, EXIT=0.
    Confirmed `--aie-generate-cdo` on AIE1 asserts `IsNPU` by design (CDO/PDI/
    xclbin are NPU-only).

## P4 — Selection / host-toolchain parity (smaller)

- [x] `--sequence-name` — **Done.** The `perSeq` filter narrows the per-runtime-
  sequence split to the named sequence (in addition to the device filter), so
  only the matching `insts_<device>_<seq>.bin` is emitted. Empty value keeps all
  sequences.
- [x] `--xclbin-input` — **Done.** Extends an existing xclbin with the freshly
  built kernel/PDI. The `buildXclbin` lambda dumps the input xclbin's
  `AIE_PARTITION:JSON` (`xclbinutil --dump-section`), merges the new device's
  first PDI into the input partition's `aie_partition.PDIs` (in-memory
  `llvm::json`), then `xclbinutil --input <in> --add-kernel … --add-replace-
  section AIE_PARTITION:JSON:<merged> --output`. Validated: 1-PDI base → 2-PDI
  extended xclbin.
- [x] `--no-materialize` — **Done.** Threads a `materialize` flag into
  `getNpuLoweringPipeline`, which gates
  `createAIEMaterializeRuntimeSequencesPass`.
- [x] **Host compilation** — **Done.** Compiles the user's positional C/C++
  source files into a host executable that drives the array via libxaiengine.
  The positional arg list now separates the `.mlir` input from host sources
  (`isHostSourceFile`/`getInputFilename`/`getHostSourceFiles`, mirroring the
  legacy driver). An `aieInc` edge runs `AIETranslateToXAIEV2` per device to
  produce `aie_inc.cpp` (lazily materialized into the workdir); a `hostExe`
  join edge (`bundle(aieInc, perDevice)`) builds the `clang++` link command
  dynamically and runs it through `ShellCommand`: `-std=c++17`, `--target`,
  `--sysroot` (+`--gcc-toolchain` for `aarch64-linux-gnu`), the `test_lib`
  `libmemory_allocator_{ion,hsa}.a`, the `xaiengine` include/lib (+`-Wl,-R`),
  `-I<workdir>` (so the host's `#include "aie_inc.cpp"` resolves),
  `-fuse-ld=lld -lm -lxaienginecdo`, the `__AIEARCH__` define keyed off the
  device generation, then user `-I`/`-L`/`-l` + sink passthrough flags + host
  sources + `-o`. Selected by `--compile-host` (host-exe filename via `-o`,
  default `a.out`, matching the legacy driver; generated artifacts land in the
  `--output-dir` directory). HSA mode (`--link_against_hsa`) adds `-DHSA_RUNTIME`
  and the `-hsa` runtime-lib arch suffix. Validated end-to-end: `aie_inc.cpp`
  generated, host source compiled, and `clang++` (peano) linked an x86_64
  executable.
- [x] `--compile` / `--no-compile`, `--link` / `--no-link` — **Done.** Accepted
  and combined into `doBuildElfs = (compile && !noCompile) && (link && !noLink)`;
  when false the graph substitutes `preBakedElfs` so core compile/link edges are
  pruned. Verified `--no-link` produces zero core compile/link execs.
- [x] `--no-compile-host`, `--no-unified` — **Done.** Negations folded in as
  `doCompileHost = compileHost && !noCompileHost` and
  `doUnified = unified && !noUnified`, driving host-exe output selection and the
  unified-vs-per-core object choice.
- [x] `--opt-level` (`-O0..3`) — **Done.** Hard rejection removed. Matches the
  legacy driver's quirky semantics exactly: `opt` is capped at `O1`
  (`--passes=default<O1>`) with `-disable-loop-idiom-memset` added at `O>=3`,
  while `llc` and the peano link step take the raw `-O<level>`. Confirmed via
  verbose output for `-O0..3`.

## P5 — Low-priority, but needs to get done eventually

- [x] `-j` / `--nthreads` — **Accepted (pass-through).** The flag parses and is
  forwarded to sub-tools that support it; the driver itself does **not** yet
  parallelize core compilation (still single-threaded). True in-driver
  parallelization remains a future addition.
- [x] Dry-run (`-n`) — **Done (best-effort).** `ShellCommand::dryRun` prints
  `Dry run - command not executed` and touches an empty output file instead of
  executing. Known limitation: in-memory edges that read a tool's output may
  fail under `-n` (e.g. `-n` combined with `--xclbin-input`), since the skipped
  command produces no real file to parse.
- [x] Reproducer / repeater scripts (`--enable/disable-repeater-scripts`,
  `--repeater-output-dir`) — **Accepted (no-op).** Parsed for CLI compatibility;
  no reproducer scripts are emitted yet.
- [x] Back-compat no-op flags (`--profile`, `--progress`, `--aie-version`) —
  **Done.** `--aie-version` prints the version and exits; `--profile` /
  `--progress` are accepted no-ops.
- [ ] **aiesim (`--aiesim`) — refactor into declarative edges.** *Functionally
  wired but non-declarative.* `--aiesim` (requires `--xbridge`) is at parity:
  the `aiesimWork` sink off `perDevice` runs `aiecc_aiesim.cpp`
  (`generateAieIncCpp` + `generateAiesim`) as an opaque blackbox that shells out
  to `aie-translate`/`aie-opt`/`clang++` and writes the `Work/` tree + `ps.so` +
  `aiesim.sh` via direct disk I/O, bypassing the `Item`/graph abstraction
  (`aiesimWork.producesFiles = false`). It works but violates the driver's
  declarative design — it should be refactored into proper graph edges (one
  `Item` per generated artifact), like the ELF/PDI flows were.

## P6 — Test suite (revisit later)

- [x] **CLI flag names realigned to the legacy driver.** The declarative
  driver's options were renamed back to the legacy spelling so `test/aiecc/**`
  invocations parse unchanged: `--workdir`→`--tmpdir`,
  `--ctrl-pkt-overlay`→`--generate-ctrl-pkt-overlay`,
  `--keep-intermediates`→`--dump-intermediates`, `-v`→`--verbose` (`-v` kept as
  an alias), `--opt-level`→`-O` (`--opt-level` kept as an alias), and the
  host-exe `-o` / artifact `--output-dir` split (see P4 *Host compilation*).
  The Peano-default toolchain selection also accepts the legacy negations
  `--no-xchesscc` / `--no-xbridge` (the former implies the latter). All option
  definitions now live in `tools/aiecc/CommandLineOptions.h`.
- [ ] Several `test/aiecc/**` tests still fail because they check the **legacy
  driver's verbose stdout strings** (e.g. `Compiling N core(s) using unified
  compilation`, `Successfully parsed input file`, `Compilation completed
  successfully`) that the declarative driver does not emit — or expect non-empty
  stdout where the driver is now silent (e.g. a successful `-n` dry-run yields
  empty output, so `FileCheck` reports an empty input). This is an
  *output-message* parity gap, separate from CLI parity; triage per-test.
  Current suite tally: **10 pass / 29 fail / 8 unsupported** — the 29 failures
  are all this output-message gap, not behavioral regressions (the CLI flags
  themselves are validated working). The flags `-n`, `-j`, `--no-compile`,
  `--no-link`, `--sequence-name` are now accepted (see P4/P5); their tests fail
  only on the verbose-string mismatch above.
