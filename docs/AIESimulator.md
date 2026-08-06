# An open AIE array simulator, to replace the Vitis dependency of `--get-aiesim`

Status: RFC / design. Nothing here is merged yet.
Scope: AIE2 (npu1) and AIE2P (npu2). AIE1 is explicitly out of scope.

## 1. Why

`aiecc --get-aiesim` is the only hardware-free way to run an mlir-aie design end to end, and it is the
last hard anchor holding Chess and Vitis in this project. Everything else has an open path:
`--no-xchesscc --no-xbridge` compiles cores with Peano today.

The simulator does not. Three separate things make it Vitis-only:

* The `sim/` work folder is consumed by `aiesimulator`, an external Vitis binary
  (`tools/aiecc/aiecc.cpp:471-482` writes `aiesim.sh`, which runs `aiesimulator --pkg-dir=...`).
* `sim/ps/ps.so` is compiled against `adf/wrapper/wrapper.h`, `xtlm.h`, `libsystemc` and `libxtlm`
  from `$AIETOOLS_ROOT` (`tools/aiecc/aiecc.cpp:380-460`).
* `--get-aiesim` forces `--xbridge` and refuses `--no-xchesscc`
  (`tools/aiecc/CommandLineOptions.h:496`, "the AIE simulator consumes Chess-compiled cores"), so
  even the core ELFs must come from Chess.

Simulator tests are gated `REQUIRES: aiesimulator` and usually `valid_xchess_license` too. They are the
largest remaining block of Chess-gated files in the tree, and they are exactly the tests that today run
on ordinary CI machines with no NPU attached. Moving them to device verification instead would push load
onto the scarce Ryzen AI runners, which is the opposite of what is wanted.

There are in fact **two** closed simulators in use here, and they want different answers:

* **`aiesimulator`**, the full-array SystemC co-simulation, driven by `aie.mlir.prj/aiesim.sh`. 24
  functional tests plus 2 that only check aiecc's own artifact plumbing
  (`test/aiecc/cpp_aiesim.mlir`, `test/aiecc/checkpoint_resume_aiesim.mlir`).
* **`xca_udm_dbg`**, a standalone single-core Chess instruction-set simulator with no chip model at all,
  driven by `test/unit_tests/aievec_tests/profiling.tcl`. 32 tests, all under
  `test/unit_tests/aievec_tests/aie2/`. These already compile the kernel under test with Peano; only the
  `testbench.cc` around it goes through `xchesscc`. They need instruction semantics and nothing else:
  no DMA, no locks, no stream switch.

That split matters, because it means the two halves of this proposal are independently useful. An
instruction simulator with no array model already addresses the larger group of tests.

The proposal is to replace both with in-tree, open, deterministic components that run Peano-compiled
code, need no license, and produce the same observable the existing tests already check (the host
program's stdout).

## 2. The seam

The interesting finding is how small the proprietary surface actually is.

mlir-aie already builds the host-side `libxaienginecdo` with `__AIESIM__` defined
(`runtime_lib/xaiengine/lib/CMakeLists.txt:57`), and that is the library a simulated host program links
against (`tools/aiecc/aiecc.cpp:374-375` passes `-L<runtime_lib>/<arch>/xaiengine/lib -lxaienginecdo`
when it builds `ps.so`). Note this is a different target from `xaienginecdo_static`, which the compiler
itself links for CDO generation and which is built `-D__AIECDO__ -D__AIEDEBUG__` without `__AIESIM__`
(`runtime_lib/CMakeLists.txt:155-160`); in that build `xaie_sim.c` compiles to stubs that refuse at run
time, so only the host-side library carries the undefined symbols.

`__AIESIM__` selects aie-rt's `XAIE_IO_BACKEND_SIM`
(`third_party/aie-rt/driver/src/io_backend/xaie_io.c:34-35`), whose implementation
(`third_party/aie-rt/driver/src/io_backend/ext/xaie_sim.c:70-598`) forwards every register access to
seven externally-declared C functions:

```c
void     ess_Write32(uint64_t Addr, uint32_t Data);          /* xaie_sim.c:45  */
uint32_t ess_Read32(uint64_t Addr);                          /* xaie_sim.c:46  */
void     ess_WriteCmd(uint8_t Command, uint8_t Col, uint8_t Row,
                      uint32_t CmdWd0, uint32_t CmdWd1, uint8_t *CmdStr);
void     ess_NpiWrite32(uint64_t Addr, uint32_t Data);       /* xaie_sim.c:49  */
uint32_t ess_NpiRead32(uint64_t Addr);                       /* xaie_sim.c:50  */
void     ess_WriteGM(uint64_t addr, const void *data, uint64_t size);
void     ess_ReadGM(uint64_t addr, void *data, uint64_t size);
```

`ess_WriteGM` / `ess_ReadGM` are the DDR window, used by the test-side allocator
(`runtime_lib/test_lib/memory_allocator.cpp:15-16,46,50`). `ess_Write128` / `ess_Read128` exist in the
Vitis wrapper (`aie_runtime_lib/AIE2P/aiesim/genwrapper_for_ps.cpp:277-298`) but nothing in aie-rt calls
them.

Those symbols carry all register and memory traffic. Everything above them is open code we already
ship: aie-rt's tile, DMA, lock, stream-switch and core modules, and the `aie_inc.cpp` that `aiecc`
generates. Core ELFs are not handed to the simulator through a side channel either.
`_XAie_LoadProgMemSection` (`third_party/aie-rt/driver/src/core/xaie_elfloader.c:221-260`) writes
program sections into program memory with ordinary block writes, so the ELF arrives as MMIO traffic
exactly as on hardware. The two `XAie_CmdWrite` cases (`SETSTACK`, `LOADSYM`,
`xaie_elfloader.c:44-45`) are debug conveniences.

**So an open simulator is a library that defines those seven symbols over a software model of the
array.** No aie-rt fork, no new IO backend, and no NEW vendored patch. That matters: aie-rt is a
third-party submodule shared with XRT and iree-amd-aie, and forking it would be a permanent tax.

One qualification, because it is easy to state this too strongly. Those `ess_*` declarations are not
pristine upstream aie-rt. Upstream `xaie_sim.c` has `#include "main_rts.h"`, a Vitis-only header;
mlir-aie's `third_party/patches/aie-rt/0001-cdo-sim-defork-fixes.patch` replaces that include with local
forward declarations of the `ess_*` functions, and `aiert.cmake` applies it at configure time. So the
de-Vitis-ing of the SIM backend has already been done, by mlir-aie, for exactly this reason. This
proposal inherits it rather than adding to it, and the line numbers cited above are post-patch. Anyone
reading pristine upstream aie-rt will not find those declarations.

### 2.1 The eighth dependency, which is a file format rather than a symbol

There is one more thing in the way, and it is not an `ess_*` symbol, so it is easy to miss. Under
`__AIESIM__`, `XAie_LoadElf` (`third_party/aie-rt/driver/src/core/xaie_elfloader.c:682-728`) runs a
block *before* it loads anything: it opens `<elf>.map`, and `XAieSim_GetStackRange`
(`xaie_elfloader.c:506-541`) scans that file for a line matching `"items) : Stack"` and parses it with
`sscanf(buffer, "    0x%8x..0x%8x (%*s")`. That is the Chess linker's map format. If the file is
missing or does not match, the function returns `XAIE_ERR` and `XAie_LoadElf` **returns without ever
calling `XAie_LoadElfPartial`**, so the ELF is silently never loaded and the generated host code's
`assert(RC == XAIE_OK)` fires.

Nothing in the aiecc build graph emits such a file for either backend, so this affects any host program
linked against the host-side `libxaienginecdo`, not just this proposal. It has simply never been
exercised, because `--get-aiesim` has always forced Chess.

The resolution keeps the no-fork property: generated host code targeting this simulator should call
`XAie_LoadElfPartial(dev, loc, elf, XAIE_LOAD_ELF_ALL)`, which is exactly what `XAie_LoadElf` does
after that block, and which is already public API (`xaie_elfloader.h:112`). The stack-range command it
skips is a Vitis profiling convenience that this model does not consume. That is a one-line change in
`lib/Targets/AIETargetXAIEV2.cpp:443`, in mlir-aie, under the same flag that selects the simulator.

This also settles a question about the current tool: `--get-aiesim` forcing `--xbridge` is not purely a
policy in aiecc's option resolver. There is a second, independent blocker in aie-rt's own loader that a
Peano ELF would hit, and no Peano link in this tree produces a Chess-format map file.

## 3. What the replacement looks like from the outside

The Vitis flow builds a package directory and then runs a separate simulator process against it:

```
aiecc --xchesscc --xbridge --get-aiesim design.mlir -- host.cpp
./design.mlir.prj/aiesim.sh | FileCheck design.mlir
```

The proposed flow drops the package directory entirely and links a normal native executable:

```
aiecc --no-xchesscc --no-xbridge --get-sim design.mlir -- host.cpp
./design.mlir.prj/design.sim | FileCheck design.mlir
```

`design.sim` is the host program (`host.cpp`, unmodified, keeping its own `main`) linked against
`libxaienginecdo` and `libaie-sim`. Running it *is* the simulation. There is no SystemC kernel, no
second process, no `graph.xpe` / `aieshim_solution.aiesol` / `scsim_config.json` / `flows_physical.json`
and no `.target` marker: those four descriptor files exist only because the Vitis simulator builds its
model from compiler-emitted metadata. This model builds itself from the register writes the design
actually performs, which is both less machinery and a stricter test.

Consequences worth stating plainly:

* Runs under `gdb`, `valgrind`, ASan and `perf` like any other binary.
* Single-threaded and deterministic. Same input, same cycle counts, every run.
* Runs Peano-compiled ELFs. The current simulator cannot: `--get-aiesim` forces `--xbridge`.
* No license check, so it can run on every CI job, not just licensed ones.
* Because the front end is register writes, the same model can replay an NPU transaction binary
  (`aiex.npu.*` lowered to TXN) without any new machinery. The Vitis simulator has no path for that.

What it will not do, at least initially: reproduce Vitis cycle counts exactly, model NoC contention or
DDR timing with any fidelity, or support AIE1.

## 4. Architecture

Two components, in the two repositories that own the two halves of the problem.

### 4.1 Fabric model, in mlir-aie (`runtime_lib/aie-sim`)

Owns everything outside the core datapath, and holds the `ess_*` entry points:

* **Address decode.** `ess_Write32` receives a flat address; column, row and register offset come back
  out by inverting aie-rt's own shifts. `_XAie_GetTileAddr`
  (`third_party/aie-rt/driver/src/common/xaie_helper.h:151-155`) builds
  `TileAddr = (Row << RowShift) | (Col << ColShift)` and every register write adds the intra-tile
  offset to it, so the layout from LSB up is `[regOff | row | col]` and inverting it is shift-and-mask.

  aie-rt's `_XAie_GetRowfromRegOff` / `_XAie_GetColfromRegOff` (`xaie_helper.c:918-931`) look like that
  inverse and are not. Masking the low `RowShift` bits cannot recover `Row`, which lives at bit
  `RowShift` and above. Checked by construction: encode `(row=3, col=5)` with `RowShift=20`,
  `ColShift=25` the way `_XAie_GetTileAddr` does, run both helpers on the result, and they return
  `row=0`, `col=3` -- the "column" function returns the row and the "row" function returns low bits of
  the register offset. They fill the Col/Row fields of a debug TXN command header
  (`_XAie_AppendWrite32` and siblings), which is a different job. Register offsets and field layouts are
  already in the tree as machine-readable tables: `third_party/aie-rt/driver/src/global/xaie2pgbl_params.h`
  is 33k lines of MIT-licensed AIE2P register definitions, with an AIE2 equivalent. The model is written
  against those headers rather than re-deriving the map.
* **Memories.** Core data and program memory, memtile memory, and the DDR window behind
  `ess_WriteGM`/`ess_ReadGM`.
* **Locks.** AIE2/AIE2P semaphores, with the acquire-with-value / release-with-value semantics used by
  both the core lock ports and the DMA task queues.
* **DMA.** Buffer descriptors, channel control, the n-dimensional address generator, BD chaining, and
  the shim and memtile variants.
* **Stream switch.** Circuit-switched connections and packet-switched rules, built from the switch
  configuration registers, so routing is whatever the design actually programmed.
* **Registers, and the fault contract.** Every offset is claimed by a component, on an explicit
  reserved-reads-zero allow-list, or modelled by nothing. Reading an offset that is unclaimed AND was
  never written is a hard named failure, because a design polling a status register nobody implemented
  would otherwise spin on a fabricated zero with no diagnostic. Reading back a value the host itself
  wrote is a pass-through rather than an invention, so it is allowed. Writes to unmodelled registers
  are recorded rather than fatal (the model will never claim every register, and refusing to run until
  it does would make it useless); `AIE_SIM_STRICT=1` promotes them, and the recorded set is emitted on
  stderr at teardown so "unmodelled" is a number rather than a surprise. Nothing is printed when the
  set is empty, which as of 2026-08-02 is the case for all three of the array tests that run end to
  end: every register they write is modelled.

  That runtime report says what one design touched. The static counterpart says what the model
  covers at all: `register_coverage <params-header> [device]` takes the register universe from
  aie-rt's own generated database (a define is an address when a sibling `_WIDTH` exists, which
  separates the ~2.1k addresses from the ~31k field defines) and asks `RegisterFile::isClaimed()`
  once per address, so the answer comes from the same lookup a running design hits rather than from
  a hand-kept list that drifts. As of 2026-08-02, AIE2P/npu2 is **1401 of 2141 addresses, 65.4%** --
  memtile 74.3%, core 63.6%, shim 54.9%; AIE2/xcve2802 is 63.0% of a slightly larger map. Unclaimed
  is not automatically a gap: trace, debug, performance counters and ECC are deliberately absent, so
  this is a trend line and a review aid, not a number to maximise.
* **Time.** A cycle counter, and only components with outstanding work are stepped. Order within a
  cycle is registration order, used as an explicit sort key rather than left to container iteration,
  which is what makes runs reproducible. The array advances inside the `ess_*` entry points, so a host
  polling loop (which is how aie-rt implements every wait) makes progress with no threads.

  Host READS advance the clock and writes do not. That asymmetry is deliberate: aie-rt's masked
  register update is a `Read32` followed by a `Write32`, and if the write also advanced, the array
  could evolve between the sampled value and the applied one and clobber a hardware-driven bit in a gap
  silicon does not have.

The fabric deliberately does not know how to execute an instruction.

### 4.2 Core engine, in llvm-aie

Executing AIE2/AIE2P instructions belongs where the ISA is defined. llvm-aie already has the whole
decode side: full disassemblers for AIE2, AIE2P and AIE2PS, generated decode tables, instruction
printers and encoders. What is missing is semantics, and semantics are a property of the backend.

There is a second, independent reason to put it there. llvm-aie today has no execution tests at all:
`llvm/test/CodeGen/AIE` compiles and FileChecks, and nothing runs. An in-tree instruction interpreter
gives the backend end-to-end execution testing for the first time, which is worth having on its own
even if mlir-aie never called it.

That side has its own RFC (`llvm/docs/AIEInstructionSimulator.md` in llvm-aie). Two things from it
matter here, because they set expectations for this proposal: bundle decode is already complete and
tested, and instruction semantics are order-1000 instructions of hand-written work that cannot be
generated, since declarative TableGen patterns cover only the scalar core. That is the long pole of the
whole effort, and it is why the array model is designed to be useful before any core executes.

### 4.3 The boundary between them

A Peano distribution ships `lib/libLLVM.so` but no LLVM headers, and mlir-aie and Peano are separately
versioned and separately built. So the fabric must not be compiled against LLVM. The boundary is a small
versioned C ABI (`runtime_lib/aie-sim/include/aiesim/aie_iss_c_abi.h`), resolved with `dlopen`:

* The engine exports one symbol, `aie_iss_get_api`, returning a vtable.
* Every memory access, lock operation, stream access and character the core produces is a callback into
  the fabric. The engine holds architectural registers and nothing else.
* `step()` returns retired / stalled / done / fault. Stalls are how blocking lock and stream ports are
  modelled: the instruction is simply not retired and is re-issued later.

The fault case is load-bearing. An instruction the engine does not model must produce a hard,
named simulation failure. A simulator that silently skips what it does not understand is worse than no
simulator, because it turns a missing feature into a wrong answer.

Engine discovery is `$AIE_SIM_CORE_ENGINE`, then `$PEANO_INSTALL_DIR/lib/libaie-iss.so`. A missing
engine is reported the first time a core is enabled, not at construction, so designs with no core code
still simulate.

### 4.4 The core-debug register window

A host does not ask the core for its registers over a side channel: it reads them through ordinary MMIO,
so the fabric must project the engine's architectural state onto tile-local offsets. This is the bridge
`installCore()` currently stubs, holding five registers at zero until an engine exists.

The window is a real, enumerable hardware structure rather than a handful of special cases. Counting
`CORE_MODULE_CORE_*` entries in aie-rt's database for AIE2P gives **233 registers over 0x30000-0x32038**,
and the layout is regular:

* **One 32-bit value per 16-byte slot** -- every family, without exception, has stride `0x10`.
* **Wider registers occupy consecutive slots** and say so in their names: `WL0_PART0/PART1`,
  `BMLL0_PART0..PART3`. Their `_WIDTH` is the *architectural* width (128 for a `WL` part), not the 32
  bits an `XAie_Read32` returns, so a reader assembles a wide register from its parts.
* The families are the register file as the ISA defines it: `R0-R31`, `P0-P7`, `M0-M7`, `DN/DJ/DC0-7`,
  `S0-S3`, `Q0-Q3`, `E0-E11`, the `WL/WH0-11` vector halves, the `BM*` accumulator partials, and a
  control block `PC FC SP LR LS LE LC CR0-1 SR` at `0x30E00-0x30E90`.

**The bridge is a table, not a mechanism.** `CoreEngine::readRegister(name, data, size)` already exists
and is already documented as serving "the core-debug register window that aie-rt exposes". So the design
is: map offset to `(register-name, part-index)`, and route the existing `onRead` through it, falling back
to the present reads-zero behaviour when no engine is loaded. Nothing new crosses the engine boundary,
and the ABI question in 4.3 is untouched by this.

**Offsets are per generation and the generations genuinely collide**, which is why the table must be
generated per `Generation` and never shared:

| offset    | AIE2      | AIE2P     |
|-----------|-----------|-----------|
| `0x30C00` | `CORE_R0` | `CORE_Q0` |
| `0x30E00` | `CORE_M0` | `CORE_PC` |
| `0x31000` | `CORE_P0` | `CORE_R0` |
| `0x31100` | `CORE_PC` | `CORE_R16`|

Every one of those four offsets is a valid register on both generations and a *different* register on
each, so a wrong-generation table does not fault -- it returns a plausible number for the wrong register.
That is the failure mode the fault contract cannot catch, and it is the argument for generating the table
from the vendored database rather than hand-transcribing five entries per generation.

**Status.** The window is claimed: 230 registers on AIE2P, 210 on AIE2, expressed as the six (resp. five)
contiguous ranges the families merge into, so the gaps between them stay unmapped and still fault. With no
engine loaded every one reads its `*_REGISTER_VALUE_DEFVAL` of 0, which is also what a core that has never
run reads on hardware. `core_window_test` checks both generations against independently written ranges,
asserts the gaps are *not* claimed, and asserts `CORE_CONTROL` still reports its reset value rather than
being swallowed by the window.

What remains is the engine dispatch: routing those reads to `readRegister` when a core engine is attached,
which needs the offset-to-name mapping and an engine to ask. Scalar families (`R`/`P`/`M`/`S` and the
control block) are one slot each and map directly; the vector and accumulator families need the
part-assembly rule above and an engine that models them at all, so they belong with the vector phase.
Writes go through `writeRegister` on the same table and can stay faulted until something needs them.

## 5. Why register-level rather than graph-level

The Vitis simulator is configured from four descriptor files that `aiecc` generates from the MLIR
(`tools/aiecc/aiecc.cpp:259-306`). That design has a blind spot: it simulates the design the compiler
*described*, not the design the configuration code *programmed*. Configuration bugs, which are a real
and recurring class in this project, are invisible to it.

Driving the model from register writes removes both the descriptor files and the blind spot, and it
generalises. Anything that reduces to register writes can be fed to the same model: the `aie_inc.cpp`
path, a CDO blob, or an XRT transaction binary. The last two need a decoder in front of the model, not
a different model, and the transaction format is small and already decoded in open code. That matters
because the transaction binary is what actually runs on npu1 and npu2 hardware, and there is currently
no hardware-free way to execute it at all.

## 5a. Compiling the aievec kernels for AIE2P

`test/unit_tests/aievec_tests/aie2` holds 25 vectorised kernels. Their RUN lines call lit
substitutions (`%vector-to-llvmir%`, `%llvmir-to-ll%`) that `test/lit.cfg.py` no longer defines, every
one is `XFAIL: *`, and the intended path ends in `xca_udm_dbg` -- the Vitis tool this component
replaces. The pipeline they describe still works and is worth writing down, because it is how kernels
get in front of the simulator:

```sh
aie-opt "$K" -affine-super-vectorize="virtual-vector-size=$VS" \
  -convert-vector-to-aievec="aie-target=aie2" \
  -lower-vector-to-aievec="aie-target=aie2" \
  -convert-aievec-to-llvm="aie-target=aie2p" \
  -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-arith-to-llvm \
  -finalize-memref-to-llvm -convert-index-to-llvm -convert-cf-to-llvm \
  -convert-func-to-llvm -reconcile-unrealized-casts -o kernel.mlir
aie-translate kernel.mlir --mlir-to-llvmir -o kernel.ll
llc -mtriple=aie2p -O2 -filetype=obj kernel.ll -o kernel.o
ld.lld -e dut --section-start=.text=0x1000 --section-start=.bss=0x30000 -o kernel.elf kernel.o
llvm-aie-run kernel.elf --entry=dut --scratch=0x0:0x40000 --coverage
```

`$VS` differs per kernel and is in that kernel's own RUN line; take it from there rather than
assuming 32.

**The target argument differs between the two halves, and getting it wrong silently costs
everything.** `convert-vector-to-aievec` and `lower-vector-to-aievec` accept only `aie` or `aie2` --
the aievec dialect is shared. `convert-aievec-to-llvm` accepts `aie2p` as well, and it is what
chooses the intrinsic namespace. Passing `aie2` there emits `llvm.aie2.*`, which `llc -mtriple=aie2p`
cannot select for any of the 25; passing `aie2p` emits `llvm.aie2p.*` and 16 of them compile.

Measured 2026-08-03: pipeline succeeds for 25/25, `llc -mtriple=aie2p` succeeds for **16**. The nine
that do not:

* **six** `*_mul_elem*` kernels stop in `aie-opt` with `aievec.mul_elem conversion is not supported
  for AIE2p` -- an explicit gap in `lib/Conversion/AIEVecToLLVM`, not a pipeline mistake;
* **`bf16_exp_lut`** stops in `llc` with `unable to translate instruction: call` (a libcall);
* **`bf16_max_reduce`** and **`bf16_min_reduce`** stop in `llc` on their reduce lowering.

## 6. What the existing tests actually need

Counted on `upstream/main`, 58 files total. The point of the breakdown is that the headline number
overstates the work: a third of these tests are already expected to fail, and most of the rest need
scalar semantics only.

| group | files | of which XFAIL | device family | what a replacement must model |
| --- | --- | --- | --- | --- |
| `aievec_tests/aie2`, single-core, `xca_udm_dbg` | 32 | 27 (2 also permanently disabled) | AIE2 ISA, no device | vector ISA (25), scalar ISA (7). No array model at all |
| `chess_compiler_tests_aie2`, `aie2/29`, `aie2/30`, `aie2p/00_itsalive` | 14 | 3 | `xcve2802` (AIE2), one `npu2` | locks, tile and memtile and shim DMA incl. n-D, shared memory, cascade, multi-column |
| `chess_compiler_tests`, `aie/31_stream_core` | 10 | 3 | `xcvc1902` (AIE1) | same, on the AIE1 register map and ISA |
| `test/aiecc/*aiesim*` | 2 | 0 | n/a | aiecc artifact plumbing, not simulation |

So the regression bar that is actually green today is about 25 tests, not 58.

Scoping consequences, stated rather than buried:

* **AIE1 is out of scope.** That leaves 10 files, 7 of them currently green. AIE1 has a different
  register map and a different ISA, and it is being deprecated upstream. Those tests keep needing Vitis
  until someone decides to port or drop them, and this proposal does not pretend otherwise.
* **`xcve2802` is in scope even though it is not an NPU.** It is the AIE2 architecture, so it is one
  more entry in the device table, not a second model.
* **AIE2P has essentially no simulator coverage today.** `test/unit_tests/aie2p/00_itsalive/aie.mlir` is
  two `aie.flow` declarations and a host `printf`; there is no `aie.core`, no DMA and no lock in it. For
  AIE2P this work is building coverage that does not exist, not reproducing coverage that does.
* Of the 24 functional array tests, 22 have scalar cores or no core at all. Only 2 use vector and
  cascade intrinsics. Vector ISA support is what unlocks the `aievec_tests` group, not the array group.

## 7. Phasing

Each phase is meant to be reviewable and mergeable on its own, and each has a test that does not need
Vitis, a license, or an NPU.

The two components can proceed in parallel; neither blocks the other.

**In mlir-aie, the array model:**

1. **Fabric core.** Address decode, memories, register file, the `ess_*` ABI, the deterministic
   scheduler. Test: host writes and reads tile and memtile memory through aie-rt and gets its data back.
2. **Locks and DMA.** Lock semantics, BD and channel model, n-D address generation, shim and memtile
   DMA. Test: a core-free design that moves data DDR to memtile to DDR, driven by a real runtime
   sequence. This alone is enough for the tests that have no `aie.core` at all
   (`chess_compiler_tests_aie2/08_tile_locks`, `09_memtile_locks`).
3. **Stream switch.** Circuit and packet routing from the switch registers. Test: multi-column routed
   flows, including packet headers.
4. **aiecc and lit integration.** `--get-sim`, a `sim` lit feature, and conversion of the existing
   simulator tests so they run on Vitis where a license is present and here where it is not.

**In llvm-aie, the core engine:**

1. **Decode loop and scalar semantics**, behind the C ABI in 4.3. Test: llvm-aie's own execution
   tests, which do not exist today in any form.
2. **The `aievec_tests` harness.** Those 32 tests already build the kernel with Peano; replacing
   `xca_udm_dbg` with an open runner removes their last Chess dependency without touching the array
   model. 7 of them are scalar and land with phase 1.
3. **Vector and accumulator ISA**, with `IntrinsicsAIE2.td` and `IntrinsicsAIE2P.td` as the checklist
   and a coverage report so that partial support is visible.
4. **Timing.** ~~Replace the flat one-bundle-per-cycle cost with the per-instruction itineraries.~~
   **Superseded 2026-08-02, and the correction matters because this phase was sized in months.** AIE is
   an in-order VLIW *without interlocks* -- `AIEHazardRecognizer.h:145` says so -- so the compiler covers
   every operand latency itself by padding the schedule with NOPs, and one bundle issues per cycle
   unconditionally. **Bundle count already IS the core-datapath cycle count**; applying the itineraries
   at simulation time would charge a second time for latency the static schedule has already paid.
   Counted rather than assumed: 110 of 120 `InstrStage` defs are one cycle, so the itineraries carry
   operand latency and functional unit, not issue occupancy, and both are consumed before the binary
   exists. Independently confirmed on hardware by an earlier from-source cycle prediction that landed
   within 8 cycles (0.2%) of a measured 4052.

   What was left of this phase is **fabric stall attribution**: on a machine with no interlocks, waiting
   on a lock, a stream, a DMA or a memory-port conflict is the only thing that can make a cycle cost
   more than one bundle. That is an `interval` reading in the array model, not itineraries in the engine.
   **DONE 2026-08-02**, and it is the `interval` shape's first producer. `Array::stepOneCycle` already
   knows whether a component did work and whether it is still busy, so it owns the attribution and
   components supply only the reason; one that cannot answer reports `unknown`, and the
   `stalls-attributed` verdict fails on any such cycle rather than letting the breakdown look complete.
   Cycles a component was not scheduled are absent rather than zero-filled, and the occupancy
   denominator is scheduled cycles, so an array that finished early does not read as one that stalled.
   Measured free: 48-tile npu2 with every DMA lock-stalled, best of 9 runs of 2M cycles, 1.87 Mcycle/s
   with attribution off against 1.97 with it on.

   Known granularity limit: a tile's DMA channels share one `Steppable`, so a track is per DMA module
   and the reason is the first stalled channel's in a fixed scan order. The record's entity convention
   allows `tile:c,r/dma:s2mm0`; reaching it means per-channel `Steppable`s, which is a scheduling
   change rather than a reporting one. The core reports only that it was the waiter, because the C ABI
   returns one `Stalled` result without naming the port.

Array phases 1 to 3 are useful before any core executes, and core phases 1 to 2 are useful before any
array exists. That is the main reason for the split.

## 8a. Design decisions settled by research, so they are not re-litigated

Researched after the first implementation, which is the wrong order and cost two of the errors below.
Recorded here so the reasons travel with the code.

* **Event-driven activation on ONE logical clock.** Not cycle-driven stepping of everything, which is
  what the first version did. Measured on a 48-tile npu2, release build: idle advance went from
  0.12 Mcycle/s to O(1) in the number of tiles, and a busy two-tile DMA route went from 0.12 to
  1.86 Mcycle/s. The shape matters more than the ratio: busy throughput is now roughly independent of
  array size (2.20 / 2.10 / 1.86 Mcycle/s at 1, 4 and 8 columns) where before it degraded linearly
  (1.03 / 0.24 / 0.12).
* **`step()` then `commit()` stays.** It is SystemC's evaluate/update phase. It was re-derived here
  from a bug rather than read from a 30-year-old standard, but it is right.
* **Do NOT separate the functional model from the timing model.** That decoupling exists for running
  far ahead of hardware time, which is not this problem, and it would destroy the relative-cycle
  invariants that have already caught a serious bug.
* **The core engine is a call-threaded interpreter, not a JIT.** Being inside LLVM does not make a JIT
  cheap: QEMU's Hexagon target generates code from Qualcomm's own architecture description and still
  landed as a follow-on to a hand-written-helper target.
* **No home-grown ISA description language.** It relocates the authoring burden into a new grammar
  that also has to be debugged, and pays off only for a retargetable tool family, which this is not.

### 8a.1 Why some interfaces refuse rather than approximate

Four places in `Components.h` and `CoreEngine.h` return "not available" where a plausible value could
have been produced. Each is a decision, not a gap:

* **Packet-mode stream masters are absent from `StreamRoute`.** A packet master names an arbiter and
  an msel mask rather than one slave, so its source is decided by arbitration at run time; any edge
  drawn for it would be invented. The packet header encoding this model uses is itself ungrounded.
  `StreamSwitchModule::packetModeMasters()` reports the count so a consumer knows the routing graph is
  partial rather than assuming the design had no other routes.
* **`coreScalarRegister` maps only the one-slot scalar families** (r0-r31, m0-m7, p0-p7, s0-s3,
  sp/lr/ls/le/lc, PC). The vector and accumulator families (wl/wh, the bm/am partials) need the
  part-assembly rule in 4.4, and aie-rt's fc/cr/sr/dp have no one-to-one engine register at all --
  llvm-aie models those bits as separate named registers (`crSat`, `crRnd`, ...), so one offset would
  have to be assembled from several. Both are vector-phase work, and guessing either produces a
  plausible wrong number, which is the one failure the fault contract cannot catch.
* **`makeTileCorePort` faults on the three neighbour bands and the core stream/cascade ports** rather
  than stalling, so a design that reaches one fails loudly instead of hanging or reading a plausible
  wrong word.
* **`CoreEngine::opcodeCoverage()` defaults to empty.** An engine that does not track coverage is not
  broken, so callers must distinguish "not tracked" from "no gaps".

### 8a.2 Why data memory is claimed on the register bus

`installMemory` claims `[0, tile.memory()->size())` on the tile's `RegisterFile`, which looks like a
layering violation until you follow the host path. aie-rt's sim IO backend has exactly one pair of
entry points -- `ess_Read32` / `ess_Write32` in `xaie_sim.c` -- for every access regardless of which
higher-level API reached it. So `XAie_Read32` at a data-memory address and `XAie_DataMemRdWord` arrive
identically, and unless data memory is reachable through the register bus a host-side buffer access
(`mlir_aie_read_buffer_local`, say) faults as unclaimed.

### 8a.3 Where the lock semantics come from

aie-rt never calls a lock API on the simulator: it reads and writes plain registers, and the semantics
live in how those registers are wired. Two disjoint per-tile ranges matter, both taken from the
vendored MIT-licensed AIE2/AIE2P tables in `third_party/aie-rt` rather than re-derived.

**The acquire/release REQUEST range.** A read at an address that folds the signed request value into
address bits performs the operation as a side effect and returns success in bit 0. That is the
documented hardware mechanism, not a simulator shortcut: `xaiegbl_regdef.h:681-690`, on
`struct XAie_LockMod`, says "the lock is acquired by reading a register", and the sim IO backend's
blocking poll is a plain `Read32` loop with no writes (`io_backend/ext/xaie_sim.c:205-225`,
`XAie_SimIO_MaskPoll`). Address arithmetic and field encoding come from `locks/xaie_locks_aieml.c:66-133`
(`_XAieMl_LockAcquire` / `_XAieMl_LockRelease`):

```
acquire off = BaseAddr + LockId*LockIdOff + RelAcqOff + (v7 << 2)
release off = BaseAddr + LockId*LockIdOff              + (v7 << 2)
v7    = (uint8_t)LockVal & 0x7F   (xaie_locks_aieml.c:34, MASK)
shift = 2                          (xaie_locks_aieml.c:35, SHIFT)
success = bit 0 of the read value  (xaie_locks_aieml.c:37-39)
```

`BaseAddr`, `LockIdOff` (0x400) and `RelAcqOff` (0x200) are per tile-type fields of the vendored
`XAie_LockMod` instances, identical between AIE2 and AIE2P (compared field by field), so `Lock.cpp`
does not branch on generation for them: `global/xaie2pgbl_reginit.c:2384-2401` (core), `:2409-2426`
(shim/noc), `:2434-2451` (memtile), against `global/xaiemlgbl_reginit.c:2448-2513`. The `BaseAddr`
values are included from the vendored param headers rather than copied, so the file tracks a submodule
bump: `XAIE2PGBL_{MEMORY,NOC,MEM_TILE}_MODULE_LOCK_REQUEST` at `xaie2pgbl_params.h:11005/19195/33546`
and the `XAIEMLGBL_` equivalents at `xaiemlgbl_params.h:11076/19226/33533`.

**The plain VALUE range** (`LockN_Value`), one 32-bit register per lock at `LockSetValBase + id*0x10`,
a 6-bit field (mask 0x3F, DEFVAL 0), read/write with no acquire semantics
(`locks/xaie_locks_aieml.c:150-198`). DEFVAL 0 is also the reset value, matching the all-zero row
`chess_compiler_tests_aie2/08_tile_locks` expects before any lock is touched. Bases:
`XAIE2PGBL_{MEMORY,NOC,MEM_TILE}_MODULE_LOCK0_VALUE` at `xaie2pgbl_params.h:10645/15919/32410`,
`XAIEMLGBL_` at `xaiemlgbl_params.h:10716/15950/32397`.

**Acquire/AcquireGreaterEqual/Release polarity** is not spelled out in aie-rt's C comments, so it is
grounded one level up in how mlir-aie produces the signed value. Three call sites agree that
AcquireGreaterEqual is encoded as a NEGATIVE request (magnitude = threshold) and plain Acquire is
non-negative (exact match): `AIEOps.td:1472-1476` (the `UseLockOp` doc); `AIERT.cpp:322-333`, which
negates on `acquireGE`; and `chess_compiler_tests_aie2/08_tile_locks/test.cpp:51-52`, which passes a
literal `-2` to mean "wait for two releases" (`AIETargetXAIEV2.cpp:855-861` shows the generated wrapper
forwards `value` unchanged).

**One corner is not nailed down by any single source line:** whether a successful plain (non-GE)
Acquire also decrements by the matched value, the way AcquireGreaterEqual explicitly does. `AIEOps.td`
is silent. `Lock.cpp` decrements in both cases -- one compare-then-subtract datapath gated only by the
sign -- because (a) the encoding gives both modes one signed 7-bit field with no separate opcode,
(b) `Components.h` defines one `tryAcquire` entry point for both, and (c) on an exact match the
subtraction always lands on zero, so it cannot be observed to do anything unsound. That is a coherence
argument, not a witnessed test, and it is called out at the point in the code where it matters.

### 8a.4 Where the stream-switch model comes from, and what it refuses

Register offsets and field layouts come from `global/xaie2pgbl_params.h` (AIE2P / npu2), spot-checked
byte-identical to the AIE2 map at every address used -- `STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0` = 0x3F100
in both `xaie2pgbl_params.h:4090` and `xaiemlgbl_params.h:4221`;
`PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL` = 0x3F000 in both `xaie2pgbl_params.h:12556` and
`xaiemlgbl_params.h:12647` -- so one table serves both `Generation` values without branching. Behaviour
comes from `stream_switch/xaie_ss.c` and the `XAie_StrmMod` layout in `xaiegbl_regdef.h:198-231`.

A header matches a slave slot when `(id & mask) == slot id`, so a SET mask bit must match rather than
being a don't-care. mlir-aie applies that same predicate to the same two fields when checking a rule for
false matches (`AIECreatePathFindFlows.cpp:1155`), and aie-rt writes the mask into `SlotMask` verbatim
(`xaie_ss.c:646`), so the register field carries the polarity the IR does.

Bundle naming and counts are cross-checked against mlir-aie's own model (`AIETargetModel.h`'s
`getNumDestSwitchboxConnections` / `getNumSourceSwitchboxConnections`, implemented for AIE2 in
`AIETargetModel.cpp:872-1019`, and the `WireBundle` enum in `AIEAttrs.td:53-60`). Every port count
matches tile type by tile type with one exception, at `kShimLayout`: for the Shim/PL trace-slave count
aie-rt says 2 and mlir-aie says 1. We follow aie-rt, since that is the register map this file drives.

`aiesim::PortBundle` has 9 members; aie-rt's `StrmSwPortType` (`xaiegbl.h:229-240`) has the same 9 plus
`UCTRLR` for the AIE2P microcontroller tile. `PortBundle` has no slot for it, so uC-tile stream ports are
out of scope -- `DeviceModel`/`TileType` has no uC tile type either. mlir-aie's `WireBundle` renames Ctrl
to TileControl and separately lists PLIO and NOC; those are shim-side names for connections aie-rt treats
as the plain South bundle plus a distinct shim-mux block (`getNumDestShimMuxConnections`), which is not
part of the stream switch.

**Two things are NOT grounded, so the model faults rather than guessing:**

* **The bit layout of a packet HEADER WORD on the wire.** aie-rt's driver only ever configures match
  registers; nothing in the vendored tree defines which bits of an in-flight stream word carry the packet
  id, because that is a datapath fact aie-rt does not model. We fix our own encoding (bits [4:0] = packet
  id, matching the 5-bit `XAIE_PACKET_ID_MAX` in `xaiegbl.h:49`) and say so where it is used.
* **True hardware arbitration.** Two slave ports whose slot rules resolve to the same (arbiter, msel) pair,
  contending in the same cycle for the same master. aie-rt's registers select an arbiter/msel per port; the
  policy resolving simultaneous contention lives in silicon we have no register-level description of.
  `stepPacketSwitch()` faults via `Array::error()` rather than inventing a tie-break.

### 8a.5 Where the DMA model comes from, and what it leaves unmodelled

Every register offset and bit field in `Dma.cpp` is read out of the vendored aie-rt tables:
`global/xaie2pgbl_params.h` (offsets/masks); `global/xaie2pgbl_reginit.c` for which struct field maps to
which word/Idx (`Aie2PTileDmaProp`, `Aie2PMemTileDmaProp`, `Aie2PShimDmaProp` and their
BdEn/Pkt/Lock/AddrMode/Buffer sub-tables); `dma/xaie_dma_aieml.c` for how a BD is packed into words
(`_XAieMl_{Tile,MemTile,Shim}DmaWriteBd`/`ReadBd`); and `dma/xaie_dma.c` for channel control and
start-queue address arithmetic (`_XAie_DmaChannelControl`, `XAie_DmaChannelSetStartQueueGeneric`,
`_XAieMl_DmaGetChannelStatus`).

AIE2 and AIE2P were spot-checked side by side for the core-tile block -- BD base 0x1D000, ctrl 0x1DE00,
start-queue 0x1DE04, status 0x1DF00, and every bit position used -- and are identical; both generations
run through the same aie-rt C functions (`xaie_dma_aieml.c` has no `#ifdef` on generation). One layout
table therefore serves both `Generation` values.

**Cross-check.** mlir-aie's own lowering confirms the base-address arithmetic without going through
aie-rt at all: `AIETargetModel.cpp:822-869` (`getDmaBdAddress` / `getDmaControlAddress`) computes the
identical `0x1D000 + bd*0x20` and `0x1DE00 + ch*0x8 (+0x10 for MM2S)` plus the memtile/shim equivalents
from first principles. `AIETargetShared.cpp:86-133` (`generateXAieDmaSetMultiDimAddr`) confirms the n-D
dimension order -- MLIR lists dims outermost-first, D0 is always the last entry -- and that stepsize and
wrap are 32-bit-word granular, matching the doc comment at `xaie_dma.c:443-447`.

**Not modelled**, either ungrounded or out of scope; each ungrounded case is a hard error at the point it
would matter rather than a silent guess:

* **Packet-switched header insertion.** PktEn/PktType/PktId are decoded, but the on-wire header word
  format was not grounded from source, so a BD with `PktEn=1` errors.
* **The Iteration dimension's address offset** (IterCurr/Iter.Wrap/Iter.StepSize). Decoded, but used only
  to detect the case where it would matter (`IterWrap>1` or `IterCurr!=0`), which errors.
* **Zero-padding** (memtile D0-D2 pad before/after): not in the required field list, not decoded.
* **AXI/NoC shim properties** (SMID, AxCache, AxQoS, BurstLen, SecureAccess), compression, out-of-order
  completion, FoT mode, controller ID, channel reset: no bus or compute modelling here needs them, so
  they are left unread.

### 8a.6 Where the per-tile device constants come from

`fillAIE2Family` in `Device.cpp` fills the constants shared by every AIE2-family device this simulator
builds (AIE2/npu1, AIE2P/npu2, and the Versal xcve2802 shape, an `AIE2TargetModel` subclass that
overrides none of them). Each constant carries its own one-line citation at the assignment; this is the
aggregate picture. Two sources agree exactly wherever both cover a field.

**(a) mlir-aie's `AIETargetModel.h`** -- `AIE2TargetModel`, the base every one of these devices derives
from; `BaseNPU1TargetModel` / `BaseNPU2TargetModel` / `VE2802TargetModel` override only shape:
`getColumnShift`/`getRowShift` at :738-739; `getLocalMemorySize` (core data) at :640 (0x10000);
`getMemTileSize` at :711 (0x80000); `getNumLocks` (16 core/shim, 64 memtile) at :645-647; `getNumBDs`
(16 core/shim, 48 memtile) at :654-656.

**(b) aie-rt's per-generation register-init tables.** The `aie2ipu` tables (npu1's
`XAIE_DEV_GEN_AIE2IPU`) and the `aie2p` tables (npu2's `XAIE_DEV_GEN_AIE2P_STRIX_B0`) are numerically
identical for every field used -- confirmed by reading both, not assumed. In
`global/xaie2ipugbl_reginit.c`: ProgMemSize/ProgMemHostOffset/DataMemSize at :2270-2273; core data mem
Size=0x10000 and memtile Size=0x80000 at :2298, :2306; lock counts 16/16/64 at :2358, :2383, :2408; BD
counts at :391-405, :627-641, :878-892. In `global/xaie2pgbl_reginit.c`: the same four at :181-184;
:2187, :2195; :2387, :2412, :2437; MemTile BDs=48 / NumChannels=6 at :1656, :1670; core BDs=16 /
channels=2 at :1893, :1907; shim BDs=16 / channels=2 at :2144, :2158.

`coreProgMemSize`, `progMemHostOffset` and every DMA-channel count have only source (b):
`AIETargetModel.h` has no field for ELF program-memory layout or per-tile-type channel counts
(`getNumBDs` covers buffer descriptors, not channels).

**`XAIE_BASE_ADDR = 0x40000000` has no aie-rt counterpart to cross-check**, and is not the one to use.
It is a constant mlir-aie picks when building the `XAie_Config` passed to `XAie_CfgInitialize`
(`AIERT.cpp:187,264`), not a property of the silicon, and aie-rt just stores what it is given. It also
never takes effect on that path: `AIERT.cpp:280` calls `XAie_SetupPartitionConfig` first, which makes
`NumCols` nonzero, so the copy at `global/xaiegbl.c:198-202` is skipped and the live base stays
`XAIE_PARTITION_BASE_ADDR` (0x0, `AIERT.cpp:190`). The base this simulator sees is the HOST program's:
the generated `mlir_aie_init_libxaie()` sets `XAieConfig->BaseAddr = 0x20000000000`
(`AIETargetXAIEV2.cpp:383`) and never calls `XAie_SetupPartitionConfig`, so `NumCols` is still 0 at
`XAie_CfgInitialize` and that copy does take effect. Using the compiler-side number instead rejects
every access a real host program makes -- caught by the aie-rt integration test, which is why that tier
exists.

### 8a.7 Why the address decode does not reuse aie-rt's inverse helpers

Address decode is the first thing the simulator has to get right: everything arrives as a flat address at
`ess_Write32` / `ess_Read32`. aie-rt builds those as `(Col << ColShift) | (Row << RowShift) | RegOff`
(`_XAie_GetTileAddr`, `common/xaie_helper.h:145`).

Its inverse helpers are not usable for the reverse direction. `_XAie_GetRowfromRegOff` and
`_XAie_GetColfromRegOff` (`xaie_helper.c:920-931`) are each off by one field:
`GetRowfromRegOff` returns the low `RowShift` bits, which is the intra-tile offset, and
`GetColfromRegOff` returns the bits between `RowShift` and `ColShift`, which is the row. The defect is
latent upstream because those two feed only the informational Col/Row fields of transaction command
headers (`xaie_helper.c:940-998`), and those consumers use the full `RegOff` for the actual access. A
simulator that decodes for real cannot inherit that.

## 8. Honest risks

* **Vector ISA size.** Vector load/store dominates the AIE2P opcode space (`VST` 184, `VLDA` 163,
  `VLDB` 95, `VLD` 55) with another 112 in the MAC family. Core phase 3 is the long pole of the whole
  proposal, and it is measured in months of hand-written semantics, not weeks. The mitigation is the
  fault contract in 4.3 plus a coverage report, so partial support is visible rather than silently
  wrong.
* **Existing tests use wall-clock sleeps, and this model has no wall clock.** Several in-scope tests
  synchronise with `usleep`/`sleep` rather than by polling a register, for example
  `test/unit_tests/chess_compiler_tests_aie2/04_shared_memory/test.cpp:69`. Nothing calls an `ess_*`
  entry point during a sleep, so zero simulated cycles pass and the sleep does nothing. Worse, the lock
  timeouts those tests pass are microsecond values calibrated against wall-clock hardware, and under
  this model `XAie_SimIO_MaskPoll` turns them into a bound on poll iterations, hence on simulated
  cycles: a `LOCK_TIMEOUT` of 100 gives a core 100 cycles to finish. Converted tests need to wait on a
  register or run to quiescence rather than sleep, and their timeouts need re-reading as iteration
  counts. This is a test-conversion cost, not a model bug, but it is not free and it is why phase 4 is
  its own phase.
* **Read-modify-write is not atomic in this model.** `XAie_SimIO_MaskWrite32` is a `Read32` followed by
  a `Write32`, and the model advances the clock on host accesses, so hardware could evolve between the
  sampled value and the applied one in a way silicon would not allow. The mitigation is that only reads
  advance the clock (see 4.1), which makes the modify-and-write half atomic with respect to the sample.
* **Register-map fidelity.** The aie-rt headers give offsets and fields, not behaviour. Behaviour comes
  from reading the aie-rt driver modules that program them, and from the tests. Where behaviour is
  genuinely unknown, the model should fault rather than guess.
* **Some datapath facts are not in the register map at all, and this is the sharpest divergence risk.**
  aie-rt configures the hardware; it does not describe what travels on a wire. Two concrete cases found
  while building the stream switch: the bit layout of an in-flight packet header word, and the polarity
  of a slot match mask. Nothing in the vendored tree defines either, because neither is a register.
  Where the model has had to fix an encoding it says so at the point of use, but a fixed encoding that
  disagrees with silicon would make packet-switched designs pass here and fail on hardware. These are
  the first things to check in the differential run against a real NPU, and until that run happens
  packet routing should be treated as unvalidated rather than merely untested. True arbitration between
  two slaves resolving to the same arbiter and msel in one cycle is in the same category and currently
  faults rather than picking a tie-break.
* **Divergence from hardware.** A simulator that disagrees with silicon is a liability. The intended
  guard is running the same design both ways on a machine that has an NPU, and comparing outputs. That
  cannot be a CI gate on unlicensed runners, but it can be a periodic job.
* **Bundle-per-cycle timing is right for the datapath, and wrong for everything else.** Revised
  2026-08-02: this entry used to say bundle-per-cycle was simply wrong. It is not -- the machine has no
  interlocks, so one bundle per cycle is what the core does (see core phase 4). What the model still
  lacks is fabric stall: lock, stream, DMA and memory-port waits. So this remains functional simulation
  with an ordering model rather than cycle-approximate, but the missing piece is the array, not the ISA.
  No test should assert an ABSOLUTE cycle count as though it predicted hardware or matched Vitis.
  Asserting a RELATIVE invariant is a different thing and is encouraged: the stream-switch tests check
  that a route takes the same number of cycles in every direction, which is not a performance claim but
  a check that the model is synchronous at all. That distinction is what caught the worst bug found so
  far, so it is worth stating rather than leaving to taste.
