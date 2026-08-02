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
  it does would make it useless); `AIE_SIM_STRICT=1` promotes them, and the recorded set is emitted as
  a coverage report so "unmodelled" is a number rather than a surprise.

  That runtime report says what one design touched. The static counterpart says what the model
  covers at all: `register_coverage <params-header> [device]` takes the register universe from
  aie-rt's own generated database (a define is an address when a sibling `_WIDTH` exists, which
  separates the ~2.1k addresses from the ~31k field defines) and asks `RegisterFile::isClaimed()`
  once per address, so the answer comes from the same lookup a running design hits rather than from
  a hand-kept list that drifts. As of 2026-08-02, AIE2P/npu2 is **1176 of 2141 addresses, 54.9%** --
  memtile 74.3%, shim 54.9%, core 35.3%; AIE2/xcve2802 is 53.7% of a slightly larger map. Unclaimed
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

**Scope.** Expose the control block plus `R`/`P`/`M`/`S` first: those are scalar, 32 bits, one slot each,
and are what a host debug read actually wants. The vector and accumulator families need the part-assembly
rule above and an engine that models them at all, so they belong with the vector phase, not here. Writes
go through `writeRegister` with the same table; nothing in the current tests needs them, so they can stay
faulted until something does.

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
4. **Timing.** Replace the flat one-bundle-per-cycle cost with the per-instruction itineraries the
   backend already carries. `aie2p/AIE2PGenSchedule.td` holds roughly 4000 `InstrItinData` entries, which
   is the densest machine-readable AIE2P timing data in either repository.

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
* **Bundle-per-cycle timing is wrong.** Until the core engine's timing phase lands, this should be
  described as functional simulation with an ordering model, not as cycle-approximate. No test should
  assert an ABSOLUTE cycle count as though it predicted hardware or matched Vitis.
  Asserting a RELATIVE invariant is a different thing and is encouraged: the stream-switch tests check
  that a route takes the same number of cycles in every direction, which is not a performance claim but
  a check that the model is synchronous at all. That distinction is what caught the worst bug found so
  far, so it is worth stating rather than leaving to taste.
