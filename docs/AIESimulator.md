# An open AIE array simulator, to replace the Vitis dependency of `--get-aiesim`

Status: RFC / design. Nothing here is merged yet.
Scope: AIE2 (npu1) and AIE2P (npu2). AIE1 is explicitly out of scope.

## 1. Why

`aiecc --get-aiesim` is the only hardware-free way to run an mlir-aie design end to end, and it is the
last hard anchor holding Chess and Vitis in this project. Everything else has an open path:
`--no-xchesscc --no-xbridge` compiles cores with Peano today.

The simulator does not. Three separate things make it Vitis-only:

* The `sim/` work folder is consumed by `aiesimulator`, an external Vitis binary
  (`tools/aiecc/aiecc.cpp:402-417` writes `aiesim.sh`, which runs `aiesimulator --pkg-dir=...`).
* `sim/ps/ps.so` is compiled against `adf/wrapper/wrapper.h`, `xtlm.h`, `libsystemc` and `libxtlm`
  from `$AIETOOLS_ROOT` (`tools/aiecc/aiecc.cpp:311-392`).
* `--get-aiesim` forces `--xbridge` and refuses `--no-xchesscc`
  (`tools/aiecc/CommandLineOptions.h:493-502`, "the AIE simulator consumes Chess-compiled cores"), so
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

mlir-aie already builds `libxaienginecdo` with `__AIESIM__` defined
(`runtime_lib/xaiengine/lib/CMakeLists.txt:57`). That selects aie-rt's `XAIE_IO_BACKEND_SIM`
(`third_party/aie-rt/driver/src/io_backend/xaie_io.c:34-35`), whose entire implementation
(`third_party/aie-rt/driver/src/io_backend/ext/xaie_sim.c`) forwards every register access to seven
externally-declared C functions:

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

Those symbols are the whole boundary. Everything above them is open code we already ship: aie-rt's
tile, DMA, lock, stream-switch and core modules, and the `aie_inc.cpp` that `aiecc` generates. Core ELFs
are not handed to the simulator through a side channel either. `_XAie_LoadProgMemSection`
(`third_party/aie-rt/driver/src/core/xaie_elfloader.c:221-260`) writes program sections into program
memory with ordinary block writes, so the ELF arrives as MMIO traffic exactly as on hardware. The two
`XAie_CmdWrite` cases (`SETSTACK`, `LOADSYM`, `xaie_elfloader.c:44-45`) are debug conveniences.

**So an open simulator is a library that defines those seven symbols over a software model of the
array.** No aie-rt fork is needed, no new IO backend, no vendored patch to maintain. That matters:
aie-rt is a third-party submodule shared with XRT and iree-amd-aie, and a fork of it would be a
permanent tax.

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
  out with aie-rt's own shifts (`_XAie_GetTileAddr`, and the inverse in
  `third_party/aie-rt/driver/src/common/xaie_helper.c:920-931`). Register offsets and field layouts are
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
* **Time.** A cycle counter and a fixed round-robin over components. The array advances inside the
  `ess_*` entry points, so a host polling loop (which is how aie-rt implements every wait) makes
  progress naturally, with no threads and no reordering.

The fabric deliberately does not know how to execute an instruction.

### 4.2 Core engine, in llvm-aie

Executing AIE2/AIE2P instructions belongs where the ISA is defined. llvm-aie already has the whole
decode side: full disassemblers for AIE2, AIE2P and AIE2PS, generated decode tables, instruction
printers and encoders. What is missing is semantics, and semantics are a property of the backend.

There is a second, independent reason to put it there. llvm-aie today has no execution tests at all:
`llvm/test/CodeGen/AIE` compiles and FileChecks, and nothing runs. An in-tree instruction interpreter
gives the backend end-to-end execution testing for the first time, which is worth having on its own
even if mlir-aie never called it.

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

## 5. Why register-level rather than graph-level

The Vitis simulator is configured from four descriptor files that `aiecc` generates from the MLIR
(`tools/aiecc/aiecc.cpp:259-306`). That design has a blind spot: it simulates the design the compiler
*described*, not the design the configuration code *programmed*. Configuration bugs, which are a real
and recurring class in this project, are invisible to it.

Driving the model from register writes removes both the descriptor files and the blind spot, and it
generalises for free. Anything that reduces to register writes can be fed to the same model: the
`aie_inc.cpp` path, a CDO blob, or an XRT transaction binary. That last one is what actually runs on
npu1 and npu2 hardware, and there is currently no hardware-free way to execute it at all.

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

## 8. Honest risks

* **Vector ISA size.** AIE2P defines 242 intrinsics and AIE2 defines 317, and the machine instruction
  count is larger still. Core phase 3 is the long pole of the whole proposal. The mitigation is the
  fault contract in 4.3 plus a coverage report, so partial support is visible rather than silently
  wrong.
* **Register-map fidelity.** The aie-rt headers give offsets and fields, not behaviour. Behaviour comes
  from reading the aie-rt driver modules that program them, and from the tests. Where behaviour is
  genuinely unknown, the model should fault rather than guess.
* **Divergence from hardware.** A simulator that disagrees with silicon is a liability. The intended
  guard is running the same design both ways on a machine that has an NPU, and comparing outputs. That
  cannot be a CI gate on unlicensed runners, but it can be a periodic job.
* **Bundle-per-cycle timing is wrong.** Until phase 6 it should be described as functional simulation
  with an ordering model, not as cycle-approximate, and tests should not check cycle counts.
