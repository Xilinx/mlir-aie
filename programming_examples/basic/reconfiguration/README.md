<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Reconfiguration

This example demonstrates **reconfiguring the NPU between `aie.device`s at
runtime** using `aiex.configure` and `aiex.run`. A single module holds several
devices, and one device's runtime sequence drives the loading and execution of
another.

The same compute-core array and drain runtime sequence are emitted in two flows
(`reconfiguration.py --flow reconfig|single`):

## `reconfig` flow (default)

The module contains three `aie.device`s:

- **`@worker`** configures an array of compute cores. Each core, immediately as
  it starts executing, writes a single `i32` (its own index) into a dedicated
  ObjectFIFO routed down to the shim, and then executes a run of no-op `aie.event`
  instructions that pad its program memory. `@worker`'s runtime sequence drains
  one `i32` from every core into the host output buffer.

- **`@empty`** is an empty device used only to reset the array between
  reconfigurations (see below).

- **`@main`** is the entry device. Its runtime sequence repeatedly reconfigures
  and runs `@worker` via `aiex.configure` / `aiex.run`.

Each core gets its own ObjectFIFO routed to the shim in its column, so the number
of cores scales the amount of stream-switch routing used across the array.

This flow is built as a full ELF (`aiecc --generate-full-elf`) and driven by the
`pytest` testbench (`test.py`).

## `single` flow

A single `aie.device` with **no reconfiguration and no `load_pdi`**. The runtime
sequence configures the DMAs once and drains the array. The cores loop, so the
configured design can simply be re-run from the host. This flow builds an
ordinary `xclbin` + instruction binary (`insts.bin`) and is driven by the same
C++ testbench:

- `run_xclbin` runs the kernel once.
- `run_runlist` chains `RECONFIGS` runs in a single `xrt::runlist`; because the
  cores loop, each run drains a fresh set of values.

## Testbench

A single `test.cpp` covers all three modes, selected at compile time so the
flows can be compared apples-to-apples (all timed in C++):

- default: xclbin flow, one run.
- `-DRUNLIST`: xclbin flow, `runs` runs chained in an `xrt::runlist`.
- `-DFULL_ELF`: full-ELF reconfig flow (`main:sequence`).

Each mode checks the output equals `[0, 1, ..., CORES-1]` and prints the elapsed
device time.

## Reconfiguration and device reset

`aiex.configure @worker` lowers to a PDI load that resets the whole device and
restarts its cores. Because `@worker`'s cores run exactly once (send their value,
then halt), re-running them requires an actual reload. The firmware, however,
treats a second back-to-back load of the *same* PDI as a no-op. To force a real
reload between two `@worker` configurations, `@main` loads the `@empty` device in
between:

```
configure @worker  →  run once
configure @empty   →  reset
configure @worker  →  run again
...
```

## Parameters

The design (`reconfiguration.py`) is parametrized by three values, wired through
the `Makefile` as `CORES`, `NOPS`, and `RECONFIGS`:

- **Array size** (`--cores`, 1–8): number of cores, each with its own ObjectFIFO
  routed to its column's shim.
- **Program-memory padding** (`--nops`): number of no-op `aie.event` instructions
  appended to each core after it sends its value.
- **Number of reconfigurations** (`--reconfigs`): in the `reconfig` flow, how
  many times `@worker` is configured and run — a plain Python `for i in range(n)`
  loop meta-programmatically emits `n` copies of the configure/run block into the
  MLIR. In the `single` flow it drives the host-side run count (the number of
  runs chained by `run_runlist`) and does not change the MLIR.

All build artifacts embed `CORES`/`NOPS`/`RECONFIGS` in their names (e.g.
`build/final_c4_n64_r1.xclbin`) so changing a parameter never picks up a stale
artifact.

## Source Files

- `reconfiguration.py`: The parametrized design. Emits MLIR to stdout for either
  flow (`--flow reconfig|single`).
- `test.cpp`: The C++ host testbench for all three run modes (see above).

## Usage

Build and run on an NPU2 (Strix) device:

```bash
make run           # reconfig flow (full ELF, -DFULL_ELF)
make run_xclbin    # single flow, one run (xclbin + insts)
make run_runlist   # single flow, RECONFIGS runs (xrt::runlist)
```

Override the parameters:

```bash
make CORES=4 NOPS=64 RECONFIGS=2 run
make CORES=4 NOPS=64 RECONFIGS=3 run_runlist
```

Generate the MLIR directly:

```bash
python3 reconfiguration.py --flow reconfig --cores 4 --nops 64 --reconfigs 2
python3 reconfiguration.py --flow single  --cores 4 --nops 64
```
