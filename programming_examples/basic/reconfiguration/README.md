<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Reconfiguration

This example benchmarks **five different ways to reconfigure and re-run an AIE
core array** on the NPU, from the traditional (slow) full xclbin reload down to
on-device partial reconfiguration via `load_pdi`, raw block-writes, and control
packets.

## The design

`reconfiguration.py` builds a `cols` x `rows` array of compute cores. Each core
writes a single `i32` (its own global index) into a dedicated ObjectFIFO. A shim
has only two S2MM DMA channels, so a column's `rows` core FIFOs are **joined in
the mem tile** and forwarded to the shim; a runtime sequence then drains every
column into the host buffer, which equals `[0, 1, ..., cols*rows - 1]`. After
sending its value each core executes a run of no-op `aie.event` instructions
that pad its program memory (up to the ~16 KB per-core limit).

Three flows are emitted from the same building blocks:

- **`--flow reconfig`**: three `aie.device`s (`@worker`, `@empty`, `@main`).
  `@main`'s runtime sequence loads `@empty` (reset) then loads and runs
  `@worker` via `aiex.configure` / `aiex.run`. Built as a full ELF, this is the
  basis of the `load_pdis`, `blockwrites`, and `control packets` approaches
  (chosen by `aiecc` flags).
- **`--flow single`**: one `aie.device`, no `load_pdi`. The cores loop, so the
  design can be re-run from the host through the ordinary xclbin + insts flow.
- **`--flow empty`**: a single empty device whose xclbin/PDI resets the array.
  The xclbin/runlist approaches load it between iterations to force a real
  reconfiguration (otherwise the configuration is cached).

## The five approaches

| chart label | mechanism | Makefile target |
|---|---|---|
| **separate xclbins** | worker + empty are separate xclbins (two contexts); each iteration runs the worker then the empty reset | `run_separate` |
| **XRT runlist** | worker + empty runs chained in one `xrt::runlist` | `run_runlist` |
| **load_pdis** | full-ELF, `aiex.configure` lowers to `load_pdi` | `run_loadpdi` |
| **blockwrites + empty reset** | full-ELF, `aiecc --expand-load-pdis` (write32s + empty PDI reset) | `run_blockwrites` |
| **control packets + load_pdi overlay** | full-ELF, `aiecc --load-pdi-to-ctrl-pkt` (stream config through DMA) | `run_ctrlpkt` |

> **Note on `XRT runlist`.** An `xrt::runlist` is bound to a single
> `hw_context` and cannot switch the loaded PDI between its runs, so the empty
> run added from a second context does **not** reset the array — the runlist
> just re-executes the already-loaded (looping) worker. Its measured time is
> therefore independent of the array size (flat ~80 us for 1x1 and 8x4 alike),
> unlike every real reconfiguration approach, so `plot.py` omits it from the
> chart. It is still measured into the CSV for the record.

## Testbench

A single `test.cpp` covers all approaches, selected at compile time (default =
separate xclbins, `-DRUNLIST`, `-DFULL_ELF`), so everything is timed in C++.
Each run performs `ITERS` timed iterations and prints:

```
runtimes_us: t0,t1,...      # per-iteration device time
stats_us: mean,min,max
```

Every iteration checks the output equals `[0, 1, ..., cols*rows-1]`.

## Parameters

- `COLS`, `ROWS`: array shape (up to 8 x 4 on NPU2 / Strix).
- `NOPS`: no-op `aie.event` instructions padding each core's program memory
  (~192 + 4·`NOPS` bytes; the 16 KB limit is reached near `NOPS=4000`).
- `ITERS`: number of timed iterations.

Design-artifact names embed `COLS`/`ROWS`/`NOPS` so changing a parameter never
picks up a stale artifact.

## Usage

Run one approach on an NPU2 (Strix) device:

```bash
make COLS=4 ROWS=2 NOPS=2000 ITERS=12 run_runlist
```

Run the full benchmark (all approaches over small/medium/large array sizes),
which writes the raw per-iteration runtimes to `benchmark.csv`, then plot it as
a grouped bar chart (`benchmark.png`, black background, runlist excluded):

```bash
python3 benchmark.py      # -> benchmark.csv
python3 plot.py           # benchmark.csv -> benchmark.png
```

## Source Files

- `reconfiguration.py`: the parametrized design (`--flow reconfig|single|empty`).
- `test.cpp`: the C++ host testbench / micro-benchmark (three compile-time modes).
- `benchmark.py`: drives every approach over several array sizes and writes the
  raw runtimes to `benchmark.csv`.
- `plot.py`: reads `benchmark.csv` and writes the grouped bar chart.
