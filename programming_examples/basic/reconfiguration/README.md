<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Reconfiguration

This example measures five approaches that reconfigure a core array on the NPU
and run it again. The approaches range from a full xclbin reload to partial
reconfiguration on the device through `load_pdi`, block writes and control
packets.

## The design

`reconfiguration.py` builds a `cols` x `rows` array of compute cores. Each core
writes one `i32`, its own global index, into a dedicated ObjectFIFO. A shim tile
has two S2MM DMA channels, so the mem tile of each column joins the `rows` core
FIFOs and forwards them to the shim tile. A runtime sequence drains every column
into the host buffer. The buffer then holds `[0, 1, ..., cols*rows - 1]`.

After a core sends its value, it executes a run of `aie.event` instructions.
These instructions pad the program memory of the core towards the limit of
16 KB.

The generator emits three variants from the same building blocks:

- `--flow reconfig`: three `aie.device` operations, `@worker`, `@empty` and
  `@main`. The runtime sequence of `@main` loads `@empty` to reset the array,
  then loads and runs `@worker` through `aiex.configure` / `aiex.run`. `aiecc`
  builds this variant as a full ELF. Flags of `aiecc` select the `load_pdis`,
  `blockwrites` and `control packets` approaches.
- `--flow single`: one `aie.device` without `load_pdi`. The cores loop, so the
  host runs the design again through the ordinary xclbin and instruction
  sequence.
- `--flow empty`: one empty device. Its xclbin and PDI reset the array. The
  xclbin and runlist approaches load it between iterations to force a
  reconfiguration, because the device caches the configuration.

## The five approaches

| chart label | mechanism | Makefile target |
|---|---|---|
| **separate xclbins** | worker and empty live in separate xclbins and separate hardware contexts; each iteration runs the worker, then the empty reset | `run_separate` |
| **XRT runlist** | `aiecc --xclbin-input` merges the WORKER and EMPTY kernels into one xclbin and one hardware context; the runlist alternates them | `run_runlist` |
| **load_pdis** | full ELF; `aiex.configure` lowers to `load_pdi` | `run_loadpdi` |
| **blockwrites + empty reset** | full ELF; `aiecc --expand-load-pdis` emits write32 and blockwrite operations, plus an empty PDI as reset | `run_blockwrites` |
| **control packets + load_pdi overlay** | full ELF; `aiecc --load-pdi-to-ctrl-pkt` streams the configuration through a DMA | `run_ctrlpkt` |

> **Why the runlist approach needs a combined xclbin.** An `xrt::runlist` binds
> to one `hw_context`. A runlist over a single kernel repeats the loaded
> configuration, and its time stays flat as the array grows. `--xclbin-input`
> puts the WORKER and EMPTY kernels, which carry distinct PDIs, into one xclbin.
> The runlist then alternates the two kernels inside one context, and each
> switch reconfigures the array.

## Testbench

One `test.cpp` serves all approaches. Preprocessor flags select the approach at
compile time: none for separate xclbins, `-DRUNLIST` for the runlist, and
`-DFULL_ELF` for the three full-ELF approaches. Each run performs `ITERS` timed
iterations and prints:

```
runtimes_us: t0,t1,...      # device time per iteration
stats_us: mean,min,max
```

Every iteration compares the output against `[0, 1, ..., cols*rows-1]`.

## Parameters

- `COLS`, `ROWS`: shape of the core array, up to 8 x 4 on NPU2 (Strix).
- `NOPS`: number of `aie.event` instructions that pad the program memory of each
  core. A core program occupies about 192 + 4·`NOPS` bytes and reaches the limit
  of 16 KB near `NOPS=4000`.
- `SWITCHBOXES`: number of unused compute-tile switchboxes that the generator
  fills with legal stream-switch connections through `aie.switchbox` and
  `aie.connect`. The connections grow the configuration that a reconfiguration
  loads, and they carry no data.
- `ITERS`: number of timed iterations.

The names of the design artifacts embed `COLS`, `ROWS`, `NOPS` and
`SWITCHBOXES`, so a change of one parameter selects a different artifact.

## Usage

Run one approach on an NPU2 (Strix) device:

```bash
make COLS=4 ROWS=2 NOPS=2000 ITERS=12 run_runlist
```

Run the benchmark over all approaches and over small, medium and large array
sizes. `benchmark.py` writes the runtime of every iteration to `benchmark.csv`,
and `plot.py` draws a grouped bar chart into `benchmark.png`:

```bash
python3 benchmark.py      # -> benchmark.csv
python3 plot.py           # benchmark.csv -> benchmark.png
```

## Scaling studies

Two further studies vary one parameter each and draw a line graph:

- Program memory (`benchmark_progmem.py` and `plot_progmem.py`): a 1x1 array
  with no filled switchboxes and `NOPS` from 0 to 4000. One line per approach,
  written to `progmem.png`.
- Switchbox configuration (`benchmark_switchbox.py` and `plot_switchbox.py`):
  one active core with one outbound flow and `SWITCHBOXES` from 0 to 24. One
  line per approach over the number of used switchboxes, written to
  `switchbox.png`.

`run_scaling.sh` holds the invocations for all three figures.

## Source Files

- `reconfiguration.py`: the design generator (`--flow reconfig|single|empty`,
  `--cols`, `--rows`, `--nops`, `--switchboxes`, `--reconfigs`).
- `test.cpp`: the host testbench with its three compile-time modes.
- `benchmark.py` and `plot.py`: the grouped bar chart over all approaches.
- `benchmark_progmem.py` and `plot_progmem.py`: runtime over core program
  memory.
- `benchmark_switchbox.py` and `plot_switchbox.py`: runtime over switchbox
  configuration.
- `run_scaling.sh`: the invocations for all three figures.
