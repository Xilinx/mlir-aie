<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
-->

# Reconfiguration

Many workloads, including large language models, execute a sequence of different
operations. A dataflow mapping can implement the whole workload as one design,
but such a design requires substantial integration work. A layer-by-layer
mapping implements each meaningful operation as a standalone operator and
dispatches the operators in sequence at run time.

Layer-by-layer mappings improve modularity because multiple workloads can reuse
the same operator implementations. They also let developers study one operation
at a time. However, they require quick reconfiguration. Slow device-image
replacement can make the layer-by-layer approach impractical.

This example compares three ways to run or reconfigure one NPU2 device image:

- `separate-dispatch`: compile an xclbin and dispatch its worker runtime directly.
- `load-pdi`: use a full ELF with direct PDI loads.
- `expand-load-pdis`: expand PDI loads into register writes.

`reconfiguration.py` contains one `@iron.jit` design. One
`DeviceConfiguration` owns the workers and their runtime sequence. The xclbin
path wraps that configuration in a `Program` whose entry is the worker runtime.
The full-ELF paths use the same configuration and add one coordinator runtime;
the coordinator enters `configuration.configure()` and calls the worker runtime.

The design accepts array dimensions, program-memory padding, switchbox padding,
and the number of configure-and-run operations per full-ELF dispatch:

```bash
python reconfiguration.py --mode expand-load-pdis --cols 4 --rows 2 \
  --nops 2000 --switchboxes 12 --reconfigs 4 --warmup 2 --iters 10
```

The script validates the output and prints every critical-section runtime plus
mean, minimum, and maximum values. Add `--csv` for one row per timed iteration.

`run_scaling.sh` invokes the same script for all modes while varying core
program-memory padding and switchbox configuration. It concatenates the rows
into one CSV:

```bash
ITERS=12 ./run_scaling.sh benchmark.csv
```
