<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Vector Scalar Addition:

This design shows an extremely simple single AIE design, which is incrementing every value in an input vector.

It shows a number of features which can then be expanded to more realistic designs.  

Firstly, a simple 1D DMA pattern is set up to access data from the input and output memories. Small `64` element subtiles are accessed from the larger `1024` element input and output vectors.  Thinking about input and output spaces are large grids, with smaller grids of work being dispatched to individual AIE cores is a fundamental, reusable concept.

Secondly, these `64` element subtiles which are now in the mem tile are split into two smaller `32` element subtiles, and sent to the AIE engine to be processed.  This shows how the multi-level memory hierarchy of the NPU can be used.

Thirdly, the design shows how the bodies of work done by each AIE core is a combination of data movement (the ObjectFifo acquire and releases) together with compute.

Finally, the overall structural design shows how complete designs are a combination of a static design, consisting of cores, connections and some part of the data movement, together with a run time sequence for controlling the design.
A single tile performs a very simple `+` operation where the kernel loads data from local memory, increments the value by `1` and stores it back.

Input data is first brought in to a MemTile using a Shim tile. The size of the input data from the Shim tile is `64xi32`. The data is stored in the MemTile and sent to the AIE tile in smaller pieces of size `32xi32`. Output data from the AIE compute tile to the Shim tile follows the same process, in reverse.


This example does not contain a C++ kernel file. The kernel is expressed in Python bindings that is then compiled with the AIE compiler to generate the AIE core binary.

## Source Files Overview

1. `vector_scalar_add.py`: An IRON (`@iron.jit`) Python design that compiles directly to NPU binaries (XCLBIN + insts.bin) via `--xclbin-path` / `--insts-path`. Running the script standalone (no `--xclbin-path`) JITs the design and verifies it on the NPU end-to-end.

1. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN + `insts.bin`, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the program verifies the results.

1. `test_runlist.cpp`: An alternate testbench that exercises the **same** xclbin/insts pair from `vector_scalar_add.py` but invokes the kernel twice in a single XRT [runlist](https://xilinx.github.io/XRT/master/html/xrt_native_apis.html), chaining run-0's output (`i + 2`) into run-1's input to produce `i + 3`. NPU2 only — `xrt::runlist` is not implemented on Phoenix (NPU1).

## Usage

### Compilation

To compile the design:
```shell
make
```

To compile the single-run C++ testbench:
```shell
make vector_scalar_add.exe
```

To compile the runlist C++ testbench (NPU2 only):
```shell
make vector_scalar_add_runlist.exe
```

### C++ Testbench

To run the single-run testbench:

```shell
make run
```

To run the runlist testbench (NPU2 only):

```shell
make run_runlist devicename=npu2
```

### Controlling artifact output

By default `@iron.jit` writes its artifacts into the on-disk cache
(`$NPU_CACHE_HOME`) with fixed names. When you need the artifacts at chosen
paths — for a Makefile, a bring-up flow, or to grab the PDI — pass explicit
paths to `.compile()`, which bypasses the cache and writes exactly where you
ask.

This design's `--aot-dir` flag demonstrates the pattern (see `aot_compile` in
`vector_scalar_add.py`):

```shell
python3 vector_scalar_add.py --aot-dir ./artifacts
```

which writes and reports each artifact:

```
Wrote artifacts:
  xclbin: artifacts/vector_scalar_add.xclbin  (ok)
  insts : artifacts/vector_scalar_add.insts.bin  (ok)
  pdi   : artifacts/vector_scalar_add.pdi  (ok)
  elf   : artifacts/vector_scalar_add.insts.elf  (ok)
```

The compile-only Makefile flow uses the same knobs as individual CLI flags —
`--xclbin-path`, `--insts-path`, and optionally `--pdi-path` / `--elf-path`.
See [`compilation_stages.md`](../../../programming_guide/compilation_stages.md#controlling-artifact-output)
for the full artifact list and how to locate the cache-mode PDI via
`get_pdi_path()`.

### Running pre-built artifacts (bring your own)

The reverse direction: run an xclbin + instruction binary that was built
*outside* the `@iron.jit` generation path. `NPUKernel` loads any such pair —
whether it came from the `--aot-dir` export above, a Makefile, a raw `aiecc`
invocation, or another tool — so you can run pre-built binaries without
re-generating the design.

This design's `--from-xclbin` / `--from-insts` flags demonstrate it (see
`run_from_artifacts` in `vector_scalar_add.py`):

```shell
# build artifacts however you like (here, reuse the --aot-dir export)
python3 vector_scalar_add.py --aot-dir ./artifacts

# then run them, bypassing JIT generation entirely
python3 vector_scalar_add.py \
    --from-xclbin ./artifacts/vector_scalar_add.xclbin \
    --from-insts ./artifacts/vector_scalar_add.insts.bin
```

Inputs are passed as IRON tensors; the output tensor is pre-allocated and
written in place, so `--problem-size` must match the size the artifacts were
compiled for. Note the run inputs are the **xclbin + insts** — the PDI is
packed *inside* the xclbin and is not a standalone run input on the IRON
runtime.

Because this path bypasses JIT generation, there is no compiled recipe to
validate the artifacts against. An xclbin built for the wrong NPU family runs
without error but silently returns zeros, so pass `--dev` to declare the
family the artifacts target — the run aborts up front if the attached device's
architecture doesn't match:

```shell
python3 vector_scalar_add.py --dev npu2 \
    --from-xclbin ./artifacts/vector_scalar_add.xclbin \
    --from-insts ./artifacts/vector_scalar_add.insts.bin
```
