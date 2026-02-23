<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------===//-->

# AIECC - C++ AIE Compiler Driver

This is a high-performance C++ implementation of the AIE compiler driver that provides full feature parity with the Python `aiecc.py` tool. The C++ version eliminates all subprocess calls to `aie-opt` and `aie-translate` by using direct MLIR C++ APIs, resulting in significantly faster compilation.

## Overview

The `aiecc` tool orchestrates the complete compilation flow for AIE (AI Engine) devices:
- **In-memory MLIR transformation** using PassManager C++ API
- **Core compilation** with Peano or xchesscc toolchain
- **Resource allocation and routing** passes
- **NPU instruction generation** via direct API calls
- **ELF generation** for NPU instructions (via aiebu-asm)
- **CDO (Configuration Data Object) generation**
- **PDI (Partial Device Image) generation**
- **xclbin packaging** (including extending existing xclbins)
- **Multi-device and multi-core support**
- **External object file linking** via `link_with` attribute

## Usage

```bash
aiecc [options] <input.mlir>
```

### Common Options

- `--version` - Show version information
- `--verbose` / `-v` - Enable verbose output
- `--tmpdir <dir>` - Directory for temporary files (default: `<input>.prj`)
- `--compile` / `--no-compile` - Enable/disable AIE core compilation
- `--link` / `--no-link` - Enable/disable AIE code linking
- `--alloc-scheme <scheme>` - Buffer allocation scheme (basic-sequential or bank-aware)
- `-O <level>` - Optimization level (0-3, default: 2)
- `-n` - Dry run mode (don't execute commands)

### Output Generation Options

- `--aie-generate-npu-insts` - Generate NPU instruction TXN sequence
  - `--npu-insts-name <pattern>` - Output filename pattern (default: `{0}_{1}.bin`)
  
- `--aie-generate-cdo` - Generate CDO (Configuration Data Object)
  
- `--aie-generate-pdi` - Generate PDI binary for configuration
  - `--pdi-name <pattern>` - Output PDI filename pattern (default: `{0}.pdi`)
  
- `--aie-generate-xclbin` - Generate xclbin
  - `--xclbin-name <pattern>` - Output xclbin filename pattern (default: `{0}.xclbin`)
  - `--xclbin-input <file>` - Extend existing xclbin with additional kernel/PDI
  - `--xclbin-kernel-name <name>` - Kernel name in xclbin (default: `MLIR_AIE`)
  - `--xclbin-instance-name <name>` - Instance name in xclbin (default: `MLIRAIE`)
  - `--xclbin-kernel-id <id>` - Kernel ID in xclbin (default: `0x901`)

- `--aie-generate-elf` - Generate ELF for NPU instructions (via aiebu-asm)
  - `--elf-name <file>` - Output ELF filename (default: `design.elf`)

- `--generate-full-elf` - Generate complete ELF with PDIs and instruction sequences
  - `--full-elf-name <file>` - Output full ELF filename (default: `aie.elf`)

- `--aie-generate-txn` - Generate transaction binary MLIR for configuration
  - `--txn-name <pattern>` - Output filename pattern (default: `{0}_transaction.mlir`)

- `--aie-generate-ctrlpkt` - Generate control packets for configuration
  - `--ctrlpkt-name <pattern>` - Output filename for control packet binary (default: `{0}_ctrlpkt.bin`)
  - `--ctrlpkt-dma-seq-name <pattern>` - Output filename for DMA sequence (default: `{0}_ctrlpkt_dma_seq.bin`)
  - `--ctrlpkt-elf-name <pattern>` - Output filename for combined ELF (default: `{0}_ctrlpkt.elf`)

### Device and Sequence Selection

- `--device-name <name>` - Compile only the specified device
- `--sequence-name <name>` - Compile only the specified runtime sequence

### Compiler Options

- `--xbridge` / `--no-xbridge` - Link using xbridge (default: enabled)
- `--xchesscc` / `--no-xchesscc` - Compile using xchesscc vs Peano (default: xchesscc)
- `--peano <dir>` - Peano compiler installation directory
- `--aietools <dir>` - Vitis aietools installation directory (auto-discovered from PATH or `AIETOOLS_ROOT`)
- `--aiesim` - Generate aiesim work folder
- `--dynamic-objFifos` - Use dynamic object FIFOs
- `--packet-sw-objFifos` - Use packet-switched flows
- `--generate-ctrl-pkt-overlay` - Generate control packet overlay
- `--dump-intermediates` - Dump intermediate MLIR files for debugging

### Compilation Mode Options

- `-j <n>` / `--nthreads <n>` - Number of parallel threads for core compilation
  - `-j 0` - Auto-detect based on CPU count
  - `-j 1` - Sequential compilation (default)
  - `-j 4` - Use 4 parallel threads
- `--unified` / `--no-unified` - Unified compilation mode
  - `--unified` - Compile all cores together into one object file, then link each separately
  - `--no-unified` - Compile cores independently (default, can use with `-j` for parallelism)

## Examples

### Basic Compilation

```bash
aiecc input.mlir
```

This will:
1. Parse the input MLIR file
2. Run resource allocation and routing passes
3. Create a project directory `input.mlir.prj/` with intermediate files

### Generate NPU Instructions

```bash
aiecc --verbose --aie-generate-npu-insts input.mlir
```

This generates NPU instruction binaries for all runtime sequences in the design.

### Generate Complete xclbin

```bash
aiecc --aie-generate-xclbin --xclbin-name=final.xclbin input.mlir
```

This generates CDO files, PDI, and packages everything into an xclbin.

### Multi-Device with Filtering

```bash
aiecc --device-name=npu1_1col \
      --sequence-name=sequence_0 \
      --aie-generate-npu-insts \
      input.mlir
```

This processes only the specified device and sequence.

### Dry Run for Debugging

```bash
aiecc -n --verbose --aie-generate-npu-insts input.mlir
```

This shows what commands would be executed without running them.

## Architecture

The C++ aiecc implementation uses **full in-memory compilation** with zero subprocess calls to `aie-opt` or `aie-translate`:

```
Input MLIR (parsed once)
    ↓
In-memory: runResourceAllocationPipeline()
  - Vector to AIEVec conversion
  - Lock ID assignment
  - Object FIFO stateful transform
  - Buffer descriptor assignment
  - Buffer address allocation
    ↓
In-memory: runRoutingPipeline()
  - Pathfinder flow routing
    ↓
For each core:
  In-memory: runLLVMLoweringPipeline()
    - Localize locks, normalize address spaces
    - Core extraction (aie-standard-lowering)
    - LLVM dialect lowering
      ↓
  In-memory: translateModuleToLLVMIR()
      ↓
  [Write LLVM IR to disk]
      ↓
  Peano opt + llc → Object file
      ↓
  In-memory: AIETranslateToLdScript()
      ↓
  Peano clang → ELF
    ↓
In-memory: Update module with ELF paths
    ↓
In-memory: runNpuLoweringPipeline()
  - BD chain materialization
  - DMA to NPU conversion
    ↓
In-memory: AIETranslateNpuToBinary() → NPU instructions
    ↓
In-memory: AIETranslateToCDODirect() → CDO files
    ↓
bootgen → PDI
    ↓
xclbinutil → xclbin
```

## Comparison with Python aiecc.py

| Feature | Python aiecc.py | C++ aiecc |
|---------|----------------|-----------|
| Language | Python 3 | C++17 |
| MLIR API | Python bindings | Native C++ API |
| Pass Execution | In-memory | **In-memory** |
| Translation | Direct API | **Direct API** |
| aie-opt subprocess | None | **None** |
| aie-translate subprocess | None | **None** |
| Core Compilation | Peano/xchesscc | **Peano/xchesscc** |
| External Object Linking | ✅ | ✅ |
| Parallel Cores (`-j`) | ✅ | **✅** |
| Unified Compilation | ✅ | **✅** |
| Host Compilation | ✅ | ❌ (deprecated) |
| Control Packets | ✅ | **✅** |
| Performance | Good | **Better** (no Python overhead) |

### Current Status

The C++ implementation provides **near-full feature parity**:
- ✅ Complete command-line argument parsing
- ✅ MLIR module loading and parsing
- ✅ Multi-device and multi-core support
- ✅ In-memory resource allocation pipeline
- ✅ In-memory routing (pathfinder flows)
- ✅ In-memory LLVM lowering per core
- ✅ In-memory MLIR to LLVM IR translation
- ✅ Core compilation with Peano toolchain
- ✅ External object file linking (`link_with` attribute)
- ✅ NPU instruction generation (direct API)
- ✅ CDO generation (direct API)
- ✅ PDI generation via bootgen
- ✅ xclbin generation via xclbinutil
- ✅ Verbose output and dry-run mode

- ✅ xchesscc compiler support (with xbridge linking)
- ✅ ELF instruction generation (via aiebu-asm)
- ✅ Full ELF generation with PDIs
- ✅ xclbin extension (`--xclbin-input`)
- ✅ xclbin customization (`--xclbin-kernel-name`, `--xclbin-instance-name`, `--xclbin-kernel-id`)
- ✅ Transaction generation (`--aie-generate-txn`)
- ✅ Control packet generation (`--aie-generate-ctrlpkt`)

### Features Not Yet Implemented

The following flags are accepted for backward compatibility but functionality is not yet ported:

| Flag | Status | Notes |
|------|--------|-------|
| `--profile` | TODO | Command execution timing |
| `--progress` | No-op | Rich progress bar (not planned for C++) |
| `--enable-repeater-scripts` | No-op | Failure reproduction scripts (not planned) |
| `--compile-host` / `--host-target` | Deprecated | Host compilation removed |

### Recently Implemented Features

| Flag | Status | Notes |
|------|--------|-------|
| `--link_against_hsa` | ✅ Implemented | Links against HSA runtime for ROCm |
| `--no-materialize` | ✅ Implemented | Skip `aie-materialize-runtime-sequences` pass |
| `--expand-load-pdis` | ✅ Implemented | Controls PDI expansion in NPU lowering |
| `--aiesim` | ✅ Implemented | Full AIE simulator work folder generation |
| `--no-aiesim` | ✅ Works | Sets aiesim=false (default behavior)

## Building

The tool is built as part of mlir-aie:

```bash
cd mlir-aie
mkdir build && cd build
cmake -G Ninja .. \
  -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
  -DMLIR_DIR=<llvm-build>/lib/cmake/mlir
ninja aiecc
```

The `aiecc` binary will be in `build/bin/`.

## Dependencies

- MLIR/LLVM libraries
- AIE dialect libraries (AIE, AIEX, AIEVec)
- External tools (found in PATH or via environment):
  - **Peano toolchain** (`PEANO_INSTALL_DIR` or `--peano`):
    - `opt` - LLVM IR optimization
    - `llc` - LLVM IR to object compilation
    - `clang` - Linking to ELF
  - `bootgen` - For PDI generation
  - `xclbinutil` - For xclbin generation

Note: `aie-opt` and `aie-translate` are **not required** - all MLIR passes and translations are executed in-memory using direct C++ API calls.

## Error Handling

The tool provides clear error messages for common issues:

```
Error: No input file specified
Error: No AIE devices found in module
Error: Could not find Peano installation
Error: Command failed with exit code 1
```

Use `--verbose` to see detailed command execution and debug issues.

## Environment

The tool respects standard MLIR/AIE environment variables:
- `PEANO_INSTALL_DIR` - Peano compiler installation directory
- `AIETOOLS_ROOT` - Vitis aietools installation directory (for xchesscc)
- `VIRTUAL_ENV` - Python virtual environment (for auto-discovering llvm-aie package)

Tools are automatically found in PATH or relative to the aiecc installation directory.

## See Also

- Python aiecc: `python/compiler/aiecc/main.py`
- AIE dialect documentation: `docs/`
