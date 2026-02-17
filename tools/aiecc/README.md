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

This is a C++ implementation of the AIE compiler driver, providing similar functionality to the Python `aiecc.py` tool. The C++ version offers better performance and deeper integration with the MLIR C++ infrastructure.

## Overview

The `aiecc` tool orchestrates the compilation flow for AIE (AI Engine) devices, including:
- MLIR transformation passes
- Resource allocation and routing
- NPU instruction generation
- CDO (Configuration Data Object) generation
- PDI (Partial Device Image) generation
- xclbin generation
- Multi-device support

## Usage

```bash
aiecc [options] <input.mlir>
```

### Common Options

- `--version` - Show version information
- `--verbose` / `-v` - Enable verbose output
- `--tmpdir <dir>` - Directory for temporary files (default: `<input>.prj`)
- `--compile` / `--no-compile` - Enable/disable AIE core compilation (**currently a no-op in the C++ implementation; reserved for future use**)
- `--link` / `--no-link` - Enable/disable AIE code linking (**currently a no-op in the C++ implementation; reserved for future use**)
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

### Device and Sequence Selection

- `--device-name <name>` - Compile only the specified device
- `--sequence-name <name>` - Compile only the specified runtime sequence

### Advanced Options

- `--xbridge` / `--no-xbridge` - Link using xbridge (default: enabled)
- `--xchesscc` / `--no-xchesscc` - Compile using xchesscc (default: enabled)
- `--aiesim` - Generate aiesim work folder
- `--dynamic-objFifos` - Use dynamic object FIFOs
- `--packet-sw-objFifos` - Use packet-switched flows
- `--generate-ctrl-pkt-overlay` - Generate control packet overlay

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

The C++ aiecc implementation follows this compilation flow:

1. **Parse Input** - Load and validate the input MLIR file
2. **Device Discovery** - Find all `aie.device` operations
3. **Resource Allocation** - Run passes for:
   - Lock ID assignment
   - Object FIFO registration and lowering
   - Buffer descriptor assignment
   - Buffer address allocation
4. **Routing** - Create pathfinder flows for inter-tile connections
5. **Artifact Generation** - For each device:
   - NPU instruction generation (if requested)
   - CDO file generation (if requested)
   - PDI generation (if requested)
   - xclbin packaging (if requested)

## Comparison with Python aiecc.py

| Feature | Python aiecc.py | C++ aiecc |
|---------|----------------|-----------|
| Language | Python 3 | C++ 17 |
| MLIR API | Python bindings | Native C++ API |
| CLI Parsing | argparse | LLVM CommandLine |
| Performance | Interpreted | Compiled native |
| Type Safety | Runtime | Compile-time |
| Dependencies | Python runtime + packages | MLIR/LLVM libraries |
| Parallel Compilation | asyncio (implemented) | Planned |
| Progress Reporting | Rich library (implemented) | Planned |

### Current Status

The C++ implementation provides:
- ✅ Complete command-line argument parsing
- ✅ MLIR module loading and parsing
- ✅ Multi-device support
- ✅ Resource allocation pass pipeline
- ✅ Routing (pathfinder flows)
- ✅ NPU instruction generation
- ✅ CDO generation
- ✅ PDI generation via bootgen
- ✅ Basic error handling and dry-run mode

Planned enhancements:
- ⏳ Core compilation (xchesscc/peano integration)
- ⏳ Parallel compilation of cores
- ⏳ Progress visualization
- ⏳ AIE simulator support
- ⏳ Host code compilation
- ⏳ Complete xclbin generation with metadata

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
- AIE dialect libraries (AIE, AIEX)
- External tools (found in PATH or relative to aiecc):
  - `aie-opt` - For running MLIR passes
  - `aie-translate` - For MLIR to binary translation
  - `bootgen` - For PDI generation (optional)
  - `xclbinutil` - For xclbin generation (optional)
  - `xchesscc` - For core compilation (optional)

## Error Handling

The tool provides clear error messages for common issues:

```
Error: No input file specified
Error: No AIE devices found in module
Error: Could not find aie-opt tool
Error: Command failed with exit code 1
```

Use `--verbose` to see detailed command execution and debug issues.

## Environment

The tool respects standard MLIR/AIE environment variables and automatically finds tools in the PATH or relative to its installation directory.

## See Also

- Python aiecc: `python/compiler/aiecc/main.py`
- AIE dialect documentation: `docs/`
- MLIR passes: Run `aie-opt --help` for available passes
