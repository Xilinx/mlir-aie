# AIECC - C++ AIE Compiler Driver

This is a C++ implementation of the AIE compiler driver, providing similar functionality to the Python `aiecc.py` tool. The C++ version offers better performance and integration with the MLIR C++ infrastructure.

## Overview

The `aiecc` tool orchestrates the compilation flow for AIE (AI Engine) devices, including:
- MLIR transformation passes
- Core compilation
- Linking
- Generation of output artifacts (NPU instructions, CDO, xclbin, etc.)

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

- `--aie-generate-npu-insts` - Generate NPU instruction stream
- `--aie-generate-cdo` - Generate CDO (Configuration Data Object)
- `--aie-generate-xclbin` - Generate xclbin file
- `--xclbin-name <name>` - Output xclbin filename (default: `{0}.xclbin`)

## Implementation Status

This is an initial implementation that provides:
- Command-line argument parsing
- MLIR module parsing and loading
- Basic MLIR transformation pipeline orchestration
- Integration with existing AIE tools (aie-opt, aie-translate)

### Future Enhancements

- Complete core compilation workflow
- Artifact generation (CDO, xclbin, PDI, etc.)
- Parallel compilation support
- AIE simulator integration
- Progress reporting

## Comparison with Python aiecc

The C++ implementation follows the same overall architecture as the Python `aiecc.py`:

| Feature | Python aiecc.py | C++ aiecc |
|---------|----------------|-----------|
| Command-line parsing | argparse | LLVM CommandLine |
| MLIR integration | Python bindings | Native C++ API |
| Async execution | asyncio | (planned) |
| Pass management | Python PM API | C++ PassManager |
| Tool invocation | subprocess | llvm::sys::Program |

## Building

The tool is built as part of the mlir-aie project:

```bash
cd mlir-aie
mkdir build && cd build
cmake -G Ninja .. \
  -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
  -DMLIR_DIR=<llvm-build>/lib/cmake/mlir
ninja aiecc
```

## Testing

Tests are located in `test/aiecc/`. Run tests with:

```bash
cd build
ninja check-aie
```

## See Also

- Python implementation: `python/compiler/aiecc/main.py`
- MLIR documentation: https://mlir.llvm.org/
- AIE documentation: https://xilinx.github.io/mlir-aie/
