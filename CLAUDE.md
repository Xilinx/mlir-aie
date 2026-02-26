# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration  
  
### 1. Plan Mode Default  
-   Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)  
-   If something goes sideways, STOP and re-plan immediately - don't keep pushing  
-   Use plan mode for verification steps, not just building  
-   Write detailed specs upfront to reduce ambiguity  
  
### 2. Subagent Strategy  
-   Use subagents liberally to keep main context window clean  
-   Offload research, exploration, and parallel analysis to subagents  
-   For complex problems, throw more compute at it via subagents  
-   One task per subagent for focused execution  
  
### 3. Self-Improvement Loop  
- After ANY correction from the user: update TASKS/PLAN/LESSONS.md"  
with the pattern  
-   Write rules for yourself that prevent the same mistake  
-   Ruthlessly iterate on these lessons until mistake rate drops  
-   Review lessons at session start for relevant project  
  
### 4. Verification Before Done  
-   Never mark a task complete without proving it works  
-   Diff behavior between main and your changes when relevant  
-   Ask yourself: "Would a staff engineer approve this?"  
-   Run tests, check logs, demonstrate correctness  
  
### 5. Demand Elegance (Balanced)  
-   For non-trivial changes: pause and ask "is there a more elegant way?"  
-   If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"  
-   Skip this for simple, obvious fixes - don't over-engineer  
-   Challenge your own work before presenting it  
  
### 6. Autonomous Bug Fixing  
-   When given a bug report: just fix it. Don't ask for hand-holding  
-   Point at logs, errors, failing tests - then resolve them  
-   Zero context switching required from the user  
-   Go fix failing CI tests without being told how  
  
## Task Management  
1.    **Plan First**: Write plan to 'tasks/todo.md with checkable items  
2.    **Verify Plan**: Check in before starting implementation  
3.    **Track Progress**: Mark items complete as you go  
4.    **Explain Changes**: High-level summary at each step  
5.    **Document Results**: Add review section to tasks/todo.md"  
6.    **Capture Lessons**: Update tasks/LESSONS.md' after corrections  
  
## Core Principles  
-   **Simplicity First**: Make every change as simple as possible. Impact minimal code.  
-   **No Laziness**: Find root causes. No temporary fixes. Fix the implementation don’t modify the existing tests. Senior developer standards.  
-   **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.  

## Repository Overview

MLIR-AIE is an MLIR-based toolchain for AI Engine-enabled devices (AMD Ryzen AI NPUs, Xilinx Versal). It provides multiple levels of abstraction through MLIR dialects to program AI Engine cores, describe data movements, and configure array connectivity. The project includes:

- **IRON (Iron Runtime for NPU)**: High-level Python APIs for NPU programming
- **MLIR Dialects**: AIE, AIEVec, AIEX, ADF, XLLVM dialects for different abstraction levels
- **Compiler Infrastructure**: Transformation passes, optimization, and code generation
- **Runtime Libraries**: Host-side (XRT, xaiengine) and device-side runtime support
- **Peano**: LLVM-based compiler for AI Engine processor cores (separate repo: llvm-aie)

## Environment Setup

Before building or running examples, the environment must be properly configured:

```bash
# 1. Initialize and update git submodules (required after cloning)
git submodule update --init --recursive

# 2. Activate Python virtual environment
source ironenv/bin/activate

# 3. Setup XRT (Xilinx Runtime)
source /opt/xilinx/xrt/setup.sh

# 4. Build from wheels (installs mlir-aie and llvm-aie/Peano from pre-built wheels)
bash utils/build-mlir-aie-from-wheels.sh

# 5. Setup mlir-aie environment (sets PATH, PYTHONPATH, LD_LIBRARY_PATH)
source utils/env_setup.sh install
```

> NOTE: xchesscc can be tested locally, the script for running it is in `/home/jmelber/vitisForACDC_9999.sh`

This environment setup is **required** before running any builds or tests.

## Build Commands

### Building from Source

If building the entire toolchain from source (instead of using wheels):

```bash
# Full build from source (requires LLVM/MLIR already built)
mkdir build && cd build
cmake -GNinja \
  -DCMAKE_INSTALL_PREFIX=../install \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm \
  ..
ninja
ninja install
```

Key CMake options:
- `AIE_RUNTIME_TARGETS`: List of runtime targets (x86_64, aarch64), default: x86_64
- `AIE_COMPILER`: Backend compiler selection (XCHESSCC, PEANO, NONE), default: XCHESSCC
- `AIE_ENABLE_BINDINGS_PYTHON`: Enable Python bindings, default: ON
- `PEANO_INSTALL_DIR`: Location of Peano compiler installation

### Building Examples

Programming examples use Makefiles:

```bash
cd programming_examples/basic/passthrough_kernel

# Build the AIE design (generates MLIR, compiles kernels, creates xclbin)
make

# Build and run on hardware
make run

# Clean build artifacts
make clean
```

## Testing

### Running LIT Tests

> **NOTE:** Use `LIT_OPTS="-j4 $LIT_OPTS"` So that we don't flood the shared machines with OOM!

The test suite uses LLVM Integrated Tester (LIT):

```bash
# From build directory
ninja check-aie

# Run tests in a specific directory
lit test/dialect/AIE -v

# Run a single test file
lit test/dialect/AIE/some_test.mlir -v

# Run with specific number of jobs
lit -j 8 test/
```

Test file suffixes: `.mlir`, `.py`, `.test`

### Running Specific Test Categories

```bash
# Run dialect tests
lit test/dialect/

# Run conversion pass tests
lit test/Conversion/

# Run NPU-XRT integration tests (requires hardware)
lit test/npu-xrt/

# Run Python API tests
lit test/python/
```

### Running Python Tests with pytest

```bash
# Run pytest on specific test directories
pytest test/python/

# Run with verbose output
pytest -v test/python/
```

## Code Architecture

### Key Directories

- **`lib/`**: Core C++ implementation
  - `lib/Dialect/`: MLIR dialect implementations (AIE, AIEVec, AIEX, ADF, XLLVM)
  - `lib/Conversion/`: Dialect conversion passes
  - `lib/Targets/`: Backend code generation (NPU, CDO, XAIEV2, HSA, etc.)
  - `lib/Transforms/`: Transformation passes

- **`include/aie/`**: Public header files and TableGen dialect definitions

- **`tools/`**: Executable tools
  - `aie-opt`: MLIR optimizer (applies transformation passes)
  - `aie-translate`: Translation between MLIR and other formats
  - `aiecc/`: C++ compiler driver (new implementation)

- **`python/`**: Python bindings and high-level APIs
  - `python/iron/`: IRON high-level API (Program, Worker, Kernel, Buffer, ObjectFifo, Runtime)
  - `python/compiler/aiecc/`: Python compiler driver (aiecc.py)
  - `python/dialects/`: Auto-generated Python dialect bindings

- **`runtime_lib/`**: Host-side C++ runtime libraries (XRT, xaiengine)

- **`aie_runtime_lib/`**: Device-side runtime for AIE cores (AIE, AIE2, AIE2P)

- **`test/`**: Test suite
  - `test/dialect/`: Per-dialect tests
  - `test/Conversion/`: Conversion pass tests
  - `test/npu-xrt/`: NPU XRT integration tests
  - `test/python/`: Python API tests

- **`programming_examples/`**: Example programs (basic, ml, vision)

- **`programming_guide/`**: Educational tutorials and guides

### MLIR Dialects

| Dialect | Purpose |
|---------|---------|
| **AIE** | Core AI Engine architecture: tiles, cores, memories, DMAs, locks |
| **AIEVec** | Vector operations optimized for AIE processing elements |
| **AIEX** | Extended operations: locks, flows, packet switching, ObjectFifos |
| **ADF** | Application Description Format: high-level dataflow graphs |
| **XLLVM** | AIE-specific LLVM IR extensions and intrinsics |

All dialects registered in: `include/aie/InitialAllDialect.h`

### Compilation Pipeline

```
Python/IRON API (python/iron/)
        ↓
MLIR AIE Dialect IR
        ↓
Transformation Passes (lib/Dialect/AIE/Transforms/):
  - AIEAssignLockIDs
  - AIEObjectFifoStatefulTransform
  - AIECreatePathFindFlows
  - AIEAssignBuffers
  - AIEGenerateColumnControlOverlay
  - Lower to Standard/LLVM dialects
        ↓
Code Generation (lib/Targets/):
  - NPU instruction generation (AIETargetNPU.cpp)
  - CDO generation (AIETargetCDODirect.cpp)
  - Backend-specific output
        ↓
Artifacts:
  - Core ELF binaries (compiled with xchesscc/Peano)
  - xclbin packages (for XRT runtime)
  - Configuration data objects
```

### Key Transformation Passes

Located in `lib/Dialect/AIE/Transforms/`:
- **AIEAssignLockIDs**: Allocates lock resources
- **AIEObjectFifoStatefulTransform**: Transforms ObjectFIFO abstractions to stateful DMAs
- **AIECreatePathFindFlows**: Performs path finding for routing
- **AIEAssignBuffers**: Memory allocation and address assignment
- **AIEFindFlows**: Flow analysis and validation
- **AIECanonicalizeDevice**: Device-specific canonicalization
- **AIEGenerateColumnControlOverlay**: Control flow code generation
- **AIELocalizeLocks**: Lock scope optimization
- **AIENormalizeAddressSpaces**: Address space normalization

### Code Generation Backends

Located in `lib/Targets/`:
- **AIETargetNPU.cpp**: XRT-based NPU instruction generation
- **AIETargetCDODirect.cpp**: Direct Configuration Data Object generation
- **AIETargetXAIEV2.cpp**: Xilinx AIEngine V2 specific codegen
- **AIETargetHSA.cpp**: HSA (Heterogeneous System Architecture) support
- **AIETargetBCF.cpp**: Board configuration files

## Working with MLIR Files

### Applying Transformation Passes

```bash
# Run specific pass on MLIR file
aie-opt --aie-assign-lock-ids input.mlir

# Chain multiple passes
aie-opt --aie-objectfifo-stateful-transform \
        --aie-create-pathfinder-flows \
        --aie-assign-buffer-addresses \
        input.mlir -o output.mlir

# Lower to LLVM dialect
aie-opt --convert-aievec-to-llvm input.mlir
```

### Translating MLIR to Other Formats

```bash
# Translate to LLVM IR
aie-translate --aie-to-llvmir input.mlir

# Generate NPU instruction sequence
aie-translate --aie-generate-npu input.mlir

# Generate CDO (Configuration Data Object)
aie-translate --aie-generate-cdo input.mlir
```

## Compiler Toolchain Components

### xchesscc vs Peano

Two options for compiling AIE core code:

- **xchesscc**: Proprietary Vitis AIE compiler (requires Vitis AIE Essentials license)
- **Peano**: Open-source LLVM-based compiler (llvm-aie repo)
  - Located at: `$PEANO_INSTALL_DIR/bin/`
  - Not added to PATH to avoid conflicts with system clang
  - Supports C/C++ compilation with AIE API header library

Set compiler via CMake: `-DAIE_COMPILER=PEANO` or `-DAIE_COMPILER=XCHESSCC`

### aiecc Compiler Driver

Two implementations exist:

1. **aiecc.py** (Python): `python/compiler/aiecc/main.py`
   - High-level orchestration
   - Invoked via Python API or command line

2. **aiecc** (C++): `tools/aiecc/aiecc.cpp`
   - New C++ implementation
   - Full compilation pipeline orchestration

## Python API (IRON)

High-level programming with `python/iron/`:

```python
from iron import Program, Worker, Kernel, Buffer, ObjectFifo, Runtime

# Create program
prog = Program()

# Define worker (AIE core)
worker = Worker(...)

# Define kernel
kernel = Kernel(...)

# Create buffers and ObjectFifos for data movement
buffer = Buffer(...)
fifo = ObjectFifo(...)

# Runtime execution
runtime = Runtime(prog)
runtime.run()
```

Key abstractions:
- **Program**: Top-level container
- **Worker**: Represents AIE core
- **Kernel**: Computation function
- **Buffer**: Memory allocation
- **ObjectFifo**: Data channel between cores/memories (automatically manages DMAs)
- **Runtime**: Execution and task management

## Common Development Workflows

### Adding a New Transformation Pass

1. Define pass in `include/aie/Dialect/AIE/Transforms/Passes.td`
2. Implement in `lib/Dialect/AIE/Transforms/AIEYourPass.cpp`
3. Register in `lib/Dialect/AIE/Transforms/Passes.cpp`
4. Add tests in `test/Passes/`
5. Update `tools/aie-opt/aie-opt.cpp` if needed

### Adding a New Dialect Operation

1. Define operation in TableGen: `include/aie/Dialect/*/IR/*.td`
2. Implement C++ methods if needed in `lib/Dialect/*/IR/`
3. Add verification logic
4. Add tests in `test/dialect/*/`
5. Update Python bindings if needed in `python/dialects/`

### Debugging Compilation Issues

```bash
# Enable verbose output in Makefile
make VERBOSE=1

# Debug MLIR transformations by examining intermediate IR
aie-opt --pass-pipeline='builtin.module(...)' input.mlir --debug

# Check generated instructions
aie-translate --aie-generate-npu input.mlir --mlir-print-ir-after-all
```

## Code Formatting

**IMPORTANT**: Always run formatting on modified files before committing. CI uses clang-format 17 and will fail if files are not properly formatted.

```bash
# Format C++ files (REQUIRED before commit)
clang-format -i <file.cpp>

# Format all modified C++ files at once
git diff --name-only --diff-filter=d | grep -E '\.(cpp|h)$' | xargs -r clang-format -i

# Format Python files and notebooks
black <file.py>
black <notebook.ipynb>

# Check formatting without modifying (useful for CI verification)
clang-format --dry-run -Werror <file.cpp>
```

Pre-commit hooks are installed via: `pre-commit install` (from requirements_dev.txt)

### Formatting Before Commit Checklist

1. Run `clang-format -i` on all modified C++ files (`.cpp`, `.h`)
2. Run `black` on all modified Python files (`.py`)
3. Verify with `git diff` that only intentional changes remain

## Important Notes

- **Environment setup is critical**: Always source ironenv, XRT setup.sh, and env_setup.sh before building
- **Peano vs xchesscc**: Peano is open-source but xchesscc may be required for certain features
- **NPU device detection**: Environment variable `NPU2` is set based on detected NPU (Strix, Strix Halo, Krackan)
- **Cross-compilation**: Support for x86_64 and aarch64 targets via `AIE_RUNTIME_TARGETS`
- **XRT required**: XRT (Xilinx Runtime) must be installed and configured for NPU execution
