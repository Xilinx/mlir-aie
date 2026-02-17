# Passthrough Kernel with C++ Bindings

This example demonstrates the same functionality as `passthrough_kernel_placed.py` but uses C++ to generate MLIR-AIE operations instead of Python bindings.

## Overview

The example creates a simple passthrough kernel that:
1. Takes input data from host memory
2. Processes it through an AIE compute tile (without modification)
3. Returns the output back to host memory

## Key Components

### C++ Generator (`passthrough_kernel_placed.cpp`)

This C++ program generates AIE configuration in MLIR textual format:

- **Device Setup**: Creates an AIE device configuration
- **Tile Declarations**: Defines Shim Tile (0,0) and Compute Tile (0,2)
- **ObjectFIFOs**: Sets up data movement channels between tiles
- **Core Logic**: Implements the passthrough operation with an external kernel
- **Runtime Sequence**: Defines the DMA operations for data transfer

The generator outputs MLIR IR to stdout, which is then compiled by aiecc.py to generate the device binary.

### Kernel (`passThrough.cc`)

The actual compute kernel (reused from `../../../aie_kernels/generic/passThrough.cc`) that performs the passthrough operation on the AIE tile.

### Host Test (`test.cpp`)

Host code that:
- Initializes the AIE device
- Allocates input/output buffers
- Transfers data to/from the device
- Verifies the results

## Differences from Python Version

| Aspect | Python Version | C++ Version |
|--------|---------------|-------------|
| **Language** | Python with IRON decorators | C++ with MLIR text generation |
| **API** | High-level Python bindings | Direct MLIR text output |
| **Generation** | Direct script execution | Compiled generator binary |
| **Expressiveness** | More concise syntax | More explicit control |
| **Type Safety** | Runtime type checking | Compile-time type checking |
| **Dependencies** | Python + aie-python-extras | C++ compiler + LLVM Support |

## Building

```bash
make
```

This will:
1. Build the C++ generator (`passthrough_kernel_gen`)
2. Run the generator to create MLIR IR
3. Compile the kernel with Peano
4. Generate the device binary with aiecc.py
5. Build the host test executable

## Running

```bash
make run
```

## Build Flow

```
passthrough_kernel_placed.cpp (C++)
    ↓ (cmake + make)
passthrough_kernel_gen (executable)
    ↓ (./passthrough_kernel_gen -d npu -i1s 4096 -os 4096)
aie2_lineBased_8b_4096.mlir (MLIR IR)
    ↓ (aiecc.py)
final_4096.xclbin (device binary)
    ↓ (with test.cpp)
passthrough_kernel_cpp.exe (host app)
```

## Key Differences in Implementation

### Python Version (Decorator-based)
```python
@device(dev)
def device_body():
    @core(ComputeTile2, "passThrough.cc.o")
    def core_body():
        # Core logic
```

### C++ Version (Text Generation)
```cpp
std::cout << "module {\n";
std::cout << "  aie.device(" << deviceEnum << ") {\n";
std::cout << "    %tile_0_2 = aie.tile(0, 2)\n";
std::cout << "    %core_0_2 = aie.core(%tile_0_2) {\n";
// Core logic
std::cout << "    } {link_with = \"passThrough.cc.o\"}\n";
```

The C++ version directly generates the MLIR textual representation, which is:
- Simpler to implement without MLIR C++ API or LLVM dependencies
- Easier to understand and maintain
- Provides complete control over the generated IR
- Requires only standard C++ (no external libraries)

## Learning Value

This example demonstrates:
1. How to generate MLIR-AIE IR programmatically using C++
2. The textual format of MLIR-AIE operations
3. The relationship between high-level Python API and underlying MLIR operations
4. How to build custom AIE generators in C++
5. An alternative approach to the MLIR C++ API for code generation

## Files

- `passthrough_kernel_placed.cpp` - C++ generator that outputs MLIR text
- `CMakeLists.txt` - Build configuration for generator and test
- `Makefile` - Build orchestration
- `test.cpp` - Host application (reused from passthrough_kernel)
- `README.md` - This file

## Requirements

- C++ compiler (g++ or clang++)
- Xilinx Vitis tools
- XRT (Xilinx Runtime)

## References

- Original Python version: `../passthrough_kernel/passthrough_kernel_placed.py`
- MLIR documentation: https://mlir.llvm.org/
- AIE dialect documentation: See programming guide
