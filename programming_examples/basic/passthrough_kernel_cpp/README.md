# Passthrough Kernel with C++ Bindings

This example demonstrates the same functionality as `passthrough_kernel_placed.py` but uses C++ bindings for MLIR-AIE operations instead of Python bindings.

## Overview

The example creates a simple passthrough kernel that:
1. Takes input data from host memory
2. Processes it through an AIE compute tile (without modification)
3. Returns the output back to host memory

## Key Components

### C++ Generator (`passthrough_kernel_placed.cpp`)

This C++ program uses the MLIR C++ API to generate AIE configuration:

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
| **Language** | Python with IRON decorators | C++ with MLIR API |
| **API** | High-level Python bindings | Lower-level C++ API |
| **Generation** | Direct script execution | Compiled generator binary |
| **Expressiveness** | More concise syntax | More verbose, explicit |
| **Type Safety** | Runtime type checking | Compile-time type checking |

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
    ↓ (./passthrough_kernel_gen)
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

### C++ Version (Builder-based)
```cpp
auto deviceOp = builder.create<DeviceOp>(loc, ...);
auto coreOp = builder.create<CoreOp>(loc, computeTile2);
// Insert core operations into coreOp's block
```

The C++ version requires explicit:
- OpBuilder management
- Block creation and insertion point handling
- Location tracking
- Type construction

## Learning Value

This example demonstrates:
1. How to use MLIR C++ API for AIE programming
2. The relationship between high-level Python API and underlying MLIR operations
3. How to build custom AIE generators in C++
4. The MLIR generation and compilation pipeline

## Files

- `passthrough_kernel_placed.cpp` - C++ generator using MLIR API
- `CMakeLists.txt` - Build configuration for generator and test
- `Makefile` - Build orchestration
- `test.cpp` - Host application (reused from passthrough_kernel)
- `README.md` - This file

## Requirements

- MLIR/LLVM installation
- AIE dialect libraries
- Xilinx Vitis tools
- XRT (Xilinx Runtime)

## References

- Original Python version: `../passthrough_kernel/passthrough_kernel_placed.py`
- MLIR documentation: https://mlir.llvm.org/
- AIE dialect documentation: See programming guide
