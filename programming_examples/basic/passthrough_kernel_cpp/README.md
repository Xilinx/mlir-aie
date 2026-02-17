# Passthrough Kernel with C++ Bindings

This example demonstrates the same functionality as `passthrough_kernel_placed.py` but uses C++ bindings for MLIR-AIE operations instead of Python bindings.

## Overview

The example creates a simple passthrough kernel that:
1. Takes input data from host memory
2. Processes it through an AIE compute tile (without modification)
3. Returns the output back to host memory

## Key Components

### C++ Generator (`passthrough_kernel_placed.cpp`)

This C++ program uses the MLIR C++ API to programmatically generate AIE configuration:

- **Device Setup**: Creates an AIE device configuration using `DeviceOp`
- **Tile Declarations**: Defines Shim Tile (0,0) and Compute Tile (0,2) using `TileOp::create`
- **ObjectFIFOs**: Sets up data movement channels using `ObjectFifoCreateOp`
- **Core Logic**: Implements the passthrough operation with an external kernel using `CoreOp`
- **Runtime Sequence**: Defines the DMA operations for data transfer using `NpuDmaMemcpyNdOp`

The generator uses OpBuilder and the MLIR C++ API to construct the IR programmatically, then outputs MLIR IR to stdout for compilation by aiecc.py.

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
| **API** | High-level Python bindings | MLIR C++ OpBuilder API |
| **Generation** | Direct script execution | Compiled generator binary |
| **Expressiveness** | More concise syntax | More verbose, explicit |
| **Type Safety** | Runtime type checking | Compile-time type checking |
| **Dependencies** | Python + aie-python-extras | C++ compiler + MLIR + AIE libraries |

## Building

```bash
make
```

This will:
1. Build the C++ generator (`passthrough_kernel_gen`) using MLIR C++ API
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
    ↓ (cmake + make with MLIR/AIE libs)
passthrough_kernel_gen (executable)
    ↓ (./passthrough_kernel_gen -d npu -i1s 4096 -os 4096)
aie2_lineBased_8b_4096.mlir (MLIR IR)
    ↓ (aiecc.py)
final_4096.xclbin (device binary)
    ↓ (with test.cpp)
passthrough_kernel_cpp.exe (host app)
```

## Key Implementation Details

### Python Version (Decorator-based)
```python
@device(dev)
def device_body():
    @core(ComputeTile2, "passThrough.cc.o")
    def core_body():
        # Core logic
```

### C++ Version (OpBuilder-based)
```cpp
auto deviceOp = builder.create<DeviceOp>(loc, AIEDeviceAttr::get(ctx, device));
auto computeTile2 = TileOp::create(builder, loc, 0, 2);
auto coreOp = CoreOp::create(builder, loc, indexType, computeTile2);
// Insert core operations into coreOp's block
```

The C++ version uses the MLIR OpBuilder pattern:
- Explicit OpBuilder management for insertion points
- Block creation and manipulation
- Location tracking for diagnostics
- Type construction using MLIR type system
- Operation creation using builder.create<> or Op::create() patterns

## Learning Value

This example demonstrates:
1. How to use MLIR C++ API for AIE programming
2. The relationship between high-level Python API and underlying MLIR C++ operations
3. How to build custom AIE generators using C++ bindings
4. The MLIR generation and compilation pipeline
5. OpBuilder patterns for creating MLIR operations programmatically

## Files

- `passthrough_kernel_placed.cpp` - C++ generator using MLIR C++ API
- `CMakeLists.txt` - Build configuration for generator and test
- `Makefile` - Build orchestration
- `test.cpp` - Host application (reused from passthrough_kernel with fixes)
- `README.md` - This file

## Requirements

- C++ compiler (g++ or clang++)
- MLIR/LLVM installation
- AIE dialect libraries
- Xilinx Vitis tools
- XRT (Xilinx Runtime)

## References

- Original Python version: `../passthrough_kernel/passthrough_kernel_placed.py`
- MLIR documentation: https://mlir.llvm.org/
- AIE dialect documentation: See programming guide
