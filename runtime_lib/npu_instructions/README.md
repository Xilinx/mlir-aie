# NPU Instruction Encoding Library

A standalone, header-only C++ library for encoding NPU transaction instructions with **zero dependencies** on LLVM or MLIR.

## Overview

This library provides the core instruction encoding functions used by both:
- **MLIR compiler** (`lib/Targets/AIETargetNPU.cpp`) - compile-time transaction generation
- **Generated runtime code** (`aie-translate --aie-generate-txn-cpp`) - runtime transaction generation
- **Host applications** - direct transaction manipulation

## Design Principles

1. **Header-only** - No build dependencies, just `#include "npu_instructions/npu_instructions.h"`
2. **Zero external dependencies** - Only standard C++ (`<cstdint>`, `<vector>`)
3. **No MLIR/LLVM** - Can be used in any C++ project
4. **Single source of truth** - Eliminates code duplication between compiler and runtime

## Usage

```cpp
#include "npu_instructions/npu_instructions.h"
#include <vector>
#include <cstdint>

std::vector<uint32_t> generate_transaction() {
    std::vector<uint32_t> txn;

    // Append instructions
    aie::npu::appendWrite32(txn, 0x1D000000, 0x12345678);
    aie::npu::appendMaskWrite32(txn, 0x1D000004, 0xABCD, 0xFFFF);
    aie::npu::appendSync(txn, 0, 0, 0, 0, 1, 1);

    // Add transaction header (must be last)
    aie::npu::prependHeader(txn);

    return txn;
}
```

## API Reference

### Instruction Encoding Functions

#### `appendWrite32(instructions, address, value)`
Writes a 32-bit value to a 32-bit address.
- **Size:** 6 words (24 bytes)
- **Opcode:** `WRITE` (0)

#### `appendMaskWrite32(instructions, address, value, mask)`
Writes a 32-bit value with mask to an address.
- **Size:** 7 words (28 bytes)
- **Opcode:** `MASKWRITE` (3)

#### `appendSync(instructions, column, row, direction, channel, columnNum, rowNum)`
Waits for a task completion token from specified tile/channel.
- **Size:** 4 words (16 bytes)
- **Opcode:** `CUSTOM_OP_TCT` (128)

#### `appendBlockWrite(instructions, address, data, dataSize)`
Writes a block of data to specified address.
- **Size:** 4 + dataSize words
- **Opcode:** `BLOCKWRITE` (1)

#### `appendLoadPdi(instructions, id, size, address)`
Loads a PDI (Programmable Device Image).
- **Size:** 4 words (16 bytes)
- **Opcode:** `LOADPDI` (8)

#### `appendAddressPatch(instructions, patchAddr, argIdx, argPlusOffset)`
Patches an address at runtime (for dynamic buffer addresses).
- **Size:** 12 words (48 bytes)
- **Opcode:** `CUSTOM_OP_DDR_PATCH` (129)

#### `appendPreempt(instructions, level)`
Preemption control.
- **Size:** 1 word (4 bytes)
- **Opcode:** `PREEMPT` (6)

#### `prependHeader(instructions, numRows, numCols, devGen, numMemTileRows)`
Prepends transaction header. **Must be called after all instructions are appended.**
- **Size:** 4 words (16 bytes)
- **Parameters:**
  - `numRows`: Number of rows in device (default: 6)
  - `numCols`: Number of columns (default: 5)
  - `devGen`: Device generation (3=NPU1, 4=NPU2, default: 4)
  - `numMemTileRows`: Number of memory tile rows (default: 1)

## Transaction Format

```
┌─────────────────────────────────────┐
│ Header (4 words)                    │
│  [0]: (numRows<<24)|(devGen<<16)|v  │
│  [1]: (numMemTileRows<<8)|numCols   │
│  [2]: operation count               │
│  [3]: total size in bytes           │
├─────────────────────────────────────┤
│ Instruction 1 (variable size)       │
├─────────────────────────────────────┤
│ Instruction 2 (variable size)       │
├─────────────────────────────────────┤
│ ...                                 │
└─────────────────────────────────────┘
```

## Examples

### Standalone Usage (No MLIR)

See `test/npu-xrt/dynamic_passthrough/test_runtime_generation.cpp`:

```cpp
std::vector<uint32_t> generate_passthrough_txn(size_t num_lines) {
    std::vector<uint32_t> txn;

    for (size_t line = 0; line < num_lines; ++line) {
        uint32_t offset = line * 64; // 64 bytes per line

        // Configure DMA for this line
        aie::npu::appendWrite32(txn, 0x1D000000, offset);
        aie::npu::appendWrite32(txn, 0x1D000004, 64);
        aie::npu::appendWrite32(txn, 0x1D000008, 0x80000000);

        // Wait for completion
        aie::npu::appendSync(txn, 0, 0, 0, 0, 1, 1);
    }

    aie::npu::prependHeader(txn);
    return txn;
}
```

### Generated from MLIR

See `test/npu-xrt/dynamic_passthrough/test_full_workflow.mlir` and the generated C++ code.

MLIR dynamic sequences automatically generate C++ code using this library via `aie-translate --aie-generate-txn-cpp`.

## Integration with MLIR

### In the Compiler

`lib/Targets/AIETargetNPU.cpp` uses this library for compile-time transaction generation:

```cpp
#include "npu_instructions/npu_instructions.h"

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {
    aie::npu::appendWrite32(instructions, *op.getAbsoluteAddress(), op.getValue());
}
```

### In Generated Code

Dynamic MLIR sequences generate C++ code that uses this library:

```mlir
aie.runtime_sequence @my_seq(%size: index) {
    %c100 = arith.constant 100 : i32
    aiex.npu.dyn_write32(%c100, %size) : i32, index
}
```

Generates:

```cpp
#include "npu_instructions/npu_instructions.h"  // Not yet, but will

std::vector<uint32_t> generate_txn_my_seq(size_t size) {
    // ... uses append functions ...
}
```

## Testing

### Standalone Library Test ✅

**Location:** `test/npu-xrt/dynamic_passthrough/test_runtime_generation.cpp`

This test demonstrates using the library without any MLIR dependencies to generate runtime transactions for variable line counts.

**Run:**
```bash
cd test/npu-xrt/dynamic_passthrough
c++ -std=c++17 -I../../../runtime_lib test_runtime_generation.cpp -o test
./test
```

**Expected output:**
```
Transaction for 1 line: 26 words
Transaction for 8 lines: 180 words
Transaction for 32 lines: 708 words
Expected size for 32 lines: 708
Actual size for 32 lines: 708
All tests passed!

Transaction structure:
  Header: 4 words (16 bytes)
  Per-line operations: ~22 words
  Scaling: Linear with num_lines
```

### Full Workflow Test ✅

**Location:** `test/npu-xrt/dynamic_passthrough/test_full_workflow.mlir` + `test_generated_wrapper.cpp`

This test demonstrates the complete MLIR → C++ → Runtime pipeline:

**Run:**
```bash
cd test/npu-xrt/dynamic_passthrough
# Generate C++ from MLIR
aie-translate --aie-generate-txn-cpp test_full_workflow.mlir -o generated.cpp
# Compile and run
c++ -std=c++17 generated.cpp test_generated_wrapper.cpp -o test
./test
```

**Expected output:**
```
Small transaction (4 writes): 28 words
Medium transaction (16 writes): 100 words
Large transaction (64 writes): 388 words

Runtime transaction generation test: PASSED
Successfully generated transactions for different problem sizes!
```

**What this proves:**
- Single MLIR compilation supports multiple runtime sizes
- Generated C++ code is valid and efficient
- No recompilation needed for different problem dimensions

## Performance Characteristics

- **Instruction overhead:** ~6 words per write32 operation
- **Scaling:** Linear with number of operations
- **Header overhead:** Fixed 4 words (16 bytes)
- **Memory:** Pre-allocate with `txn.reserve()` for efficiency

## Device Compatibility

- **NPU1** (Phoenix, Hawk): `devGen = 3`
- **NPU2** (Strix, Kraken): `devGen = 4` (default)

## Future Enhancements

- Add BD (Buffer Descriptor) configuration helpers
- Add packet header encoding
- Add shimDMA configuration
- Optimize header operation count calculation
