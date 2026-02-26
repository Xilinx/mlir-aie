# Dynamic Runtime Sequences with SSA Values

## Context

Currently, runtime sequence operations in MLIR-AIE have **static values baked into the IR** at compile time. Python programming examples use Python to templatize and generate different runtime sequences for various problem sizes, but this requires recompilation for each configuration.

**Goal:** Enable runtime sequences that accept SSA values instead of static attributes, allowing:
1. Use of standard dialects (SCF) for loops and control flow within runtime sequences
2. Lowering to **templated C++ code** that can be compiled as a `.so` library
3. Problem sizes determined at **runtime** in host code, not compile time
4. Single compiled artifact supporting multiple configurations

---

## Current State Analysis

### Existing Runtime Sequence Operations

Located in `include/aie/Dialect/AIEX/IR/AIEX.td`:

| Operation | Static (Attributes) | Dynamic (SSA) |
|-----------|---------------------|---------------|
| `npu.dma_memcpy_nd` | `static_offsets`, `static_sizes`, `static_strides`, `id`, `metadata` | `memref`, variadic offsets/sizes/strides |
| `npu.write32` | `address`, `value`, `column`, `row` | None |
| `npu.maskwrite32` | `address`, `value`, `mask`, coords | None |
| `npu.blockwrite` | `address`, coords | `memref` (data) |
| `npu.sync` | `column`, `row`, `direction`, `channel` | None |
| `npu.push_queue` | All parameters | None |
| `dma_configure_task` | `direction`, `channel`, `issue_token` | `tile` |
| `dma_start_task` | None | `task` (SSA reference) |

**Key Observation:** The DMA task API already uses SSA values for task references, but most low-level NPU operations use static attributes exclusively.

### Translation Constraints

From `lib/Targets/AIETargetNPU.cpp`:

The NPU instruction format requires:
- 32-bit addresses with tile coordinates encoded in upper bits
- Fixed instruction word layouts (6-12 words per instruction)
- Transaction header with operation count and size

Current translation assumes **all values are compile-time constants** because it directly encodes them into instruction words.

### Python Templatization Pattern

From `python/iron/`:

```python
# Sizes computed from input tensors at JIT compile time
M, K, N = input0.shape[0], input0.shape[1], input1.shape[1]
M_div_m = M // m  # Baked into generated IR

# TAPs (TensorAccessPatterns) encode static strides
a_taps = TensorTiler2D.simple_tiler(
    tensor_dims=(M, K),  # Static values
    tile_dims=(m, k)
)
```

---

## Proposed Architecture

### Level 1: SSA-Enabled Operations

Extend existing ops to accept SSA values for key parameters:

```tablegen
// New: Dynamic write operation
def AIEX_NpuDynWrite32Op : AIEX_Op<"npu.dyn_write32", [
    HasParent<"RuntimeSequenceOp">
]> {
    let arguments = (ins
        AnySignlessInteger:$address,  // SSA value
        AnySignlessInteger:$value,    // SSA value
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
    );
}

// New: Dynamic DMA configuration
def AIEX_NpuDynDmaMemcpyNdOp : AIEX_Op<"npu.dyn_dma_memcpy_nd", [
    HasParent<"RuntimeSequenceOp">,
    AttrSizedOperandSegments
]> {
    let arguments = (ins
        AnyMemRef:$memref,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,      // All SSA values
        Variadic<Index>:$strides,
        SymbolRefAttr:$metadata,
        I32Attr:$id
    );
}
```

### Level 2: SCF Integration

Allow standard control flow in runtime sequences:

```mlir
aie.runtime_sequence(%buf: memref<?xi32>, %size: index, %iterations: index) {
    // SCF for loop with dynamic bounds
    scf.for %i = %c0 to %iterations step %c1 {
        %offset = arith.muli %i, %tile_size : index
        aiex.npu.dyn_dma_memcpy_nd(%buf, %offset, %size, %c1)
    }
}
```

### Level 3: C++ Template Lowering

New translation pass: `--aie-generate-txn-cpp`

**Output:** C++ source file with template functions:

```cpp
// Generated from MLIR runtime sequence
#include <vector>
#include <cstdint>

namespace aie_runtime {

template<typename SizeT>
std::vector<uint32_t> generate_txn_my_sequence(
    uint64_t buf_addr,
    SizeT size,
    SizeT iterations
) {
    std::vector<uint32_t> txn;
    txn.reserve(estimated_size);

    // Transaction header
    append_header(txn, /*num_ops=*/iterations * 3);

    // Unrolled/dynamic loop
    for (SizeT i = 0; i < iterations; ++i) {
        SizeT offset = i * TILE_SIZE;
        append_dma_memcpy_nd(txn, buf_addr + offset, size, /*stride=*/1);
    }

    return txn;
}

} // namespace aie_runtime
```

### Level 4: Shared Library Interface

Build system integration to compile generated C++ to `.so`:

```cpp
// Public API in generated header
extern "C" {
    // Returns transaction buffer, caller owns memory
    uint32_t* aie_gen_my_sequence(
        uint64_t buf_addr,
        size_t size,
        size_t iterations,
        size_t* out_txn_size
    );

    void aie_free_txn(uint32_t* txn);
}
```

---

## Implementation Plan

### Step 1: Define Dynamic Operations

**Files:**
- `include/aie/Dialect/AIEX/IR/AIEX.td`

Add SSA-enabled variants:
- `aiex.npu.dyn_write32` - address and value as SSA
- `aiex.npu.dyn_memcpy_nd` - sizes/strides as SSA
- `aiex.npu.dyn_sync` - channel/direction as SSA

Keep static ops for backwards compatibility; new ops are opt-in.

### Step 2: Add Verification and Canonicalization

**Files:**
- `lib/Dialect/AIEX/IR/AIEXDialect.cpp`
- `lib/Dialect/AIEX/Transforms/`

Verifiers ensure:
- SSA values have appropriate types (i32, i64, index)
- Required static attributes still present
- Parent is RuntimeSequenceOp

Canonicalization:
- Fold constant SSA values back to attributes where possible
- Combine consecutive operations where safe

### Step 3: Create C++ Code Generation Backend

**New Files:**
- `lib/Targets/AIETargetCppTxn.cpp`
- `include/aie/Targets/AIETargets.h` (update)

Translation logic:
1. Walk RuntimeSequenceOp body
2. For each operation, emit C++ code:
   - Static values → literals
   - SSA values → template parameters or function arguments
   - SCF loops → C++ for loops
3. Generate instruction encoding as inline functions
4. Emit transaction header computation

### Step 4: Add aie-translate Registration

**Files:**
- `lib/Targets/AIETargets.cpp`

```cpp
TranslateFromMLIRRegistration registrationToCppTxn(
    "aie-generate-txn-cpp",
    "Generate C++ transaction sequence code",
    [](ModuleOp module, raw_ostream &output) {
        return AIETranslateToCppTxn(module, output);
    },
    [](DialectRegistry &registry) {
        registry.insert<AIE::AIEDialect, AIEX::AIEXDialect>();
    });
```

### Step 5: Implement SCF Loop Handling

**Files:**
- `lib/Targets/AIETargetCppTxn.cpp`

For `scf.for`:
```cpp
void emitScfFor(scf::ForOp forOp, raw_ostream &os) {
    os << "for (auto " << getVarName(forOp.getInductionVar())
       << " = " << emitValue(forOp.getLowerBound())
       << "; " << getVarName(forOp.getInductionVar())
       << " < " << emitValue(forOp.getUpperBound())
       << "; " << getVarName(forOp.getInductionVar())
       << " += " << emitValue(forOp.getStep()) << ") {\n";
    // Emit body
    os << "}\n";
}
```

### Step 6: Add CMake Integration for .so Generation

**New Files:**
- `cmake/modules/AIECppTxn.cmake`

```cmake
function(aie_generate_txn_library)
    cmake_parse_arguments(ARG "" "NAME;MLIR_SOURCE" "" ${ARGN})

    # Generate C++ from MLIR
    add_custom_command(
        OUTPUT ${ARG_NAME}_txn.cpp ${ARG_NAME}_txn.h
        COMMAND aie-translate --aie-generate-txn-cpp
                ${ARG_MLIR_SOURCE} -o ${ARG_NAME}_txn.cpp
        DEPENDS ${ARG_MLIR_SOURCE}
    )

    # Build shared library
    add_library(${ARG_NAME}_txn SHARED ${ARG_NAME}_txn.cpp)
endfunction()
```

### Step 7: Update IRON API for Dynamic Sequences

**Files:**
- `python/iron/runtime/runtime.py`

```python
class Runtime:
    def sequence(self, *arg_types, dynamic_sizes=False):
        """
        If dynamic_sizes=True, sizes are emitted as SSA values
        rather than static attributes.
        """
```

---

## Key Files Summary

| File | Changes |
|------|---------|
| `include/aie/Dialect/AIEX/IR/AIEX.td` | Add `npu.dyn_*` operations |
| `lib/Dialect/AIEX/IR/AIEXDialect.cpp` | Verification, builders |
| `lib/Targets/AIETargetCppTxn.cpp` | **NEW** - C++ codegen |
| `lib/Targets/AIETargets.cpp` | Register translation |
| `include/aie/Targets/AIETargets.h` | API declaration |
| `cmake/modules/AIECppTxn.cmake` | **NEW** - Build helpers |
| `python/iron/runtime/runtime.py` | Dynamic mode support |

---

## Verification

### Unit Tests

```bash
# Test dynamic operations parse correctly
lit test/dialect/AIEX/dynamic_ops.mlir

# Test C++ code generation
lit test/Targets/generate_cpp_txn.mlir
```

### Integration Test

```bash
# 1. Generate C++ from test MLIR
aie-translate --aie-generate-txn-cpp \
    test/Targets/dynamic_sequence.mlir \
    -o test_txn.cpp

# 2. Compile to shared library
clang++ -shared -fPIC test_txn.cpp -o libtest_txn.so

# 3. Run test program that calls the library
./test_dynamic_txn

# 4. Compare generated transaction bytes with expected
diff <(xxd generated.bin) <(xxd expected.bin)
```

### Performance Test

```bash
# Compare: static compilation vs dynamic library
# Should show minimal overhead from dynamic generation
python benchmarks/compare_static_dynamic.py
```

---

## Design Considerations

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **SSA ops + C++ backend** | Full flexibility, standard dialects work, single compilation | More complex implementation, runtime dependency |
| **Parameterized binary format** | Simpler, no runtime codegen | Limited flexibility, custom format |
| **JIT compilation** | Maximum flexibility | Requires LLVM JIT at runtime, heavy dependency |

**Recommendation:** SSA ops + C++ backend provides the best balance of flexibility and simplicity.

### Scope Decision

**All parameters** will be made dynamic:
- Sizes and strides (tensor dimensions, access patterns)
- Offsets (starting positions)
- Loop bounds (iteration counts via SCF)
- Addresses (already partially supported via address patching)

### Open Questions

1. **Loop unrolling:** Should SCF loops always be preserved in C++, or optionally unrolled?
2. **Template vs function args:** Which parameters should be templates vs runtime arguments?
3. **Error handling:** How to handle invalid parameter combinations at runtime?
4. **Memory management:** Who allocates/frees the transaction buffer?

---

## Dependencies

- No new external dependencies
- Requires C++17 for generated code (std::vector, templates)
- Optional: XRT for runtime execution

---

## Example End-to-End Flow

### 1. MLIR Source with Dynamic Values

```mlir
module {
  aie.device(npu1_4col) {
    // ... tile and buffer definitions ...

    aie.runtime_sequence(%buf: memref<?xi32>, %M: index, %N: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %tile_size = arith.constant 1024 : index

      // Dynamic iteration count
      %tiles = arith.divui %M, %tile_size : index

      scf.for %i = %c0 to %tiles step %c1 {
        %offset = arith.muli %i, %tile_size : index
        aiex.npu.dyn_dma_memcpy_nd(%buf, %offset, %tile_size, %c1)
        aiex.npu.dyn_sync(%c0, %c0)
      }
    }
  }
}
```

### 2. Generated C++ Code

```cpp
#include "aie_txn_runtime.h"

namespace aie_runtime {

std::vector<uint32_t> generate_sequence(
    uint64_t buf_addr,
    size_t M,
    size_t N
) {
    constexpr size_t tile_size = 1024;
    size_t tiles = M / tile_size;

    std::vector<uint32_t> txn;

    for (size_t i = 0; i < tiles; ++i) {
        size_t offset = i * tile_size;
        append_dma_memcpy_nd(txn, buf_addr + offset * sizeof(int32_t),
                             tile_size, 1);
        append_sync(txn, 0, 0);
    }

    prepend_header(txn);
    return txn;
}

} // namespace aie_runtime
```

### 3. Host Application Usage

```cpp
#include "my_design_txn.h"
#include <xrt/xrt_bo.h>

int main() {
    // Problem size determined at runtime
    size_t M = get_user_input();
    size_t N = get_user_input();

    // Allocate buffer
    auto bo = xrt::bo(device, M * N * sizeof(int32_t), ...);

    // Generate transaction for this specific size
    auto txn = aie_runtime::generate_sequence(bo.address(), M, N);

    // Submit to device
    kernel(bo, txn.data(), txn.size() * sizeof(uint32_t));

    return 0;
}
```

## Comments from your boss:

this should include factoring out all the code in AIETargetNPU.cpp into its own stand-alone library that can be used both from the generated .so as well as from the existing MLIR. This is where the actual instruction sequence encoding is captured. AI loves to duplicate code, and I'd like to keep those encodings central in one spot rather than duplicating. So make AIETargetNPU.cpp independent of LLVM/MLIR, then reference it where needed from both compiler and runtime code. I've only skimmed the document, assume it was written by AI, but if it's intended as instructions to AI, I'd also include an instruction that it should use the emitC dialect/pass to generate C code from the scf constructs.
