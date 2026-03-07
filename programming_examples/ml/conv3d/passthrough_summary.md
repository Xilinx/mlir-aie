# Conv3D Passthrough Test - Summary

## Overview

Created a minimal passthrough test to isolate the NPU timeout issue in conv3d by removing all computation and simplifying data flow.

## Files Created

### 1. Kernel: `passthrough_3d.cc`
**Location**: `/scratch/jmelber/mlir-aie/aie_kernels/aie2/passthrough_3d.cc`

**Key differences from conv3dk3.cc**:
- Single input plane (not 3 planes for sliding window)
- Simple memcpy (no convolution loops)
- No weights parameter
- No border handling logic
- ~40 lines vs. ~233 lines

```cpp
void passthrough_3d_ui8(uint8_t *input_plane, uint8_t *output_plane,
                        const int32_t input_width, const int32_t input_height,
                        const int32_t input_channels) {
  event0();
  int32_t plane_size = input_height * input_width * input_channels;
  memcpy(output_plane, input_plane, plane_size);
  event1();
}
```

### 2. IRON Design: `conv3d_passthrough.py`
**Location**: `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/conv3d_passthrough.py`

**Key differences from conv3d.py**:
- No weights ObjectFifo
- No RTP (runtime parameters) buffer
- No WorkerRuntimeBarrier
- Simple depth=2 buffering (not depth=3 for triple buffering)
- Single input plane per iteration (not 3-plane window)
- 5 kernel parameters vs. 15
- ~140 lines vs. ~238 lines

**ObjectFifo structure**:
```
Original:
  Input:   L3 → L2 (depth=3) → L1
  Weights: L3 → L2 (depth=1) → L1
  Output:  L1 → L2 → L3 (depth=2)

Passthrough:
  Input:   L3 → L2 (depth=2) → L1
  Output:  L1 → L2 (depth=2) → L3
```

**Core function simplification**:
```python
# Original: Complex 3-plane sliding window
for z in range_(8):
    elem_act = of_act.acquire(1)  # Acquires from 3-plane window
    elemOut = of_out.acquire(1)
    conv3dk3(elem_act, elem_act, elem_act, elemWts, elemOut, ...)  # 3 planes + weights
    of_act.release(1)
    of_out.release(1)

# Passthrough: Simple single-plane copy
for z in range_(depth):
    elem_in = of_in.acquire(1)   # Single plane
    elem_out = of_out.acquire(1)
    passthrough_fn(elem_in, elem_out, ...)  # Just copy
    of_in.release(1)
    of_out.release(1)
```

### 3. Build Configuration: `Makefile.passthrough`
**Location**: `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/Makefile.passthrough`

**Key changes**:
- Smaller default dimensions: 4×4×4×8 (512 bytes) vs. 8×8×8×8 (4096 bytes)
- Separate build directory: `build_pt/` to avoid conflicts
- Uses ironenv Python path explicitly
- No vectorization flags (scalar only)

### 4. Test Script: `test_passthrough.py`
**Location**: `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/test_passthrough.py`

**Features**:
- Simple pass/fail validation (output == input)
- Timing information
- Clear diagnostic output
- Configurable dimensions via command line
- Shows first mismatches if validation fails

## Build Status

Successfully built:
- ✅ Kernel object: `build_pt/passthrough_3d_ui8.o` (856 bytes)
- ✅ MLIR design: `build_pt/aie2.mlir` (47 lines, 2.5 KB)
- ✅ XCLBin: `build_pt/final.xclbin` (9.0 KB)
- ✅ Instructions: `build_pt/insts.bin` (300 bytes)

## Generated MLIR Analysis

The passthrough generates clean, minimal MLIR:
- Single tile used: `%tile_0_2` (AIE core)
- Memory tile: `%mem_tile_0_1`
- Shim tile: `%shim_noc_tile_0_0`
- Two ObjectFifo chains (input and output)
- Simple DMA configuration (single dimension, no complex strides)
- Core function with straightforward acquire/process/release loop

## Complexity Comparison

| Aspect | Full Conv3D | Passthrough | Reduction |
|--------|-------------|-------------|-----------|
| Kernel LOC | 233 | 40 | 83% fewer |
| Design LOC | 238 | 140 | 41% fewer |
| Kernel params | 15 | 5 | 67% fewer |
| Input planes | 3 | 1 | 67% fewer |
| ObjectFifos | 3 chains | 2 chains | 33% fewer |
| Buffer depth | 3 (triple) | 2 (double) | 33% less |
| Data size | 4096 bytes | 512 bytes | 87% smaller |
| Runtime deps | RTP + barrier | None | 100% fewer |

## Testing Strategy

### Step 1: Run Passthrough
```bash
cd /scratch/jmelber/mlir-aie/programming_examples/ml/conv3d
python3 test_passthrough.py -x build_pt/final.xclbin -i build_pt/insts.bin
```

### Step 2: Interpretation

**If passthrough SUCCEEDS**:
- Basic NPU, DMA, and ObjectFifo infrastructure works
- Problem is likely in:
  - Convolution kernel logic
  - 3-plane sliding window management
  - Weights handling
  - RTP/barrier synchronization

**If passthrough FAILS/TIMES OUT**:
- Issue is in fundamental data flow
- Further simplification needed:
  - Try even smaller dimensions (2×2×2×8)
  - Try single depth plane (depth=1)
  - Check NPU hardware/driver

### Step 3: Incremental Complexity

After passthrough works, add back features one at a time:
1. Add weights buffer (no computation)
2. Add RTP and barrier (still no computation)
3. Add simple computation (1×1×1 kernel)
4. Add 3×3×3 convolution
5. Add 3-plane sliding window

## Environment Requirements

```bash
# Set up environment
source /scratch/jmelber/mlir-aie/utils/env_setup.sh /scratch/jmelber/mlir-aie/ironenv

# Required tools (should be in PATH after env_setup):
# - aiecc.py (compiler driver)
# - aiecc (C++ compiler)
# - PEANO_INSTALL_DIR (llvm-aie toolchain)
```

## Key Insights

1. **Simplified Data Flow**: The passthrough eliminates the complex 3-plane sliding window, making it easier to debug basic ObjectFifo behavior.

2. **No Computation**: Using memcpy removes any risk of kernel computation errors or timing issues.

3. **Minimal Dependencies**: No weights, RTP, or barriers reduces synchronization complexity.

4. **Small Test Size**: 512 bytes vs. 4KB makes it faster to compile and easier to inspect data.

5. **Clear Pass/Fail**: Simple validation (input == output) gives unambiguous results.

## Next Actions

1. Run the passthrough test on NPU hardware
2. Observe if it succeeds or times out
3. Use results to determine next debugging steps
4. Document findings to guide conv3d fixes
