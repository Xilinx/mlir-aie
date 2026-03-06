# 3D Convolution (Conv3D)

This example demonstrates 3D convolution on AMD AI Engine NPUs using the IRON Worker API.

## Overview

3D convolution extends 2D convolution by adding a depth dimension, making it suitable for:
- 3D medical imaging (CT scans, MRI volumes)
- Video processing (temporal convolution across frames)
- Volumetric data analysis

This implementation:
- **Kernel size**: 3×3×3 (27 multiply-accumulate operations per output)
- **Data types**: uint8 activations, int8 weights, uint8 output
- **Padding**: 1 (maintains spatial dimensions)
- **Border handling**: Replicates edge planes/pixels

## Implementation

### Phase 1: Single-Core Scalar (Current)

Single compute core processes the entire 3D volume sequentially:
- **Input**: D×H×W×C (depth, height, width, channels)
- **Weights**: 3×3×3×C_in×C_out
- **Output**: D×H×W×C_out

**Data layout**:
- Input/Output: `D{C/8}H{C8}W` (channel-groups of 8)
- Weights: `{O/8}{I/8}KDHW{I8}{O8}` (optimized for AIE vector units)

### Future Phases

- **Phase 2**: Vectorized single-core (AIE MMUL intrinsics, ~10× speedup)
- **Phase 3**: Multi-core parallel (4-core, then 32-core)
- **Phase 4**: Production testing and benchmarking

## Building and Running

### Prerequisites

```bash
source <path-to-mlir-aie>/utils/env_setup.sh install
```

### Build

```bash
cd programming_examples/ml/conv3d
make
```

### Run on NPU

```bash
make run_py depth=8 height=8 width=8 in_channels=8 out_channels=8
```

### Run with Tracing

```bash
make trace_py trace_size=16384
```

### Test Different Sizes

```bash
# Tiny (for debugging)
make run_py depth=4 height=4 width=4 in_channels=8 out_channels=8

# Small (baseline)
make run_py depth=8 height=8 width=8 in_channels=8 out_channels=8

# Medium
make run_py depth=16 height=16 width=16 in_channels=16 out_channels=16
```

## Files

- `conv3d.py`: IRON Python design (single Worker, ObjectFifos, Runtime)
- `test.py`: Host test program with PyTorch reference
- `aie_kernels/aie2/conv3dk3.cc`: Scalar 3D convolution kernel
- `aie_kernels/aie2/conv3dk3.h`: Kernel header
- `Makefile`: Build system
- `CMakeLists.txt`: CMake integration

## Validation

Output is validated against PyTorch `nn.Conv3d` with:
- Tolerance: `2 × int8_scale` (0.0156)
- Quantization-aware comparison

## Performance

Current (scalar): ~20ms for 8×8×8×8 volume

Target (vectorized multi-core): ~1ms for 64×64×64×512 volume

## Design Pattern

This example demonstrates:
- **Worker API**: Task definition with `core_fn` and `Worker`
- **ObjectFifos**: L3→L2→L1 data flow with automatic DMA
- **Runtime parameters**: Dynamic scale factor via RTP buffer
- **Border handling**: Top/middle/bottom plane logic for 3D edges

## Next Steps

1. Profile scalar performance with tracing
2. Implement vectorized kernel using AIE MMUL intrinsics
3. Scale to multi-core parallel design
4. Benchmark peak throughput (volumes/sec)
