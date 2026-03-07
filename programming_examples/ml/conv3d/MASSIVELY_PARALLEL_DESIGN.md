# Massively Parallel Conv3D Design Pattern

## Overview

The `conv3d_massively_parallel.py` design demonstrates a highly scalable Conv3D implementation that can utilize 1 to 32 AI Engine cores with parallel shim DMA channels. This document describes the "stampable block" design pattern used to achieve this scalability.

## Design Pattern: Stampable Blocks

### Concept

A **stampable block** is a repeatable unit of computation and data movement that can be replicated across the device. Instead of writing custom logic for different core counts, we define a basic pattern and "stamp" it out multiple times.

### Key Principles

1. **Column-based organization**: Each column has its own shim tile for independent DMA
2. **Vertical stacking**: Multiple cores within a column share the same shim
3. **Spatial parallelism**: Work is divided by spatial dimension (height)
4. **Simple core function**: No conditionals - same code runs on all cores

### Architecture

```
NPU2 Device (8 columns × 4 compute rows)

Column:     0        1        2        3        4        5        6        7
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Shim (0) │ DMA │ DMA │ DMA │ DMA │ DMA │ DMA │ DMA │ DMA │  ← 8 parallel DMA channels
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Mem  (1) │ Buf │ Buf │ Buf │ Buf │ Buf │ Buf │ Buf │ Buf │  ← Memory tiles
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Core (2) │  C0 │  C1 │  C2 │  C3 │  C4 │  C5 │  C6 │  C7 │  ← First row of cores
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Core (3) │  C8 │  C9 │ C10 │ C11 │ C12 │ C13 │ C14 │ C15 │  ← Second row of cores
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Core (4) │ C16 │ C17 │ C18 │ C19 │ C20 │ C21 │ C22 │ C23 │  ← Third row of cores
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Core (5) │ C24 │ C25 │ C26 │ C27 │ C28 │ C29 │ C30 │ C31 │  ← Fourth row of cores
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

## Supported Configurations

### 8 Cores (8 columns × 1 row)
- **Device**: NPU2 (full width, single compute row)
- **Pattern**: 8 independent blocks (1 core each)
- **Shim DMA**: 8 input + 8 output channels (maximum parallelism)
- **Height split**: Each core processes height/8 rows

```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│  ← 8 parallel DMAs
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C0 │  C1 │  C2 │  C3 │  C4 │  C5 │  C6 │  C7 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### 16 Cores (8 columns × 2 rows)
- **Device**: NPU2 (full width, two compute rows)
- **Pattern**: 8 blocks of 2 cores each
- **Shim DMA**: 8 input + 8 output channels
- **Height split**: Each core processes height/16 rows

```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│  ← 8 parallel DMAs
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C0 │  C2 │  C4 │  C6 │  C8 │ C10 │ C12 │ C14 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C1 │  C3 │  C5 │  C7 │  C9 │ C11 │ C13 │ C15 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### 32 Cores (8 columns × 4 rows)
- **Device**: NPU2 (full device)
- **Pattern**: 8 blocks of 4 cores each
- **Shim DMA**: 8 input + 8 output channels
- **Height split**: Each core processes height/32 rows

```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│ Shim│  ← 8 parallel DMAs
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C0 │  C4 │  C8 │ C12 │ C16 │ C20 │ C24 │ C28 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C1 │  C5 │  C9 │ C13 │ C17 │ C21 │ C25 │ C29 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C2 │  C6 │ C10 │ C14 │ C18 │ C22 │ C26 │ C30 │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  C3 │  C7 │ C11 │ C15 │ C19 │ C23 │ C27 │ C31 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

### Other Supported Configurations
- **1 Core**: NPU2Col1 (single column)
- **2 Cores**: NPU2Col2 (2 columns × 1 row)
- **4 Cores**: NPU2Col4 (4 columns × 1 row)

## Data Distribution Strategy

### Spatial Parallelism

The design uses **spatial parallelism** across the height dimension:

1. **Input splitting**: Each core receives `height/n_cores` contiguous rows
2. **Weight broadcasting**: All cores receive the same weights
3. **Output concatenation**: Outputs are reassembled in height order

### TensorAccessPattern

The `TensorAccessPattern` class defines the data movement for each core:

```python
# For core_id processing rows [start_row, end_row)
offset = core_id * (depth * height_per_core * width * in_channels)

in_tap = TensorAccessPattern(
    tensor_dims=(1, tensorInSize),
    offset=offset,
    sizes=[1, 1, 1, actIn_per_core * depth],
    strides=[0, 0, 0, 1]
)
```

This creates a **contiguous slice** of the input tensor for each core.

### Parallel Shim DMA

Each column uses its own shim tile (row 0) for DMA:

```python
# Fill inputs using column-specific shims
for col in range(n_cols):
    for row in range(n_rows_per_col):
        rt.fill(
            of_in_fifos[col][row].prod(),
            I,
            in_taps[core_id],
            placement=Tile(col, 0)  # Shim at column 'col'
        )
```

**Benefits**:
- Up to 8 parallel DMA channels (8 shim tiles)
- Each shim handles data for its column's cores
- Avoids DMA bottlenecks on single shim

## Core Function Design

### Simplicity Principle

The core function has **no conditionals** - identical code runs on all cores:

```python
def core_fn(of_wts, of_in, of_out, kernel):
    elemWts = of_wts.acquire(1)

    for d in range_(depth):
        plane = of_in.acquire(1)
        elemOut = of_out.acquire(1)

        kernel(
            plane, plane, plane,
            elemWts, elemOut,
            width, height_per_core, in_channels, out_channels,
            3, 3, 1,  # 2D conv per plane
            1, 10, 0
        )

        of_in.release(1)
        of_out.release(1)

    of_wts.release(1)
```

**Why this matters**:
- **Easier debugging**: Same code path for all cores
- **Better performance**: No branch mispredictions
- **Scalability**: Add more cores without changing logic

### 2D Convolution Per Plane

Each core applies 2D convolution on each depth plane:
- Kernel size: 3×3×1 (not 3×3×3)
- Processes `depth` planes sequentially
- Each plane: `height_per_core × width × channels`

This approach:
- Simplifies data dependencies
- Reduces memory footprint per core
- Still achieves 3D effect across depth dimension

## Performance Characteristics

### Expected Speedup

Theoretical speedup is **linear with core count** (up to memory bandwidth limits):

| Cores | Columns | Shim DMAs | Expected Speedup | Use Case |
|-------|---------|-----------|------------------|----------|
| 1     | 1       | 1+1       | 1× (baseline)    | Testing |
| 2     | 2       | 2+2       | ~2×              | Small models |
| 4     | 4       | 4+4       | ~4×              | Medium models |
| 8     | 8       | 8+8       | ~7-8×            | Large models |
| 16    | 8       | 8+8       | ~12-15×          | Very large models |
| 32    | 8       | 8+8       | ~20-28×          | Maximum throughput |

### Bottleneck Analysis

1. **Cores 1-8**: Compute bound, linear scaling
2. **Cores 16**: DMA bandwidth starts to saturate (8 shims shared)
3. **Cores 32**: Memory bandwidth saturated, sub-linear scaling

### Optimal Configurations

**For large volumes (128×128×64)**:
- **16 cores**: Best balance of throughput and efficiency
- Height per core: 8 rows (good cache locality)
- All 8 shims fully utilized

**For huge volumes (256×256×128)**:
- **32 cores**: Maximum throughput
- Height per core: 8 rows
- Saturates device bandwidth

## Usage Examples

### 8-Core Configuration (64×64×8 volume)

```bash
python conv3d_massively_parallel.py \
    --n_cores 8 \
    --depth 8 \
    --width 64 \
    --height 64 \
    --in_channels 8 \
    --out_channels 8 \
    > aie2.mlir
```

Each core processes: 8 rows × 64 cols × 8 channels = 4096 elements per plane

### 16-Core Configuration (128×128×16 volume)

```bash
python conv3d_massively_parallel.py \
    --n_cores 16 \
    --depth 16 \
    --width 128 \
    --height 128 \
    --in_channels 16 \
    --out_channels 16 \
    > aie2.mlir
```

Each core processes: 8 rows × 128 cols × 16 channels = 16384 elements per plane

### 32-Core Configuration (256×256×32 volume)

```bash
python conv3d_massively_parallel.py \
    --n_cores 32 \
    --depth 32 \
    --width 256 \
    --height 256 \
    --in_channels 32 \
    --out_channels 32 \
    > aie2.mlir
```

Each core processes: 8 rows × 256 cols × 32 channels = 65536 elements per plane

## Design Patterns for Other Applications

### When to Use Stampable Blocks

✅ **Good fit**:
- Spatial parallelism (split by dimension)
- Independent computation per block
- Regular data distribution
- Symmetric workloads

❌ **Poor fit**:
- Irregular data dependencies
- Dynamic workload distribution
- Asymmetric computation patterns
- Data-dependent control flow

### Adapting the Pattern

To adapt this pattern for other applications:

1. **Choose parallelism dimension**: Which dimension to split (height, width, channels, etc.)
2. **Define block size**: How much work per core (balance compute vs data movement)
3. **Map to columns**: How many cores per column (maximize shim DMA parallelism)
4. **Create access patterns**: Use TensorAccessPattern for clean slicing
5. **Keep core function simple**: No conditionals if possible

### Example: Matrix Multiply

```python
# Split output matrix by rows across columns
for col in range(n_cols):
    for row in range(n_rows_per_col):
        # Each core computes rows [start:end] of output
        core_id = col * n_rows_per_col + row
        rows_per_core = M // n_cores
        start_row = core_id * rows_per_core

        # Place at Tile(col, 2+row) - same column-based pattern
        worker = Worker(matmul_fn, ..., placement=Tile(col, 2+row))
```

## Implementation Checklist

When implementing a stampable block design:

- [ ] Choose total core count (power of 2, up to 32)
- [ ] Determine column/row layout (maximize shim usage)
- [ ] Calculate per-core data sizes (fit in L1 memory)
- [ ] Define TensorAccessPatterns for data distribution
- [ ] Create simple core function (no conditionals)
- [ ] Use column-based shim placement (parallel DMA)
- [ ] Test with 1, 2, 4 cores before scaling to 16/32

## Summary

The stampable block pattern provides:

1. **Predictable scaling**: 1 → 2 → 4 → 8 → 16 → 32 cores
2. **Maximum DMA parallelism**: Up to 8 shim tiles
3. **Simple implementation**: Column-major organization
4. **Clean data distribution**: TensorAccessPattern slicing
5. **No conditionals**: Same code on all cores

This pattern is ideal for spatial parallelism in image processing, video processing, and volumetric computations on the AI Engine array.
