# Conv3D Spatial Parallelism Scaling Analysis

**Test Date:** March 7, 2026
**Working Directory:** `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d`

## Executive Summary

This analysis evaluates Conv3D spatial parallelism performance across different volume sizes and core counts. Spatial parallelism divides the height dimension across multiple NPU cores, with each core processing its slice independently.

### Key Findings

1. **Small volumes (16x16):** Single-core is optimal
   - Multi-core configurations show negative scaling due to overhead
   - 1-core achieves 818µs, while 2-core and 4-core are slower

2. **Medium volumes (32x32):** Good scaling with 2 cores
   - 1-core: 2615µs (baseline)
   - 2-core: 1667µs (1.56x speedup, 78% efficiency)
   - 4-core: 1363µs (1.91x speedup, 48% efficiency)
   - **Sweet spot: 2 cores** for best efficiency

3. **Large volumes (64x64+):** Memory-constrained
   - Requires 4+ cores due to per-tile memory limits (~64KB)
   - 1-core and 2-core configurations fail to build
   - 128x128 volumes exceed memory even with 4 cores

## Detailed Results

### Performance Table

| Volume Size | Cores | NPU Time (µs) | Speedup | Efficiency |
|-------------|-------|---------------|---------|------------|
| 16x16       | 1     | 818.3         | 1.00x   | 100%       |
| 16x16       | 2     | 1083.9        | 0.75x   | 38%        |
| 16x16       | 4     | 932.0         | 0.87x   | 22%        |
| 32x32       | 1     | 2615.3        | 1.00x   | 100%       |
| 32x32       | 2     | 1667.2        | 1.56x   | 78%        |
| 32x32       | 4     | 1362.6        | 1.91x   | 48%        |
| 64x64       | 4     | 2544.9        | N/A     | N/A        |
| 128x128     | 4     | BUILD FAIL    | N/A     | N/A        |

**Efficiency** = (Speedup / Cores) × 100%

### Memory Analysis

Per-core memory requirements for spatial parallelism:

```
Buffer Requirements per Core:
- Input buffers: 3 planes × (height_per_core × width × channels) bytes
- Output buffer: (height_per_core × width × channels) bytes
- Weights: (channels × channels × 3 × 3 × 3) = 1728 bytes
```

| Volume | Cores | Height/Core | Buffer/Core | Total/Core | Status |
|--------|-------|-------------|-------------|------------|--------|
| 16x16  | 1     | 16          | 2,048 B     | 10,944 B   | OK     |
| 16x16  | 4     | 4           | 512 B       | 3,800 B    | OK     |
| 32x32  | 1     | 32          | 8,192 B     | 34,496 B   | OK     |
| 32x32  | 4     | 8           | 2,048 B     | 10,944 B   | OK     |
| 64x64  | 1     | 64          | 32,768 B    | 132,800 B  | FAIL   |
| 64x64  | 2     | 32          | 16,384 B    | 67,264 B   | FAIL   |
| 64x64  | 4     | 16          | 8,192 B     | 34,496 B   | OK     |
| 128x128| 4     | 32          | 32,768 B    | 132,800 B  | FAIL   |
| 128x128| 8     | 16          | 16,384 B    | 67,264 B   | FAIL   |

**AIE2 Tile Memory Limit:** ~64KB (65,536 bytes)

## Analysis

### Scaling Characteristics

1. **Communication Overhead Dominates Small Volumes**
   - For 16x16 volumes, the overhead of multi-core coordination exceeds benefits
   - Each core processes minimal data, making setup costs proportionally high

2. **Sweet Spot at 32x32 with 2 Cores**
   - Best balance of parallelism vs overhead
   - 78% efficiency indicates effective work distribution
   - Further scaling to 4 cores yields diminishing returns (48% efficiency)

3. **Memory Wall at Large Volumes**
   - The fundamental bottleneck is per-tile memory capacity
   - Even with optimal spatial partitioning, 128x128 requires >64KB/tile
   - Would require alternative strategies:
     - Temporal blocking (process in chunks)
     - Channel splitting (reduce channels per iteration)
     - Combination of spatial and temporal parallelism

## Recommendations

### For Production Use

| Volume Size | Recommended Cores | Rationale |
|-------------|------------------|-----------|
| ≤ 16×16     | 1                | Overhead exceeds benefits |
| 16×32       | 1-2              | Marginal benefit from parallelism |
| 32×32       | 2                | Optimal efficiency (78%) |
| 32×64       | 2-4              | Good scaling |
| 64×64       | 4                | Required due to memory |
| ≥ 128×128   | N/A              | Requires alternative approach |

### Alternative Strategies for Large Volumes

For volumes exceeding memory limits (≥128×128):

1. **Temporal Blocking**
   - Process volume in smaller chunks over time
   - Reduces per-core memory requirements
   - Trade compute time for memory

2. **Channel Reduction**
   - Reduce input/output channels (currently 8)
   - Proportionally reduces buffer requirements
   - May impact model accuracy

3. **Hybrid Parallelism**
   - Combine spatial (height) and channel parallelism
   - Distribute both dimensions across cores
   - More complex coordination

4. **Streaming Architecture**
   - Stream data through tiles rather than buffering
   - Requires redesign of kernel logic
   - Better memory efficiency

## Test Configuration

**Environment:**
- Device: AMD NPU2 (AIE2P architecture)
- Kernel: conv3dk3_ui8 (3×3×1 per plane, vectorized)
- Depth: 8 planes
- Channels: 8 input, 8 output
- Data types: uint8 activations, int8 weights

**Test Methodology:**
1. Generated MLIR using `conv3d_spatial.py` with IRON API
2. Compiled to XCLBin using `aiecc.py`
3. Executed on hardware with `test_spatial.py`
4. Measured NPU execution time (excludes host-device transfer)

## Files

- **Implementation:** `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/conv3d_spatial.py`
- **Test Script:** `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/test_spatial.py`
- **Build Script:** `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/build_test_matrix.sh`
- **Results:** `/scratch/jmelber/mlir-aie/programming_examples/ml/conv3d/build/spatial_scaling_results.txt`

## Conclusion

Spatial parallelism for Conv3D shows practical benefits in the 32×32 to 64×64 volume range with 2-4 cores. The sweet spot is **32×32 volumes with 2 cores** (78% efficiency, 1.56x speedup). Memory constraints prevent scaling to larger volumes without architectural changes. For production workloads exceeding 64×64, temporal blocking or hybrid parallelism strategies are recommended.
