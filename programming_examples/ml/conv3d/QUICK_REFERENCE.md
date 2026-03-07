# Conv3D Spatial Parallelism Quick Reference

## Test Results Summary

| Volume  | 1-Core    | 2-Core    | 4-Core    | Winner      |
|---------|-----------|-----------|-----------|-------------|
| 16×16   | 818 µs ✓  | 1084 µs   | 932 µs    | **1-core**  |
| 32×32   | 2615 µs   | 1667 µs ✓ | 1363 µs   | **2-core**  |
| 64×64   | MEM FAIL  | MEM FAIL  | 2545 µs ✓ | **4-core**  |
| 128×128 | MEM FAIL  | MEM FAIL  | MEM FAIL  | N/A         |

## Optimal Configuration

**32×32 volumes with 2 cores**
- Speedup: 1.56x
- Efficiency: 78%
- NPU Time: 1667 µs

## Quick Guidelines

```
if volume <= 16×16:
    use 1 core           # Overhead dominates
elif volume == 32×32:
    use 2 cores          # Sweet spot (78% efficiency)
elif volume == 64×64:
    use 4 cores          # Only option (memory constraint)
else:  # volume >= 128×128
    redesign required    # Temporal blocking or hybrid parallelism
```

## Build & Run

```bash
# Build a configuration
python3 conv3d_spatial.py npu2_2col 8 32 32 8 8 > build/test.mlir
cd build && aiecc.py --aie-generate-xclbin --aie-generate-npu-insts \
    --no-compile-host --no-xchesscc --no-xbridge \
    --xclbin-name=test.xclbin --npu-insts-name=test_insts.bin test.mlir

# Test it
python3 test_spatial.py build/test.xclbin build/test_insts.bin 2 32 32
```

## Device Options

- `npu2` → 1 core (NPU2Col1)
- `npu2_2col` → 2 cores (NPU2Col2)
- `npu2_4col` → 4 cores (NPU2Col4)

## Files

- **Implementation:** `conv3d_spatial.py`
- **Test script:** `test_spatial.py`
- **Full report:** `build/SUMMARY.txt`
- **Detailed analysis:** `SPATIAL_SCALING_ANALYSIS.md`
