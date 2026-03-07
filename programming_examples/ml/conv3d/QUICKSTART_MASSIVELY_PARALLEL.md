# Quick Start: Massively Parallel Conv3D

This guide shows how to build and run the massively parallel Conv3D design on NPU2.

## Prerequisites

- NPU2 device (Strix/Strix Halo/Krackan)
- MLIR-AIE toolchain installed
- Vitis for xchesscc compiler
- Python environment with PyTorch

## Basic Usage

### 1. Build 8-Core Configuration (Default)

```bash
cd programming_examples/ml/conv3d

# Build with default configuration (8 cores, 64x64x8 volume)
make -f Makefile.massively_parallel all

# Or specify custom dimensions
make -f Makefile.massively_parallel all \
    N_CORES=8 \
    HEIGHT=64 \
    WIDTH=64 \
    DEPTH=8 \
    IN_CHANNELS=8 \
    OUT_CHANNELS=8
```

This generates:
- `conv3d_mp_8cores.mlir` - AIE design
- `conv3d_mp_8cores.xclbin` - NPU binary
- `conv3d_mp_8cores.insts.txt` - Runtime instructions

### 2. Run Test

```bash
make -f Makefile.massively_parallel run_py
```

Expected output:
```
Configuration: 8 cores, Volume: 8x64x64, Channels: 8→8
Avg NPU time: 1234us.

✓ PASS! (8 cores)
```

## Scaling to More Cores

### 16-Core Configuration

```bash
# Build for 16 cores (requires height divisible by 16)
make -f Makefile.massively_parallel all \
    N_CORES=16 \
    HEIGHT=128 \
    WIDTH=128 \
    DEPTH=16

# Run test
make -f Makefile.massively_parallel run_py \
    N_CORES=16 \
    HEIGHT=128 \
    WIDTH=128 \
    DEPTH=16
```

### 32-Core Configuration (Maximum)

```bash
# Build for 32 cores (full NPU2 device)
make -f Makefile.massively_parallel all \
    N_CORES=32 \
    HEIGHT=256 \
    WIDTH=256 \
    DEPTH=32 \
    IN_CHANNELS=32 \
    OUT_CHANNELS=32

# Run test
make -f Makefile.massively_parallel run_py \
    N_CORES=32 \
    HEIGHT=256 \
    WIDTH=256 \
    DEPTH=32 \
    IN_CHANNELS=32 \
    OUT_CHANNELS=32
```

## Pre-Configured Test Targets

### Test All Configurations

```bash
# Runs 8, 16, and 32 core tests sequentially
make -f Makefile.massively_parallel test_all
```

### Individual Test Targets

```bash
# Test 8 cores (64x64x8 volume)
make -f Makefile.massively_parallel test_8cores

# Test 16 cores (128x128x16 volume)
make -f Makefile.massively_parallel test_16cores

# Test 32 cores (256x128x16 volume)
make -f Makefile.massively_parallel test_32cores
```

## Benchmarking

Run larger workloads for performance measurement:

```bash
# Benchmark 8 cores with 16 channels
make -f Makefile.massively_parallel bench_8cores

# Benchmark 16 cores with 16 channels
make -f Makefile.massively_parallel bench_16cores

# Benchmark 32 cores with 32 channels (largest)
make -f Makefile.massively_parallel bench_32cores
```

## Configuration Rules

### Core Count
- **Valid values**: 1, 2, 4, 8, 16, 32
- **Device mapping**:
  - 1 core: NPU2Col1 (1 column)
  - 2 cores: NPU2Col2 (2 columns)
  - 4 cores: NPU2Col4 (4 columns)
  - 8 cores: NPU2 (8 columns × 1 row)
  - 16 cores: NPU2 (8 columns × 2 rows)
  - 32 cores: NPU2 (8 columns × 4 rows)

### Dimension Constraints

- **Height**: Must be divisible by `N_CORES`
  - Example: For 16 cores, height must be 16, 32, 48, 64, 80, 96, 112, 128, ...

- **Width**: Must be divisible by 8
  - Example: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, ...

- **Channels**: Input and output channels must be divisible by 8
  - Example: 8, 16, 24, 32, 40, 48, 56, 64, ...

- **Depth**: No constraints, but affects memory usage

### Example Valid Configurations

```bash
# Small test (1 core)
N_CORES=1 HEIGHT=8 WIDTH=8 DEPTH=4 IN_CHANNELS=8 OUT_CHANNELS=8

# Medium (8 cores)
N_CORES=8 HEIGHT=64 WIDTH=64 DEPTH=8 IN_CHANNELS=16 OUT_CHANNELS=16

# Large (16 cores)
N_CORES=16 HEIGHT=128 WIDTH=128 DEPTH=16 IN_CHANNELS=32 OUT_CHANNELS=32

# Maximum (32 cores)
N_CORES=32 HEIGHT=256 WIDTH=256 DEPTH=32 IN_CHANNELS=64 OUT_CHANNELS=64
```

## Direct Python Usage

You can also run the Python scripts directly:

### Generate MLIR

```bash
python3 conv3d_massively_parallel.py \
    --n_cores 16 \
    --depth 16 \
    --width 128 \
    --height 128 \
    --in_channels 16 \
    --out_channels 16 \
    > conv3d_mp_16cores.mlir
```

### Run Test

```bash
# After building xclbin and insts.txt
python3 test_massively_parallel.py \
    -x conv3d_mp_16cores.xclbin \
    -i conv3d_mp_16cores.insts.txt \
    -k MLIR_AIE \
    --n_cores 16 \
    --depth 16 \
    --width 128 \
    --height 128 \
    --in_channels 16 \
    --out_channels 16
```

## Performance Tips

### For Best Throughput

1. **Use all 8 columns**: Maximizes shim DMA parallelism
   - 8, 16, or 32 cores

2. **Balance height per core**: Aim for 8-16 rows per core
   - Too few rows → poor cache locality
   - Too many rows → memory pressure

3. **Increase channels**: More work per core
   - 16 or 32 channels better than 8

### Example High-Performance Configuration

```bash
make -f Makefile.massively_parallel all run_py \
    N_CORES=16 \
    HEIGHT=128 \
    WIDTH=128 \
    DEPTH=16 \
    IN_CHANNELS=32 \
    OUT_CHANNELS=32
```

This gives:
- 16 cores (8 columns × 2 rows)
- 8 rows per core (good locality)
- 128 × 32 = 4096 elements per row
- Full shim DMA parallelism

## Troubleshooting

### Error: "Height must be divisible by n_cores"

**Solution**: Adjust HEIGHT to be a multiple of N_CORES

```bash
# Wrong: HEIGHT=100, N_CORES=16 (100 % 16 = 4)
# Right: HEIGHT=96, N_CORES=16 (96 % 16 = 0)
```

### Error: "Width must be divisible by 8"

**Solution**: Use width = 8, 16, 24, 32, 40, 48, 56, 64, ...

```bash
# Wrong: WIDTH=50
# Right: WIDTH=48 or WIDTH=56
```

### Error: "Invalid configuration for device"

**Cause**: Requesting more cores than device supports

**Solution**: NPU2 has maximum 32 cores (8 columns × 4 rows)

### Build Fails: "xchesscc_wrapper not found"

**Solution**: Source Vitis environment

```bash
source /opt/xilinx/Vitis/2024.2/settings64.sh
```

### Test Fails: "Cannot open xclbin"

**Solution**: Ensure xclbin was built successfully

```bash
ls -lh conv3d_mp_*.xclbin
# Should show file size > 0
```

## Viewing Design Details

### MLIR Output

```bash
# View generated MLIR
cat conv3d_mp_8cores.mlir | less

# Search for specific patterns
grep "aie.tile" conv3d_mp_8cores.mlir
grep "aiex.npu.dma_memcpy_nd" conv3d_mp_8cores.mlir
```

### Resource Usage

```bash
# Count tiles used
grep "aie.tile" conv3d_mp_8cores.mlir | wc -l

# Count DMA operations
grep "aiex.npu.dma_memcpy_nd" conv3d_mp_8cores.mlir | wc -l
```

## Cleaning Up

```bash
# Remove all build artifacts
make -f Makefile.massively_parallel clean

# Remove logs
rm -rf log/
```

## Next Steps

- Read [MASSIVELY_PARALLEL_DESIGN.md](MASSIVELY_PARALLEL_DESIGN.md) for design details
- Adapt the pattern for your own application
- Experiment with different configurations
- Profile performance using NPU trace

## Support

For issues or questions:
1. Check [MASSIVELY_PARALLEL_DESIGN.md](MASSIVELY_PARALLEL_DESIGN.md) for design patterns
2. Validate configuration constraints
3. Review error messages carefully
4. Compare with working examples in test targets
