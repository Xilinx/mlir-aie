# Conv3D Passthrough Test

This directory contains a minimal passthrough test to isolate NPU timeout issues in the conv3d implementation.

## Purpose

The passthrough test simplifies the full conv3d pipeline by:
- Using a single input plane instead of 3-plane sliding window
- Removing all convolution computation (just memcpy)
- Minimal ObjectFifo usage (depth=2 double buffering instead of depth=3)
- No weights buffer needed
- No runtime parameters (RTP) or barriers

This helps isolate whether issues are in:
- The kernel logic
- The ObjectFifo configuration
- The sliding window pattern
- Basic data flow through the NPU

## Files Created

1. **Kernel**: `/scratch/jmelber/mlir-aie/aie_kernels/aie2/passthrough_3d.cc`
   - Simple memcpy from input plane to output plane
   - No computation, just data movement

2. **IRON Design**: `conv3d_passthrough.py`
   - Minimal ObjectFifo setup
   - Simple loop over depth slices
   - No weights, RTP, or barriers

3. **Makefile**: `Makefile.passthrough`
   - Builds the passthrough kernel and xclbin
   - Uses smaller test dimensions (4x4x4x8)

4. **Test Script**: `test_passthrough.py`
   - Verifies output matches input
   - Shows timing and diagnostic information

## Building

```bash
# From the conv3d directory
cd /scratch/jmelber/mlir-aie/programming_examples/ml/conv3d

# Build everything (requires environment)
source ../../utils/env_setup.sh ../../ironenv
make -f Makefile.passthrough

# Or build steps individually:
make -f Makefile.passthrough build_pt/passthrough_3d_ui8.o  # Compile kernel
make -f Makefile.passthrough build_pt/aie2.mlir              # Generate MLIR
make -f Makefile.passthrough build_pt/final.xclbin           # Build xclbin
```

## Running

```bash
# Test with default dimensions (4x4x4x8)
python3 test_passthrough.py -x build_pt/final.xclbin -i build_pt/insts.bin

# Test with custom dimensions
python3 test_passthrough.py -x build_pt/final.xclbin -i build_pt/insts.bin \
    -d 8 -ht 8 -wd 8 -c 8
```

## Generated MLIR

The passthrough design generates much simpler MLIR than the full conv3d:
- Single ObjectFifo chain: L3 → L2 → L1 → L2 → L3
- Simple acquire/release pattern (no sliding window complexity)
- Straightforward DMA configuration
- ~47 lines of MLIR vs. more complex full version

## Test Dimensions

Default parameters (can be changed in Makefile.passthrough):
- Depth: 4 planes
- Height: 4 pixels
- Width: 4 pixels
- Channels: 8
- Total size: 4×4×4×8 = 512 bytes

These minimal dimensions help isolate issues while keeping compile/run times fast.

## Expected Behavior

If the passthrough test **succeeds**:
- All output bytes match input bytes exactly
- This confirms basic ObjectFifo and DMA functionality works
- Issue is likely in the convolution kernel logic or 3-plane sliding window

If the passthrough test **fails** or **times out**:
- Issue is in the basic data flow infrastructure
- Could be ObjectFifo configuration, DMA setup, or core sequencing
- Further simplification may be needed

## Troubleshooting

If you see timeout errors:
1. Check NPU is accessible: `lspci | grep AMD`
2. Verify XRT is installed: `xbutil examine`
3. Try smaller dimensions (2x2x2x8)
4. Check kernel logs in build_pt/ directory

## Next Steps

After the passthrough test:
1. If it works: Add back complexity incrementally (weights, then computation, then 3-plane window)
2. If it fails: Simplify further or check NPU hardware/driver setup
