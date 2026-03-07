# Quick Start: Conv3D Passthrough Test

## What is this?

A minimal passthrough test that just copies data through the NPU without doing any convolution. This isolates whether the NPU timeout issue is in the kernel logic or the basic data flow.

## Files

```
conv3d/
├── conv3d_passthrough.py          # IRON design (simplified)
├── test_passthrough.py             # Test script
├── Makefile.passthrough            # Build configuration
├── build_pt/                       # Build artifacts (created)
│   ├── passthrough_3d_ui8.o       # Compiled kernel
│   ├── aie2.mlir                  # Generated MLIR
│   ├── final.xclbin               # NPU binary
│   └── insts.bin                  # NPU instructions

aie_kernels/aie2/
└── passthrough_3d.cc               # Simple memcpy kernel
```

## Build (Already Done!)

The passthrough has already been built. Files are in `build_pt/`:
- Kernel: 856 bytes
- MLIR: 47 lines (2.5 KB)
- XCLBin: 9.0 KB
- Instructions: 300 bytes

To rebuild:
```bash
cd /scratch/jmelber/mlir-aie/programming_examples/ml/conv3d

# Set up environment
source ../../utils/env_setup.sh ../../ironenv

# Build everything
make -f Makefile.passthrough clean
make -f Makefile.passthrough
```

## Run Test

```bash
cd /scratch/jmelber/mlir-aie/programming_examples/ml/conv3d

# Run with default dimensions (4x4x4x8 = 512 bytes)
python3 test_passthrough.py -x build_pt/final.xclbin -i build_pt/insts.bin

# Expected output:
# === Passthrough 3D Test ===
# Depth: 4, Height: 4, Width: 4, Channels: 8
# Plane size: 128 bytes
# Total tensor size: 512 bytes
#
# Loading xclbin: build_pt/final.xclbin
# Loaded 300 bytes of instructions
#
# Input data range: 0 to 255
# First 16 values: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15]
#
# Running kernel...
# Kernel completed in XX.XX ms
#
# Output data range: 0 to 255
# First 16 values: [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15]
#
# *** SUCCESS: Output matches input perfectly! ***
# All 512 bytes passed through correctly
```

## Interpret Results

### ✅ SUCCESS (output matches input)
The basic NPU data flow works! Problem is in:
- Convolution kernel computation
- 3-plane sliding window
- Weights handling
- RTP/barrier synchronization

**Next steps**: Add complexity incrementally to find the issue.

### ❌ TIMEOUT or FAILURE
Basic data flow has issues. Problem might be:
- ObjectFifo configuration
- DMA setup
- Core sequencing
- NPU hardware/driver

**Next steps**: Simplify further or check hardware.

## Test with Different Dimensions

```bash
# Smaller test (2x2x2x8 = 64 bytes)
python3 test_passthrough.py -x build_pt/final.xclbin -i build_pt/insts.bin \
    -d 2 -ht 2 -wd 2 -c 8

# Larger test (8x8x8x8 = 4096 bytes) - requires rebuild
# Edit Makefile.passthrough: depth=8, height=8, width=8
source ../../utils/env_setup.sh ../../ironenv
make -f Makefile.passthrough clean
make -f Makefile.passthrough
python3 test_passthrough.py -x build_pt/final.xclbin -i build_pt/insts.bin \
    -d 8 -ht 8 -wd 8 -c 8
```

## What's Different from Full Conv3D?

| Feature | Full Conv3D | Passthrough |
|---------|-------------|-------------|
| **Kernel** | 3x3x3 convolution | memcpy |
| **Input planes** | 3 (sliding window) | 1 |
| **Weights** | Yes (3x3x3 kernel) | No |
| **RTP/Barriers** | Yes | No |
| **Kernel params** | 15 | 5 |
| **ObjectFifos** | 3 chains | 2 chains |
| **Buffer depth** | 3 (triple) | 2 (double) |
| **Code complexity** | 233 lines | 40 lines |
| **Test size** | 4096 bytes | 512 bytes |

## Troubleshooting

**"ModuleNotFoundError: No module named 'aie'"**
```bash
# Make sure to use the ironenv Python
source ../../utils/env_setup.sh ../../ironenv
```

**"Could not find 'aiecc' binary"**
```bash
# Environment not set up
source ../../utils/env_setup.sh ../../ironenv
which aiecc  # Should show: /scratch/jmelber/mlir-aie/ironenv/bin/aiecc
```

**NPU timeout or device not found**
```bash
# Check NPU is accessible
lspci | grep AMD
xbutil examine

# Check XRT version
xbutil --version
```

## Key Files to Examine

1. **Generated MLIR**: `build_pt/aie2.mlir`
   - Shows ObjectFifo configuration
   - Core function with acquire/release pattern
   - DMA configuration

2. **Kernel**: `/scratch/jmelber/mlir-aie/aie_kernels/aie2/passthrough_3d.cc`
   - Simple memcpy implementation
   - Good reference for minimal AIE kernel

3. **Design**: `conv3d_passthrough.py`
   - Clean IRON API example
   - Shows basic ObjectFifo usage
   - Simple worker setup

## More Information

- Detailed comparison: `passthrough_summary.md`
- Full documentation: `PASSTHROUGH_README.md`
- Original conv3d: `conv3d.py` and `aie_kernels/aie2/conv3dk3.cc`
