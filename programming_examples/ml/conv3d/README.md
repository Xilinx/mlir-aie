# Conv3D Example

3D Convolution implementation for NPU using IRON Worker API.

## Build and Run

```bash
source ../../../ironenv/bin/activate
make clean
make build/final.xclbin
make run_py
```

## Current Status

- ✅ Working end-to-end on NPU
- ✅ Tests passing for 8×8×8 and 16×8×8 volumes
- ⚠️ Currently uses 2D convolution (3×3×1) per depth plane
- 🚧 TODO: Implement true 3D sliding window (3×3×3)

## Parameters

Default: `depth=8, height=8, width=8, in_channels=8, out_channels=8`

Custom test:
```bash
python3 test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE \
    -d 16 -ht 8 -wd 8 -ic 8 -oc 8
```

## Architecture

- **Kernel**: AIE2/AIE2P scalar implementation (aie_kernels/aie2p/conv3dk3.cc)
- **Design**: IRON Worker API (conv3d.py)
- **Test**: PyTorch reference validation (test.py)
