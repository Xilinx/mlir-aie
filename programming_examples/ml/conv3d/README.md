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
- ✅ Tests passing for all depth values (1, 2, 4, 8, 16, 32)
- ✅ **True 3D convolution with 3×3×3 kernel**
- ✅ Sliding window implementation for depth dimension
- ⏱️ Execution time: ~15.6ms for 8×8×8 volume

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
