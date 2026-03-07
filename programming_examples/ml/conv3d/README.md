# Conv3D Example

3D Convolution for NPU using vectorized AIE intrinsics.

## Build and Run

```bash
source ../../../ironenv/bin/activate
make
make run_py
```

## Performance

**Single NPU Core (vectorized):**
- 8×8×8 volume: **~500µs**
- 30× faster than scalar implementation
- Utilizes AIE2P matrix multiply units

## Features

- ✅ True 3D convolution (3×3×3 kernel)
- ✅ Sliding window over depth dimension
- ✅ Vectorized using `aie::mmul<4,8,8,uint8,int8>`
- ✅ All depth values supported (1, 2, 4, 8, 16, 32)
- ✅ Production-ready performance

## Testing

```bash
# Default: 8x8x8 volume
python3 test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE

# Custom sizes
python3 test.py -x build/final.xclbin -i build/insts.bin -k MLIR_AIE \
    -d 16 -ht 8 -wd 8 -ic 8 -oc 8
```

## Multi-Core Scaling

Multi-core implementation is WIP. Challenge: IRON API doesn't properly compile complex if/else logic inside loops for sliding window management. Potential solutions:
1. Move sliding window logic into C++ kernel (accept depth_index parameter)
2. Simplify to 2D convolution per plane for multi-core
3. Implement in pure MLIR instead of IRON Python API

## Implementation

- **Kernel**: `aie_kernels/aie2p/conv3dk3.cc` (vectorized)
- **Design**: `conv3d.py` (IRON Worker API)
- **Test**: `test.py` (PyTorch validation)
