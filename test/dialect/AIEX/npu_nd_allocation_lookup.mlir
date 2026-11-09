//===- npu_nd_allocation_lookup.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s -o /dev/null

// Allocation resolution is device-local, independent of declaration order,
// and uses only the root of metadata, including on the runtime-operand path.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, 4, 16][0, 256, 32, 1]) {id = 0 : i64, metadata = @dma::@unused} : memref<4096xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, %n, 16][0, 256, 32, 1]) {id = 1 : i64, metadata = @dma::@unused} : memref<4096xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @dma(%tile, MM2S, 0)
  }
}

// -----

// An unresolved static allocation is allowed before allocation lowering.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 1, 1, 2][1572864, 0, 0, 1]) {id = 0 : i64, metadata = @missing} : memref<4096xi32>
    }
  }
}

// -----

// A symbol of another kind is also unresolved for the static tile checks.
module {
  aie.device(npu1) {
    aie.runtime_sequence @not_alloc() {}
    aie.runtime_sequence @copy(%arg0: memref<4096xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 1, 1, 2][1572864, 0, 0, 1]) {id = 0 : i64, metadata = @not_alloc} : memref<4096xi32>
    }
  }
}

// -----

// The same oversized stride must not be deferred once the allocation exists.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      // expected-error @+1 {{Stride 3 is 1572864 (encoded as 1572863), which exceeds the [1:1048576] range.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 1, 1, 2][1572864, 0, 0, 1]) {id = 0 : i64, metadata = @dma} : memref<4096xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @dma(%tile, MM2S, 0)
  }
}

// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4096xi32>, %n: i64) {
      // expected-error @+1 {{runtime sizes/strides require a shim_dma_allocation to resolve the tile; none found.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, %n, 16][0, 256, 32, 1]) {id = 0 : i64, metadata = @missing} : memref<4096xi32>
    }
  }
}

// -----

// A collection must not reuse a same-named allocation from a sibling device.
module {
  aie.device(npu1) @first {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @dma(%tile, MM2S, 0)
    aie.runtime_sequence @copy(%arg0: memref<4096xi32>, %n: i64) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, %n, 16][0, 256, 32, 1]) {id = 0 : i64, metadata = @dma} : memref<4096xi32>
    }
  }
  aie.device(npu1) @second {
    aie.runtime_sequence @dma() {}
    aie.runtime_sequence @copy(%arg0: memref<4096xi32>, %n: i64) {
      // expected-error @+1 {{runtime sizes/strides require a shim_dma_allocation to resolve the tile; none found.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 2, %n, 16][0, 256, 32, 1]) {id = 0 : i64, metadata = @dma} : memref<4096xi32>
    }
  }
}

// -----

// Constants carried by SSA operands still take the static, deferred path.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4096xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0, %c0, %c0, %c0][%c1, %c1, %c1, %c16][%c0, %c0, %c0, %c1]) {id = 0 : i64, metadata = @missing} : memref<4096xi32>
    }
  }
}

// -----

// A runtime offset alone requires an allocation, just like a runtime size.
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4096xi32>, %offset: i64) {
      // expected-error @+1 {{runtime sizes/strides require a shim_dma_allocation to resolve the tile; none found.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, %offset][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @missing} : memref<4096xi32>
    }
  }
}

// -----

// A static AIE1 transfer with a resolved ShimNOC allocation remains valid.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(2, 0)
    aie.shim_dma_allocation @dma(%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma} : memref<64xi32>
    }
  }
}

// -----

// Static AIE1 transfers reject ShimPL tiles, which have no ShimDMA.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @dma(%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      // expected-error @+1 {{Unsupported tile type at (0, 0) Must be ShimNOC, Mem or Core.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @dma} : memref<64xi32>
    }
  }
}

// -----

// Runtime AIE1 transfers also require a ShimNOC allocation.
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @dma(%tile, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>, %n: i64) {
      // expected-error @+1 {{runtime sizes/strides are only supported for shim NOC tile DMAs.}}
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, %n, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @dma} : memref<64xi32>
    }
  }
}

// -----

// Without a device, getTargetModel uses the AIE1 compatibility model. The
// static verifier returns before allocation lookup; a symbol hook must too.
module {
  func.func @copy(%arg0: memref<64xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @missing} : memref<64xi32>
    return
  }
}
