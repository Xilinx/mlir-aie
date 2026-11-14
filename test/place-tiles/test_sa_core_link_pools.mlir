//===- test_sa_core_link_pools.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t 2> %t.err
// RUN: FileCheck %s --check-prefix=WARN --input-file=%t.err
// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t 2>&1 >/dev/null | FileCheck %s --check-prefix=ADDR

// A link through a core tile keeps one pool there, as the split builds it with
// the link's pool owner and depth, and the SA charges that pool, through the
// ends' planned DMA ends.

// The join's pool is the output's: 4 objects of 8 KB beside 40000 bytes.
// WARN: [[@LINE+3]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: 'aie.buffer' op could not be placed: buffer "out__buf3" needs 8192 bytes
module @join_core {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @in1(%b, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<2048xi32>>
    %big = aie.buffer(%l) {sym_name = "big"} : memref<10000xi32>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 1024] [])
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}

// -----

// A 1->1 link whose output is larger keeps the output's pool.
// WARN: [[@LINE+3]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: 'aie.buffer' op could not be placed: buffer "out__buf3" needs 8192 bytes
module @link_core {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 1 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<2048xi32>>
    %big = aie.buffer(%l) {sym_name = "big"} : memref<10000xi32>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// A distribute's pool is its input's, which the SA charged before as well.
// WARN: [[@LINE+5]]:3: warning: SA placer's resource estimate is over capacity
// WARN-NOT: over capacity
// ADDR: error: 'aie.buffer' op could not be placed: buffer "in__cons__buf3" needs 8192 bytes
// ADDR-NOT: error
module @dist_core_full {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%s, {%l}, 4 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @o0(%l, {%a}, 4 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @o1(%l, {%b}, 4 : i32) : !aie.objectfifo<memref<1024xi32>>
    %big = aie.buffer(%l) {sym_name = "big"} : memref<10000xi32>
    aie.objectfifo.link [@in] -> [@o0, @o1] ([] [0, 1024])
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @o0(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @o0(Consume, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @o1(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @o1(Consume, 1)
      aie.end
    }
  }
}
