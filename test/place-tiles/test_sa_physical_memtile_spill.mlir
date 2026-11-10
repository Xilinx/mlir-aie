//===- test_sa_physical_memtile_spill.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=3" %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: not aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-objectfifo-allocate %t 2>&1 >/dev/null | FileCheck %s --check-prefix=ALLOC

// The SA's mem tile spill estimate counts what a physical mem tile already
// holds, its objectFIFO pools and its aie.buffers, at its position, as it
// counts a pinned logical mem tile's. On one column the logical link's
// 256 KB lands on (0, 1) beside 320 KB the physical tile holds, which
// allocation cannot fit in 512 KB; the SA warns, as it does for the pinned
// logical form. Before, it counted logical mem tiles only and placed both
// silently. On two columns the logical link still goes to (1, 1) and
// allocation succeeds.
// ALLOC-COUNT-2: error: 'aie.objectfifo.pool' op could not place buffers in accessible memory with available capacity
// ALLOC-NOT: error

// WARN: [[@LINE+2]]:3: warning: SA placer's resource estimate is over capacity
module @pools_1col {
  aie.device(npu2_1col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.tile(0, 1)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<32768xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.objectfifo @pin(%sh, {%hm}, 2 : i32) : !aie.objectfifo<memref<40960xi32>>
    aie.objectfifo @pout(%hm, {%c2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@pin] -> [@pout] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %0 = aie.objectfifo.acquire @pout(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @pout(Consume, 1)
      aie.end
    }
  }
}

// -----

// WARN: [[@LINE+2]]:3: warning: SA placer's resource estimate is over capacity
module @buffer_1col {
  aie.device(npu2_1col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.tile(0, 1)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %hb = aie.buffer(%hm) {sym_name = "hand"} : memref<81920xi32>
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<32768xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}

// -----

// WARN-NOT: over capacity
// CHECK-LABEL: module @pools_2col
// CHECK: aie.objectfifo @out(%mem_tile_1_1,
module @pools_2col {
  aie.device(npu2_2col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.tile(0, 1)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<24576xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.objectfifo @pin(%sh, {%hm}, 2 : i32) : !aie.objectfifo<memref<40960xi32>>
    aie.objectfifo @pout(%hm, {%c2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@pin] -> [@pout] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %0 = aie.objectfifo.acquire @pout(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @pout(Consume, 1)
      aie.end
    }
  }
}
