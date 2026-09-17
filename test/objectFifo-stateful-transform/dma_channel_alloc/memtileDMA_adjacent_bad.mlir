//===- memtileDMA_adjacent_bad.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// Five S2MM channels fit the ordinary capacity of six, but not the capacity
// of four for adjacent-memory access. The full local memory forces the
// five-input join's shared pool onto the neighboring MemTile.
// CHECK: error: 'aie.tile' op number of input DMA channel exceeded!
// CHECK-SAME: requires at least 5 S2MM channels, but capacity is 4 for adjacent MemTile access
// CHECK: note: DMA endpoint @in0_cons_dma for ObjectFifo @in0 requires adjacent MemTile access
// CHECK: note: DMA endpoint @in1_cons_dma for ObjectFifo @in1 requires adjacent MemTile access
// CHECK: note: DMA endpoint @in2_cons_dma for ObjectFifo @in2 requires adjacent MemTile access
// CHECK: note: DMA endpoint @in3_cons_dma for ObjectFifo @in3 requires adjacent MemTile access
// CHECK: note: DMA endpoint @in4_cons_dma for ObjectFifo @in4 requires adjacent MemTile access
module {
  aie.device(npu2) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %shim2 = aie.tile(2, 0)
    %mem = aie.tile(0, 1)
    %reserved = aie.buffer(%mem) : memref<524288xi8>
    aie.objectfifo @in0(%shim0, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%shim0, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2(%shim1, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in3(%shim1, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in4(%shim2, {%mem}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mem, {%shim0}, 2 : i32) : !aie.objectfifo<memref<80xi32>>
    aie.objectfifo.link [@in0, @in1, @in2, @in3, @in4] -> [@out]([0, 16, 32, 48, 64] [])
  }
}
