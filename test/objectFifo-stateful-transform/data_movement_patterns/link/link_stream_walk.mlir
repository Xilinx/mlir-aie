//===- link_stream_walk.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s

// A link output whose dimensionsToStream walks past one input object walks
// the stream: the outer dimension enumerates outputs, the inner ones are each
// output's transfer, and padding the outer dimension adds zero objects before
// and after the stream. The MemTile holds the stream's objects side by side
// between those zeros, filled by one iterating descriptor and drained by one
// descriptor per output.

// Three-row windows over four rows, one row of zeros above and below.
module @window {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @windows(%mem dimensionsToStream [<size = 4, stride = 8>, <size = 3, stride = 8>, <size = 8, stride = 1>], {%core}, 2 : i32) {padDimensions = #aie<bd_pad_layout_array[<const_pad_before = 1, const_pad_after = 1>, <const_pad_before = 0, const_pad_after = 0>, <const_pad_before = 0, const_pad_after = 0>]>} : !aie.objectfifo<memref<24xi32>>
    aie.objectfifo.link [@rows] -> [@windows] ([] [])
    %c = aie.core(%core) {
      %0 = aie.objectfifo.acquire @windows (Consume, 1) : memref<24xi32>
      aie.objectfifo.release @windows (Consume, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: module @window
// CHECK-DAG:   %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "rows_cons_buff_0"} : memref<48xi32> = dense<0>
// CHECK-DAG:   %[[FREE:.*]] = aie.lock(%{{.*}}) {init = 4 : i32, sym_name = "rows_cons_prod_lock_0"}
// CHECK-DAG:   %[[FULL:.*]] = aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "rows_cons_cons_lock_0"}
// CHECK-NOT:   sym_name = "rows_cons_buff_1"
// CHECK:       aie.memtile_dma
// CHECK:         aie.dma_start(S2MM, 0, ^[[FILL:.*]], ^
// CHECK:       ^[[FILL]]:
// CHECK:         aie.use_lock(%[[FREE]], AcquireGreaterEqual, %c1_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<48xi32> offset = 8 len = 8) {iteration = #aie.bd_iteration<size = 4, stride = 8, current = 0>}
// CHECK:         aie.use_lock(%[[FULL]], Release, %c1_i32{{.*}})
// CHECK:         aie.next_bd ^[[FILL]]
// CHECK:         aie.dma_start(MM2S, 0, ^[[OUT0:.*]], ^
// CHECK:       ^[[OUT0]]:
// CHECK:         aie.use_lock(%[[FULL]], AcquireGreaterEqual, %c2_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<48xi32> offset = 0 len = 24 sizes = [3, 8] strides = [8, 1])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c0_i32{{.*}})
// CHECK:         aie.use_lock(%[[FULL]], AcquireGreaterEqual, %c1_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<48xi32> offset = 8 len = 24 sizes = [3, 8] strides = [8, 1])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c1_i32{{.*}})
// CHECK:         aie.use_lock(%[[FULL]], AcquireGreaterEqual, %c1_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<48xi32> offset = 16 len = 24 sizes = [3, 8] strides = [8, 1])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c1_i32{{.*}})
// CHECK:         aie.use_lock(%[[FULL]], AcquireGreaterEqual, %c0_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<48xi32> offset = 24 len = 24 sizes = [3, 8] strides = [8, 1])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c2_i32{{.*}})
// CHECK:         aie.next_bd ^[[OUT0]]

// -----

// Every other row of five, each walked as two halves: no row dimension, so
// each output is one object. A skipped row is released once the next output
// has acquired it.
module @subsample {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @rows(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @evens(%mem dimensionsToStream [<size = 3, stride = 16>, <size = 2, stride = 1>, <size = 4, stride = 2>], {%core}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@rows] -> [@evens] ([] [])
    %c = aie.core(%core) {
      %0 = aie.objectfifo.acquire @evens (Consume, 1) : memref<8xi32>
      aie.objectfifo.release @evens (Consume, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: module @subsample
// CHECK-DAG:   %[[BUF:.*]] = aie.buffer(%{{.*}}) {sym_name = "rows_cons_buff_0"} : memref<40xi32>
// CHECK-DAG:   %[[FREE:.*]] = aie.lock(%{{.*}}) {init = 5 : i32, sym_name = "rows_cons_prod_lock_0"}
// CHECK:       aie.dma_bd(%[[BUF]] : memref<40xi32> offset = 0 len = 8) {iteration = #aie.bd_iteration<size = 5, stride = 8, current = 0>}
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:         aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<40xi32> offset = 0 len = 8 sizes = [2, 4] strides = [1, 2])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c1_i32{{.*}})
// CHECK:         aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c2_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<40xi32> offset = 16 len = 8 sizes = [2, 4] strides = [1, 2])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c2_i32{{.*}})
// CHECK:         aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c2_i32{{.*}})
// CHECK:         aie.dma_bd(%[[BUF]] : memref<40xi32> offset = 32 len = 8 sizes = [2, 4] strides = [1, 2])
// CHECK:         aie.use_lock(%[[FREE]], Release, %c2_i32{{.*}})
