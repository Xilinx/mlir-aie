//===- consumed_by_join.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A later segment of a join is taken only after the segments ahead of it in
// each object, so an input reaching the output through one is proven in whole
// objects of the join: granule_bytes, in output bytes.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// The second segment's core releases only on some passes, as when a dispatch
// skips it: the join's drainer waits for it, so a whole object received still
// shows the input taken.

// CHECK-LABEL: module @later_segment_sometimes_skipped
// CHECK: aie.route_endpoint @inb__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 2, granule_bytes = 256>], fifoName = "inb"}
module @later_segment_sometimes_skipped {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @ina(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @inb(%shim, {%core3}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%core2, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%core3, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 32] [])
    %flag = aie.buffer(%core3) : memref<1xi32>
    aie.core(%core2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @ina(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @a(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @ina(Consume, 1)
        aie.objectfifo.release @a(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%core3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @inb(Consume, 1) : memref<32xi32>
        %f = memref.load %flag[%c0] : memref<1xi32>
        %emit = arith.cmpi ne, %f, %c0_i32 : i32
        scf.if %emit {
          %o = aie.objectfifo.acquire @b(Produce, 1) : memref<32xi32>
          aie.objectfifo.release @b(Produce, 1)
        }
        aie.objectfifo.release @inb(Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The join's drainer reads with a transform, which stays within each segment,
// so the second segment still goes after the first in each object.

// CHECK-LABEL: module @later_segment_transformed_drainer
// CHECK: aie.route_endpoint @inb__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 2, granule_bytes = 256>], fifoName = "inb"}
module @later_segment_transformed_drainer {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @ina(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @inb(%shim, {%core3}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%core2, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%core3, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%mem dimensionsToStream [<size = 16, stride = 2>, <size = 2, stride = 1>], {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a, @b] -> [@out] ([0, 32] [])
    aie.core(%core2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @ina(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @a(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @ina(Consume, 1)
        aie.objectfifo.release @a(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%core3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @inb(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @b(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @inb(Consume, 1)
        aie.objectfifo.release @b(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// @inb passes a later segment of one join, then a later segment of a second:
// its lag behind the output is two nested roundings, more than one granule
// models, so it stays unproven. @ina passes only the second join's later
// segment, so one granule (an object of @out, 384 bytes); @inc passes the
// second join's first segment, so none.

// CHECK-LABEL: module @two_later_segments
// CHECK: aie.route_endpoint @ina__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 3, granule_bytes = 384>], fifoName = "ina"}
// CHECK: aie.route_endpoint @inb__prod__dma({{.*}}) DMA {fifoName = "inb"}
// CHECK: aie.route_endpoint @inc__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 3>], fifoName = "inc"}
module @two_later_segments {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem0 = aie.tile(0, 1)
    %mem1 = aie.tile(1, 1)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    %core12 = aie.tile(1, 2)
    aie.objectfifo @ina(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @inb(%shim, {%core3}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @inc(%shim, {%core12}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @a(%core2, {%mem0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%core3, {%mem0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @ab(%mem0, {%mem1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a, @b] -> [@ab] ([0, 32] [])
    aie.objectfifo @c(%core12, {%mem1}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%mem1, {%shim}, 2 : i32) : !aie.objectfifo<memref<96xi32>>
    aie.objectfifo.link [@c, @ab] -> [@out] ([0, 32] [])
    aie.core(%core2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @ina(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @a(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @ina(Consume, 1)
        aie.objectfifo.release @a(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%core3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @inb(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @b(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @inb(Consume, 1)
        aie.objectfifo.release @b(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%core12) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @inc(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @c(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @inc(Consume, 1)
        aie.objectfifo.release @c(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}
