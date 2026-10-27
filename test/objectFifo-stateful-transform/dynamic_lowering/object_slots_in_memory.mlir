//===- object_slots_in_memory.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's objects stay in their slots wherever they are still picked at run
// time, rather than being carried through control flow as memrefs.

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll -split-input-file %s | FileCheck %s

// A static core whose inner loop runs 3 times over a depth-2 objectFifo: every
// outer iteration leaves the objects rotated by one, so they do not fold.

// CHECK-LABEL: aie.device(npu2)
// CHECK:     %[[CORE:.*]] = aie.core
// CHECK:       %[[S0:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       memref.store %{{.*}}buff_0, %[[S0]][]
// CHECK:       %[[S1:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       memref.store %{{.*}}buff_1, %[[S1]][]
// CHECK:       scf.for
// CHECK-NOT:   iter_args
// CHECK:         %[[OBJ:.*]] = memref.load %[[S0]][] : memref<memref<8xi8>>
// CHECK:         memref.load %[[OBJ]]
// CHECK:       aie.end
module {
  aie.device(npu2) {
    %t1 = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    aie.buffer @buf(%t2) : memref<8xi8>
    aie.objectfifo @fifo(%t1, {%t2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>
    %core = aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %o = %c0 to %cmax step %c1 {
        scf.for %i = %c0 to %c3 step %c1 {
          %x = aie.objectfifo.acquire @fifo(Consume, 1) : memref<8xi8>
          %v = memref.load %x[%c0] : memref<8xi8>
          %buf = aie.get_buffer @buf : memref<8xi8>
          memref.store %v, %buf[%c0] : memref<8xi8>
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    } {dynamic_objfifo_lowering = false}
  }
}

// -----

// As above, but the core also picks between its own buffers with selects that
// fold once the inner loop is unrolled. The objects are still rotated by one
// every outer iteration, so they stay in their slots: the folded selects must
// not make up for objects promotion would carry through the loop.

// CHECK-LABEL: aie.device(npu2)
// CHECK:     aie.core
// CHECK:       %[[S0:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       %[[S1:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       scf.for
// CHECK-NOT:   iter_args
// CHECK:         %[[OBJ:.*]] = memref.load %[[S0]][] : memref<memref<8xi8>>
// CHECK:         memref.load %[[OBJ]]
// CHECK-NOT:     arith.select
// CHECK:       aie.end
module {
  aie.device(npu2) {
    %t1 = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    aie.buffer @b0(%t2) : memref<8xi8>
    aie.buffer @b1(%t2) : memref<8xi8>
    aie.objectfifo @fifo(%t1, {%t2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>
    %core = aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %o = %c0 to %cmax step %c1 {
        scf.for %i = %c0 to %c3 step %c1 {
          %x = aie.objectfifo.acquire @fifo(Consume, 1) : memref<8xi8>
          %v = memref.load %x[%c0] : memref<8xi8>
          %r = arith.remui %i, %c2 : index
          %even = arith.cmpi eq, %r, %c0 : index
          %b0 = aie.get_buffer @b0 : memref<8xi8>
          %b1 = aie.get_buffer @b1 : memref<8xi8>
          %d0 = arith.select %even, %b0, %b1 : memref<8xi8>
          memref.store %v, %d0[%c0] : memref<8xi8>
          %d1 = arith.select %even, %b1, %b0 : memref<8xi8>
          memref.store %v, %d1[%c0] : memref<8xi8>
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    } {dynamic_objfifo_lowering = false}
  }
}

// -----

// A dynamic core whose loop is a CFG cycle keeps its objects in their slots
// rather than threading them through block arguments. The loop header may
// still carry the held-object count, which only a later canonicalize removes.

// CHECK-LABEL: aie.device(npu2)
// CHECK:     aie.core
// CHECK:       %[[S0:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       %[[S1:.*]] = memref.alloca() : memref<memref<8xi8>>
// CHECK:       cf.br ^[[LOOP:bb[0-9]+]](%{{.*}} : index{{(, i32)?}})
// CHECK:     ^[[LOOP]](%{{[0-9]+}}: index{{(, %[0-9]+: i32)?}}):
// CHECK:       %[[OBJ:.*]] = memref.load %[[S0]][] : memref<memref<8xi8>>
// CHECK:       memref.load %[[OBJ]]
module {
  aie.device(npu2) {
    %t1 = aie.tile(0, 1)
    %t2 = aie.tile(0, 2)
    aie.buffer @buf(%t2) : memref<8xi8>
    aie.objectfifo @fifo(%t1, {%t2}, 2 : i32) : !aie.objectfifo<memref<8xi8>>
    %core = aie.core(%t2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      cf.br ^loop(%c0 : index)
    ^loop(%i: index):
      %o = aie.objectfifo.acquire @fifo(Consume, 1) : memref<8xi8>
      %v = memref.load %o[%c0] : memref<8xi8>
      %buf = aie.get_buffer @buf : memref<8xi8>
      memref.store %v, %buf[%c0] : memref<8xi8>
      aie.objectfifo.release @fifo(Consume, 1)
      %n = arith.addi %i, %c1 : index
      %cond = arith.cmpi slt, %n, %c14 : index
      cf.cond_br %cond, ^loop(%n : index), ^exit
    ^exit:
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
