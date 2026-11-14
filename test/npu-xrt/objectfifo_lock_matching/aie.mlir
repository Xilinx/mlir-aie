//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Four cores in one column pass 16 values through 7, 6, 6 and 5 intra-tile
// fifos, each adding one, so the output is the input plus 24. Their 58 lock
// segments' locks fit the column's 64 only as a matching: placed in pool
// order, the last ones find no room.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.objectfifo @in0(%t00, {%t01}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1(%t01, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@in1] ([] [])
    aie.objectfifo @hop0(%t02, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @hop1(%t03, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @hop2(%t04, {%t05}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%t05, {%t01}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out0(%t01, {%t00}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@out1] -> [@out0] ([] [])
    aie.objectfifo @c2_0(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2_1(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2_2(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2_3(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2_4(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2_5(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c2_6(%t02, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3_0(%t03, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3_1(%t03, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3_2(%t03, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3_3(%t03, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3_4(%t03, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c3_5(%t03, {%t03}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c4_0(%t04, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c4_1(%t04, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c4_2(%t04, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c4_3(%t04, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c4_4(%t04, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c4_5(%t04, {%t04}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c5_0(%t05, {%t05}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c5_1(%t05, {%t05}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c5_2(%t05, {%t05}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c5_3(%t05, {%t05}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c5_4(%t05, {%t05}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %one = arith.constant 1 : i32
      %src = aie.objectfifo.acquire @in1(Consume, 1) : memref<16xi32>
      %p0 = aie.objectfifo.acquire @c2_0(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %src[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p0[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_0(Produce, 1)
      aie.objectfifo.release @in1(Consume, 1)
      %q0 = aie.objectfifo.acquire @c2_0(Consume, 1) : memref<16xi32>
      %p1 = aie.objectfifo.acquire @c2_1(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q0[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p1[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_1(Produce, 1)
      aie.objectfifo.release @c2_0(Consume, 1)
      %q1 = aie.objectfifo.acquire @c2_1(Consume, 1) : memref<16xi32>
      %p2 = aie.objectfifo.acquire @c2_2(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q1[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p2[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_2(Produce, 1)
      aie.objectfifo.release @c2_1(Consume, 1)
      %q2 = aie.objectfifo.acquire @c2_2(Consume, 1) : memref<16xi32>
      %p3 = aie.objectfifo.acquire @c2_3(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q2[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p3[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_3(Produce, 1)
      aie.objectfifo.release @c2_2(Consume, 1)
      %q3 = aie.objectfifo.acquire @c2_3(Consume, 1) : memref<16xi32>
      %p4 = aie.objectfifo.acquire @c2_4(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q3[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p4[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_4(Produce, 1)
      aie.objectfifo.release @c2_3(Consume, 1)
      %q4 = aie.objectfifo.acquire @c2_4(Consume, 1) : memref<16xi32>
      %p5 = aie.objectfifo.acquire @c2_5(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q4[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p5[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_5(Produce, 1)
      aie.objectfifo.release @c2_4(Consume, 1)
      %q5 = aie.objectfifo.acquire @c2_5(Consume, 1) : memref<16xi32>
      %p6 = aie.objectfifo.acquire @c2_6(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q5[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p6[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c2_6(Produce, 1)
      aie.objectfifo.release @c2_5(Consume, 1)
      %q6 = aie.objectfifo.acquire @c2_6(Consume, 1) : memref<16xi32>
      %d = aie.objectfifo.acquire @hop0(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q6[%j] : memref<16xi32>
        memref.store %v, %d[%j] : memref<16xi32>
      }
      aie.objectfifo.release @hop0(Produce, 1)
      aie.objectfifo.release @c2_6(Consume, 1)
      aie.end
    }
    aie.core(%t03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %one = arith.constant 1 : i32
      %src = aie.objectfifo.acquire @hop0(Consume, 1) : memref<16xi32>
      %p0 = aie.objectfifo.acquire @c3_0(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %src[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p0[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c3_0(Produce, 1)
      aie.objectfifo.release @hop0(Consume, 1)
      %q0 = aie.objectfifo.acquire @c3_0(Consume, 1) : memref<16xi32>
      %p1 = aie.objectfifo.acquire @c3_1(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q0[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p1[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c3_1(Produce, 1)
      aie.objectfifo.release @c3_0(Consume, 1)
      %q1 = aie.objectfifo.acquire @c3_1(Consume, 1) : memref<16xi32>
      %p2 = aie.objectfifo.acquire @c3_2(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q1[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p2[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c3_2(Produce, 1)
      aie.objectfifo.release @c3_1(Consume, 1)
      %q2 = aie.objectfifo.acquire @c3_2(Consume, 1) : memref<16xi32>
      %p3 = aie.objectfifo.acquire @c3_3(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q2[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p3[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c3_3(Produce, 1)
      aie.objectfifo.release @c3_2(Consume, 1)
      %q3 = aie.objectfifo.acquire @c3_3(Consume, 1) : memref<16xi32>
      %p4 = aie.objectfifo.acquire @c3_4(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q3[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p4[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c3_4(Produce, 1)
      aie.objectfifo.release @c3_3(Consume, 1)
      %q4 = aie.objectfifo.acquire @c3_4(Consume, 1) : memref<16xi32>
      %p5 = aie.objectfifo.acquire @c3_5(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q4[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p5[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c3_5(Produce, 1)
      aie.objectfifo.release @c3_4(Consume, 1)
      %q5 = aie.objectfifo.acquire @c3_5(Consume, 1) : memref<16xi32>
      %d = aie.objectfifo.acquire @hop1(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q5[%j] : memref<16xi32>
        memref.store %v, %d[%j] : memref<16xi32>
      }
      aie.objectfifo.release @hop1(Produce, 1)
      aie.objectfifo.release @c3_5(Consume, 1)
      aie.end
    }
    aie.core(%t04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %one = arith.constant 1 : i32
      %src = aie.objectfifo.acquire @hop1(Consume, 1) : memref<16xi32>
      %p0 = aie.objectfifo.acquire @c4_0(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %src[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p0[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c4_0(Produce, 1)
      aie.objectfifo.release @hop1(Consume, 1)
      %q0 = aie.objectfifo.acquire @c4_0(Consume, 1) : memref<16xi32>
      %p1 = aie.objectfifo.acquire @c4_1(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q0[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p1[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c4_1(Produce, 1)
      aie.objectfifo.release @c4_0(Consume, 1)
      %q1 = aie.objectfifo.acquire @c4_1(Consume, 1) : memref<16xi32>
      %p2 = aie.objectfifo.acquire @c4_2(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q1[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p2[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c4_2(Produce, 1)
      aie.objectfifo.release @c4_1(Consume, 1)
      %q2 = aie.objectfifo.acquire @c4_2(Consume, 1) : memref<16xi32>
      %p3 = aie.objectfifo.acquire @c4_3(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q2[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p3[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c4_3(Produce, 1)
      aie.objectfifo.release @c4_2(Consume, 1)
      %q3 = aie.objectfifo.acquire @c4_3(Consume, 1) : memref<16xi32>
      %p4 = aie.objectfifo.acquire @c4_4(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q3[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p4[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c4_4(Produce, 1)
      aie.objectfifo.release @c4_3(Consume, 1)
      %q4 = aie.objectfifo.acquire @c4_4(Consume, 1) : memref<16xi32>
      %p5 = aie.objectfifo.acquire @c4_5(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q4[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p5[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c4_5(Produce, 1)
      aie.objectfifo.release @c4_4(Consume, 1)
      %q5 = aie.objectfifo.acquire @c4_5(Consume, 1) : memref<16xi32>
      %d = aie.objectfifo.acquire @hop2(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q5[%j] : memref<16xi32>
        memref.store %v, %d[%j] : memref<16xi32>
      }
      aie.objectfifo.release @hop2(Produce, 1)
      aie.objectfifo.release @c4_5(Consume, 1)
      aie.end
    }
    aie.core(%t05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %one = arith.constant 1 : i32
      %src = aie.objectfifo.acquire @hop2(Consume, 1) : memref<16xi32>
      %p0 = aie.objectfifo.acquire @c5_0(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %src[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p0[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c5_0(Produce, 1)
      aie.objectfifo.release @hop2(Consume, 1)
      %q0 = aie.objectfifo.acquire @c5_0(Consume, 1) : memref<16xi32>
      %p1 = aie.objectfifo.acquire @c5_1(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q0[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p1[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c5_1(Produce, 1)
      aie.objectfifo.release @c5_0(Consume, 1)
      %q1 = aie.objectfifo.acquire @c5_1(Consume, 1) : memref<16xi32>
      %p2 = aie.objectfifo.acquire @c5_2(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q1[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p2[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c5_2(Produce, 1)
      aie.objectfifo.release @c5_1(Consume, 1)
      %q2 = aie.objectfifo.acquire @c5_2(Consume, 1) : memref<16xi32>
      %p3 = aie.objectfifo.acquire @c5_3(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q2[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p3[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c5_3(Produce, 1)
      aie.objectfifo.release @c5_2(Consume, 1)
      %q3 = aie.objectfifo.acquire @c5_3(Consume, 1) : memref<16xi32>
      %p4 = aie.objectfifo.acquire @c5_4(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q3[%j] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %p4[%j] : memref<16xi32>
      }
      aie.objectfifo.release @c5_4(Produce, 1)
      aie.objectfifo.release @c5_3(Consume, 1)
      %q4 = aie.objectfifo.acquire @c5_4(Consume, 1) : memref<16xi32>
      %d = aie.objectfifo.acquire @out1(Produce, 1) : memref<16xi32>
      scf.for %j = %c0 to %c16 step %c1 {
        %v = memref.load %q4[%j] : memref<16xi32>
        memref.store %v, %d[%j] : memref<16xi32>
      }
      aie.objectfifo.release @out1(Produce, 1)
      aie.objectfifo.release @c5_4(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) { metadata = @out0, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_memcpy_nd (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) { metadata = @in0, id = 0 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_wait { symbol = @out0 }
    }
  }
}
