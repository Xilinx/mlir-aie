//===- forever_loop_counter.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's INT64_MAX loop over an objectFIFO reaches LLVM as an unconditional
// backedge, with no counter and no bound comparison, on both FIFO ends.

// RUN: %aiecc -O0 --get='llvmIR_{0}.ll' --tmpdir=%t.o0 --output-dir=%t.o0 %s
// RUN: FileCheck %s -DCORE=core_0_2 --input-file=%t.o0/llvmIR_main_core_0_2.ll
// RUN: FileCheck %s -DCORE=core_0_3 --input-file=%t.o0/llvmIR_main_core_0_3.ll

// RUN: %aiecc --get='llvmIR_{0}.ll' --tmpdir=%t.default --output-dir=%t.default %s
// RUN: FileCheck %s -DCORE=core_0_2 --input-file=%t.default/llvmIR_main_core_0_2.ll
// RUN: FileCheck %s -DCORE=core_0_3 --input-file=%t.default/llvmIR_main_core_0_3.ll

// CHECK:       define void @[[CORE]]()
// CHECK-NOT:   9223372036854775807
// CHECK:       br label %[[HEAD:[0-9]+]]
// CHECK:       {{^}}[[HEAD]]:
// CHECK-NEXT:  call void @llvm.aie2p.acquire
// CHECK-NOT:   9223372036854775807
// CHECK:       call void @llvm.aie2p.release
// CHECK-NOT:   9223372036854775807
// CHECK:       br label %[[HEAD]]
// CHECK-NEXT:  }

module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @of(%t02, {%t03}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %out = aie.buffer(%t03) {sym_name = "out"} : memref<16xi32>
    %producer = aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %cmax = arith.constant 9223372036854775807 : index
      %v = arith.constant 7 : i32
      scf.for %iter = %c0 to %cmax step %c1 {
        %elem = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
        scf.for %i = %c0 to %c16 step %c1 {
          memref.store %v, %elem[%i] : memref<16xi32>
        }
        aie.objectfifo.release @of (Produce, 1)
      }
      aie.end
    }
    %consumer = aie.core(%t03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %iter = %c0 to %cmax step %c1 {
        %elem = aie.objectfifo.acquire @of (Consume, 1) : memref<16xi32>
        scf.for %i = %c0 to %c16 step %c1 {
          %x = memref.load %elem[%i] : memref<16xi32>
          memref.store %x, %out[%i] : memref<16xi32>
        }
        aie.objectfifo.release @of (Consume, 1)
      }
      aie.end
    }
  }
}
