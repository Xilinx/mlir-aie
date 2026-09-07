//===- inline_loop_block_arg.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// A callee sequence whose body contains an scf.for, with the induction
// variable used inside the loop.
//
// The induction variable is a block argument, so it has no defining op and the
// "is this value defined inside the op being inlined?" test in
// collectReferencedSSAValues cannot see it that way. Without treating block
// arguments explicitly it is collected as an *external* reference, reaches
// copyReferencedSSAValues, and fails there on its null getDefiningOp() check
// with "Referenced value is not defined by an operation".

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)

    // CHECK-LABEL: aie.runtime_sequence @main_seq
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @loop_device}
      // The loop is inlined intact, and the induction variable stays the
      // loop's own -- it must not have been hoisted into the caller.
      // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:   %[[C:.*]] = arith.index_cast %[[IV]] : index to i32
      // CHECK:   aiex.npu.rtp_write({{.*}}, 0, %[[C]])
      // CHECK: }
      aiex.configure @loop_device {
        aiex.run @loop_seq(%arg0) : (memref<64xi32>)
      }
    }
  }

  // CHECK: aie.device(npu2) @loop_device
  aie.device(npu2) @loop_device {
    %tile_0_2 = aie.tile(0, 2)
    %rtp_0_0 = aie.buffer(%tile_0_2) {sym_name = "rtp_0_0", address = 0xDEADBEEF : i32} : memref<1xi32>

    aie.runtime_sequence @loop_seq(%arg0: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %v = arith.index_cast %i : index to i32
        aiex.npu.rtp_write(@rtp_0_0, 0, %v) : i32
      }
    }
  }
}
