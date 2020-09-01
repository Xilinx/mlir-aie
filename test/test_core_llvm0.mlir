// RUN: aie-opt --aie-llvm-lowering %s | FileCheck %s

//CHECK-LABEL: module @test_core_llvm0 {
//CHECK:   llvm.func @core11() {
//CHECK:     %[[CH0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//CHECK:     %[[CH1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
//CHECK:     %[[C16:.*]] = llvm.mlir.constant(16 : i32) : !llvm.i32
//CHECK:     %[[C32:.*]] = llvm.mlir.constant(32 : i128) : !llvm.i128
//CHECK:     llvm.call @llvm.aie.put.ms(%[[CH0]], %[[C16]]) : (!llvm.i32, !llvm.i32) -> ()
//CHECK:     llvm.call @llvm.aie.put.wms(%[[CH1]], %[[C32]]) : (!llvm.i32, !llvm.i128) -> ()
//CHECK:     %[[C64:.*]] = llvm.mlir.constant(64 : i384) : !llvm.i384
//CHECK:     llvm.call @llvm.aie.put.mcd(%[[C64]]) : (!llvm.i384) -> ()
//CHECK:     llvm.return
//CHECK:   }
//CHECK:   llvm.func @core21() {
//CHECK:     %[[CH0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
//CHECK:     %[[CH1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
//CHECK:     %[[R0:.*]] = llvm.call @llvm.aie.get.ss(%[[CH0]]) : (!llvm.i32) -> !llvm.i32
//CHECK:     %[[R1:.*]] = llvm.call @llvm.aie.get.ss(%[[CH1]]) : (!llvm.i32) -> !llvm.i32
//CHECK:     %{{.*}} = llvm.add %[[R0]], %[[R1]] : !llvm.i32
//CHECK:     %{{.*}} = llvm.call @llvm.aie.get.scd() : () -> !llvm.i384
//CHECK:     llvm.return
//CHECK:   }
//CHECK: }

// Test LLVM lowering to some AIE scalar intrinsic functions (streams, cascades)
// Each core's region is lowered to LLVM Dialect
module @test_core_llvm0 {
  %tile11 = AIE.tile(1, 1)
  %tile21 = AIE.tile(2, 1)

  %core11 = AIE.core(%tile11) {
    %0 = constant 0 : i32
    %1 = constant 1 : i32
    %val0 = constant 16 : i32
    %val1 = constant 32 : i128
    AIE.putStream(%val0 : i32,  %0 : i32)
    AIE.putStream(%val1 : i128, %1 : i32)
    %val2 = constant 64 : i384
    AIE.putCascade(%val2 : i384)
    AIE.end
  }

  %core21 = AIE.core(%tile21) {
    %0 = constant 0 : i32
    %1 = constant 1 : i32
    //%val0 = AIE.getStream(0) : i32
    %val0 = AIE.getStream(%0 : i32) : i32
    %val1 = AIE.getStream(%1 : i32) : i32
    %2 = addi %val0, %val1 : i32
    %3 = AIE.getCascade() : i384
    AIE.end
  }

}
