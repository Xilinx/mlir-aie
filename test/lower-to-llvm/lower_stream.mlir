// RUN: aie-opt --aie-llvm-lowering="tilecol=1 tilerow=1" %s | FileCheck --check-prefix=CHECK11 %s
// RUN: aie-opt --aie-llvm-lowering="tilecol=2 tilerow=1" %s | FileCheck --check-prefix=CHECK21 %s

//CHECK11:   llvm.func @core11() {
//CHECK11:     %[[CH0:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK11:     %[[CH1:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK11:     %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
//CHECK11:     %[[C32:.*]] = llvm.mlir.constant(32 : i128) : i128
//CHECK11:     llvm.call @llvm.aie.put.ms(%[[CH0]], %[[C16]]) : (i32, i32) -> ()
//CHECK11:     llvm.call @llvm.aie.put.wms(%[[CH1]], %[[C32]]) : (i32, i128) -> ()
//CHECK11:     %[[C64:.*]] = llvm.mlir.constant(64 : i384) : i384
//CHECK11:     llvm.call @llvm.aie.put.mcd(%[[C64]]) : (i384) -> ()
//CHECK11:     llvm.return
//CHECK11:   }

//CHECK21:   llvm.func @core21() {
//CHECK21:     %[[CH0:.*]] = llvm.mlir.constant(0 : i32) : i32
//CHECK21:     %[[CH1:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK21:     %[[R0:.*]] = llvm.call @llvm.aie.get.ss(%[[CH0]]) : (i32) -> i32
//CHECK21:     %[[R1:.*]] = llvm.call @llvm.aie.get.ss(%[[CH1]]) : (i32) -> i32
//CHECK21:     %{{.*}} = llvm.add %[[R0]], %[[R1]] : i32
//CHECK21:     %{{.*}} = llvm.call @llvm.aie.get.scd() : () -> i384
//CHECK21:     llvm.return
//CHECK21:   }

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
