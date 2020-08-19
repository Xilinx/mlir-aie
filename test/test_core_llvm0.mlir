// RUN: aie-opt --aie-llvm-lowering %s | FileCheck %s

//CHECK-LABEL: module @test_core_llvm0 {
//CHECK:   llvm.func @llvm.aie.put.ms(!llvm.i32, !llvm.i1)
//CHECK:   llvm.func @llvm.aie.put.wms(!llvm.i128, !llvm.i1)
//CHECK:   llvm.func @llvm.aie.put.fms(!llvm.float, !llvm.i1)
//CHECK:   llvm.func @llvm.aie.get.ss(!llvm.i1) -> !llvm.i32
//CHECK:   llvm.func @llvm.aie.get.wss(!llvm.i1) -> !llvm.i128
//CHECK:   llvm.func @llvm.aie.get.fss(!llvm.i1) -> !llvm.float
//CHECK:   llvm.func @llvm.aie.put.mcd(!llvm.i384)
//CHECK:   llvm.func @llvm.aie.get.scd() -> !llvm.i384
//CHECK:   llvm.func @llvm.aie.lock.acquire.reg(!llvm.i32, !llvm.i32)
//CHECK:   llvm.func @llvm.aie.lock.release.reg(!llvm.i32, !llvm.i32)
//CHECK:   llvm.func @core11() {
//CHECK:     %0 = llvm.mlir.constant(false) : !llvm.i1
//CHECK:     %1 = llvm.mlir.constant(true) : !llvm.i1
//CHECK:     %2 = llvm.mlir.constant(16 : i32) : !llvm.i32
//CHECK:     %3 = llvm.mlir.constant(32 : i128) : !llvm.i128
//CHECK:     llvm.call @llvm.aie.put.ms(%2, %0) : (!llvm.i32, !llvm.i1) -> ()
//CHECK:     llvm.call @llvm.aie.put.wms(%3, %1) : (!llvm.i128, !llvm.i1) -> ()
//CHECK:     %4 = llvm.mlir.constant(64 : i384) : !llvm.i384
//CHECK:     llvm.call @llvm.aie.put.mcd(%4) : (!llvm.i384) -> ()
//CHECK:     llvm.return
//CHECK:   }
//CHECK:   llvm.func @core21() {
//CHECK:     %0 = llvm.mlir.constant(false) : !llvm.i1
//CHECK:     %1 = llvm.mlir.constant(true) : !llvm.i1
//CHECK:     %2 = llvm.call @llvm.aie.get.ss(%0) : (!llvm.i1) -> !llvm.i32
//CHECK:     %3 = llvm.call @llvm.aie.get.ss(%1) : (!llvm.i1) -> !llvm.i32
//CHECK:     %4 = llvm.add %2, %3 : !llvm.i32
//CHECK:     %5 = llvm.call @llvm.aie.get.scd() : () -> !llvm.i384
//CHECK:     llvm.return
//CHECK:   }
//CHECK: }

// Test LLVM lowering to some AIE scalar intrinsic functions (streams, cascades)
// Each core's region is lowered to LLVM Dialect
module @test_core_llvm0 {
  %tile11 = AIE.tile(1, 1)
  %tile21 = AIE.tile(2, 1)

  %core11 = AIE.core(%tile11) {
    %0 = constant 0 : i1
    %1 = constant 1 : i1
    %val0 = constant 16 : i32
    %val1 = constant 32 : i128
    AIE.putStream(%val0 : i32,  %0 : i1)
    AIE.putStream(%val1 : i128, %1 : i1)
    %val2 = constant 64 : i384
    AIE.putCascade(%val2 : i384)
    AIE.end
  }

  %core21 = AIE.core(%tile21) {
    %0 = constant 0 : i1
    %1 = constant 1 : i1
    //%val0 = AIE.getStream(0) : i32
    %val0 = AIE.getStream(%0 : i1) : i32
    %val1 = AIE.getStream(%1 : i1) : i32
    %2 = addi %val0, %val1 : i32
    %3 = AIE.getCascade() : i384
    AIE.end
  }

}
