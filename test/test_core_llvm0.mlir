// RUN: aie-opt --aie-llvm-lowering %s | FileCheck %s

// CHECK-LABEL: module @test_core_llvm0 {
// CHECK-NEXT:   llvm.func @llvm.aie.put.ms(!llvm.i32, !llvm.i32)
// CHECK-NEXT:   llvm.func @llvm.aie.put.wms(!llvm.i32, !llvm.i128)
// CHECK-NEXT:   llvm.func @llvm.aie.put.fms(!llvm.i32, !llvm.float)
// CHECK-NEXT:   llvm.func @llvm.aie.get.ss(!llvm.i32) -> !llvm.i32
// CHECK-NEXT:   llvm.func @llvm.aie.get.wss(!llvm.i32) -> !llvm.i128
// CHECK-NEXT:   llvm.func @llvm.aie.get.fss(!llvm.float) -> !llvm.float
// CHECK-NEXT:   llvm.func @llvm.aie.put.mcd(!llvm.i384)
// CHECK-NEXT:   llvm.func @llvm.aie.get.scd() -> !llvm.i384
// CHECK-NEXT:   llvm.func @llvm.aie.lock.acquire.reg(!llvm.i32, !llvm.i32)
// CHECK-NEXT:   llvm.func @llvm.aie.lock.release.reg(!llvm.i32, !llvm.i32)
// CHECK-NEXT:   llvm.func @core11() {
// CHECK-NEXT:     %0 = llvm.mlir.constant(16 : i32) : !llvm.i32
// CHECK-NEXT:     %1 = llvm.mlir.constant(32 : i128) : !llvm.i128
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT:     llvm.call @llvm.aie.put.ms(%2, %0) : (!llvm.i32, !llvm.i32) -> ()
// CHECK-NEXT:     %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:     llvm.call @llvm.aie.put.wms(%3, %1) : (!llvm.i32, !llvm.i128) -> ()
// CHECK-NEXT:     %4 = llvm.mlir.constant(64 : i384) : !llvm.i384
// CHECK-NEXT:     llvm.call @llvm.aie.put.mcd(%4) : (!llvm.i384) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @core21() {
// CHECK-NEXT:     %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:     %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT:     %2 = llvm.call @llvm.aie.get.ss(%1) : (!llvm.i32) -> !llvm.i32
// CHECK-NEXT:     %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:     %4 = llvm.call @llvm.aie.get.ss(%3) : (!llvm.i32) -> !llvm.i32
// CHECK-NEXT:     %5 = llvm.add %2, %4 : !llvm.i32
// CHECK-NEXT:     %6 = llvm.call @llvm.aie.get.scd() : () -> !llvm.i384
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test LLVM lowering to some AIE scalar intrinsic functions (streams, cascades)
// Each core's region is lowered to LLVM Dialect
module @test_core_llvm0 {
  %tile11 = AIE.tile(1, 1)
  %tile21 = AIE.tile(2, 1)

  %core11 = AIE.core(%tile11) {
    %val0 = constant 16 : i32
    %val1 = constant 32 : i128
    AIE.putStream(0, %val0 : i32)
    AIE.putStream(1, %val1 : i128)
    %val2 = constant 64 : i384
    AIE.putCascade(%val2 : i384)
    AIE.end
  }

  %core21 = AIE.core(%tile21) {
    %0 = constant 1 : i32
    %val0 = AIE.getStream(0) : i32
    %val1 = AIE.getStream(1) : i32
    %1 = addi %val0, %val1 : i32
    %2 = AIE.getCascade() : i384
    AIE.end
  }

}
