// RUN: aie-opt --aie-llvm-lowering %s | FileCheck %s

// CHECK-LABEL: module @test_core_llvm1 {
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
// CHECK-NEXT:     %0 = llvm.mlir.constant(256 : index) : !llvm.i64
// CHECK-NEXT:     %1 = llvm.alloca %0 x !llvm.i32 : (!llvm.i64) -> !llvm<"i32*">
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT:     %3 = llvm.mlir.constant(56 : i32) : !llvm.i32
// CHECK-NEXT:     llvm.call @llvm.aie.lock.acquire.reg(%3, %2) : (!llvm.i32, !llvm.i32) -> ()
// CHECK-NEXT:     %4 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:     %5 = llvm.mlir.constant(16 : index) : !llvm.i64
// CHECK-NEXT:     %6 = llvm.getelementptr %1[%5] : (!llvm<"i32*">, !llvm.i64) -> !llvm<"i64*">
// CHECK-NEXT:     llvm.store %4, %6 : !llvm<"i64*">
// CHECK-NEXT:     %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:     %8 = llvm.mlir.constant(56 : i32) : !llvm.i32
// CHECK-NEXT:     llvm.call @llvm.aie.lock.release.reg(%8, %7) : (!llvm.i32, !llvm.i32) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm.func @core21() {
// CHECK-NEXT:     %0 = llvm.mlir.constant(256 : index) : !llvm.i64
// CHECK-NEXT:     %1 = llvm.alloca %0 x !llvm.i32 : (!llvm.i64) -> !llvm<"i32*">
// CHECK-NEXT:     %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:     %3 = llvm.mlir.constant(24 : i32) : !llvm.i32
// CHECK-NEXT:     llvm.call @llvm.aie.lock.acquire.reg(%3, %2) : (!llvm.i32, !llvm.i32) -> ()
// CHECK-NEXT:     %4 = llvm.mlir.constant(16 : index) : !llvm.i64
// CHECK-NEXT:     %5 = llvm.getelementptr %1[%4] : (!llvm<"i32*">, !llvm.i64) -> !llvm<"i64*">
// CHECK-NEXT:     %6 = llvm.load %5 : !llvm<"i64*">
// CHECK-NEXT:     %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT:     %8 = llvm.mlir.constant(24 : i32) : !llvm.i32
// CHECK-NEXT:     llvm.call @llvm.aie.lock.release.reg(%8, %7) : (!llvm.i32, !llvm.i32) -> ()
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
// CHECK-NEXT:}

// Test LLVM lowering for lock accesses and memory accesses (LockOp, UseLockOp, and BufferOp)
// Things to make sure:
//   - LockID: depending on which tile (or memory module) a lock is instantiated, create a lock ID
//             that has correct offset from a core's view (based on cardinal direction)
//   - Buffer: depending on which tile (or memory module) a buffer is instantiated, create an LLVM
//             static allocation (for now) for each core that can access to the buffer
module @test_core_llvm1 {
  %tile11 = AIE.tile(1, 1)
  %tile21 = AIE.tile(2, 1)

  %lock11_8 = AIE.lock(%tile11, 8)
  %buf11_0  = AIE.buffer(%tile11) : memref<256xi32>

  %core11 = AIE.core(%tile11) {
    AIE.useLock(%lock11_8, "Acquire", 0, 0)
    %0 = constant 1 : i32
    %i = constant 16 : index
    store %0, %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, "Release", 1, 0)
    AIE.end
  }

  %core21 = AIE.core(%tile21) {
    AIE.useLock(%lock11_8, "Acquire", 1, 0)
    %i = constant 16 : index
    %0 = load %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, "Release", 0, 0)
    AIE.end
  }
}
