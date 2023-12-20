// RUN: aie-opt --aie-localize-locks --aie-standard-lowering="tilecol=1 tilerow=3" %s | FileCheck --check-prefix=CHECK %s

// CHECK: module @test attributes {llvm.target_triple = "aie"} {
// CHECK:   func.func private @kernel(%arg0: index) {
// CHECK-NEXT:     %0 = arith.index_cast %arg0 : index to i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     call @llvm.aie.lock.acquire.reg(%0, %c0_i32) : (i32, i32) -> ()
// CHECK-NEXT:     return
// CHECK:   }
// CHECK:   func.func @core_1_3() {
// CHECK:     %c48 = arith.constant 48 : index
// CHECK:     call @kernel(%c48) : (index) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module @test {
 aie.device(xcvc1902) {
  %tile13 = aie.tile(1, 3)
  %lock13_3 = aie.lock(%tile13, 0)

  func.func private @kernel(%lock : index) {
    aie.use_lock(%lock, "Acquire", 0)
    return
  }

  %core13 = aie.core(%tile13) {
    func.call @kernel(%lock13_3) : (index) -> ()
    aie.end
  }
 }
}
