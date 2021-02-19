// RUN: aie-opt --aie-standard-lowering="tilecol=3 tilerow=3" %s | FileCheck --check-prefix=CHECK33 %s
// RUN: aie-opt --aie-standard-lowering="tilecol=4 tilerow=3" %s | FileCheck --check-prefix=CHECK43 %s
// CHECK33-LABEL:  func @core33() {
// CHECK33:    %0 = get_global_memref @a : memref<4xi32>
// CHECK33:    %c0 = constant 0 : index
// CHECK33:    %c377_i32 = constant 377 : i32
// CHECK33:    store %c377_i32, %0[%c0] : memref<4xi32>
// CHECK33:    return
// CHECK33:  }

// CHECK43-LABEL:  func @core43() {
// CHECK43:    %0 = get_global_memref @a : memref<4xi32>
// CHECK43:    %c0 = constant 0 : index
// CHECK43:    %1 = load %0[%c0] : memref<4xi32>
// CHECK43:    return
// CHECK43:  }

module @codegen1 {
  %t33 = AIE.tile(3, 3)
  %a = AIE.buffer(%t33) { sym_name = "a" } : memref<4xi32>
  %core33 = AIE.core(%t33) {
    %0 = constant 0 : index
    %377 = constant 377 : i32
    store %377, %a[%0] : memref<4xi32>
    AIE.end
  }
  %t34 = AIE.tile(4, 3)

  %core34 = AIE.core(%t34) {
    %0 = constant 0 : index
    %1 = load %a[%0] : memref<4xi32>
//    AIE.debug(%1 : i32)
    AIE.end
  }
}
