// RUN: aie-opt --aie-llvm-lowering %s | FileCheck %s
// CHECK-LABEL: module @codegen1 {
// CHECK:  llvm.func @core33() {
// CHECK:    %[[VAR1:.*]] = llvm.mlir.addressof @a : !llvm<"[4 x i32]*">
// CHECK:    %[[VAR5:.*]] = llvm.getelementptr %[[VAR1]][%{{.*}}, %{{.*}}] : (!llvm<"[4 x i32]*">, !llvm.i64, !llvm.i64) -> !llvm<"i32*">
// CHECK:    llvm.store %{{.*}}, %[[VAR5]] : !llvm<"i32*">
// CHECK:  }
// CHECK:  llvm.func @core43() {
// CHECK:    %[[VAR1:.*]] = llvm.mlir.addressof @a : !llvm<"[4 x i32]*">
// CHECK:    %[[VAR5:.*]] = llvm.getelementptr %[[VAR1]][%{{.*}}, %{{.*}}] : (!llvm<"[4 x i32]*">, !llvm.i64, !llvm.i64) -> !llvm<"i32*">
// CHECK:    %{{.*}} = llvm.load %[[VAR5]] : !llvm<"i32*">
// CHECK:  }
// CHECK:       }

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
