// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %3 = "AIE.tile"() {col=2:i32, row=3:i32} : () -> index
  %4 = "AIE.tile"() {col=3:i32, row=2:i32} : () -> index
  "AIE.flow"(%3, %4) {sourceBundle=0:i32,sourceChannel=1:i32,
                      destBundle=4:i32,destChannel=0:i32} : (index, index) -> ()
}
