// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
%c2 = constant 2 : index
  %c3 = constant 3 : index
	%3 = "AIE.core"(%c2, %c3) : (index, index) -> index
		%4 = "AIE.core"(%c3, %c2) : (index, index) -> index
"AIE.flow"(%3, %4) {sourcePort=1:i32,destPort=0:i32} : (index, index) -> ()
}
