// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

// arbiter() {
// %1 = masterset(north:1, east:2);
// }
// packetrules(east:1) {
// rule(10000|getRow(), %1);
// }
module {
  %00 = AIE.tile(0, 0)
  %01 = AIE.tile(0, 1)
  %02 = AIE.tile(0, 2)
  %03 = AIE.tile(0, 3)
  %10 = AIE.tile(1, 0)
  %11 = AIE.tile(1, 1)
  %12 = AIE.tile(1, 2)
  %13 = AIE.tile(1, 3)
  %20 = AIE.tile(2, 0)
  %21 = AIE.tile(2, 1)
  %22 = AIE.tile(2, 2)
  %23 = AIE.tile(2, 3)
  %30 = AIE.tile(3, 0)
  %31 = AIE.tile(3, 1)
  %32 = AIE.tile(3, 2)
  %33 = AIE.tile(3, 3)
  %p0 = AIE.plio(0)
  %p1 = AIE.plio(1)
  %p2 = AIE.plio(2)
  %p3 = AIE.plio(3)
  // North flowing input activations
  AIE.flow(%p0, "North" : 0, %00, "ME" : 1)
  AIE.flow(%p0, "North" : 0, %10, "ME" : 1)
  AIE.flow(%p0, "North" : 0, %20, "ME" : 1)
  AIE.flow(%p0, "North" : 0, %30, "ME" : 1)
  AIE.flow(%p1, "North" : 0, %01, "ME" : 1)
  AIE.flow(%p1, "North" : 0, %11, "ME" : 1)
  AIE.flow(%p1, "North" : 0, %21, "ME" : 1)
  AIE.flow(%p1, "North" : 0, %31, "ME" : 1)
  AIE.flow(%p2, "North" : 0, %02, "ME" : 1)
  AIE.flow(%p2, "North" : 0, %12, "ME" : 1)
  AIE.flow(%p2, "North" : 0, %22, "ME" : 1)
  AIE.flow(%p2, "North" : 0, %32, "ME" : 1)
  AIE.flow(%p3, "North" : 0, %03, "ME" : 1)
  AIE.flow(%p3, "North" : 0, %13, "ME" : 1)
  AIE.flow(%p3, "North" : 0, %23, "ME" : 1)
  AIE.flow(%p3, "North" : 0, %33, "ME" : 1)
  // South-west flowing results
  AIE.flow(%33, "ME" : 0, %p3, "South" : 0)
  AIE.flow(%32, "ME" : 0, %p2, "South" : 0)
  AIE.flow(%31, "ME" : 0, %p1, "South" : 0)
  AIE.flow(%30, "ME" : 0, %p0, "South" : 0)
}
