
// arbiter() {
// %1 = masterset(north:1, east:2);
// }
// packetrules(east:1) {
// rule(10000|getRow(), %1);
// }
module {
  %00 = aie.core(0, 0)
  %01 = aie.core(0, 1)
  %02 = aie.core(0, 2)
  %03 = aie.core(0, 3)
  %10 = aie.core(1, 0)
  %11 = aie.core(1, 1)
  %12 = aie.core(1, 2)
  %13 = aie.core(1, 3)
  %20 = aie.core(2, 0)
  %21 = aie.core(2, 1)
  %22 = aie.core(2, 2)
  %23 = aie.core(2, 3)
  %30 = aie.core(3, 0)
  %31 = aie.core(3, 1)
  %32 = aie.core(3, 2)
  %33 = aie.core(3, 3)
  %p0 = aie.plio(0)
  %p1 = aie.plio(1)
  %p2 = aie.plio(2)
  %p3 = aie.plio(3)
  // North flowing input activations
  aie.flow(%p0, "North" : 0, %00, "ME" : 1)
  aie.flow(%p0, "North" : 0, %10, "ME" : 1)
  aie.flow(%p0, "North" : 0, %20, "ME" : 1)
  aie.flow(%p0, "North" : 0, %30, "ME" : 1)
  aie.flow(%p1, "North" : 0, %01, "ME" : 1)
  aie.flow(%p1, "North" : 0, %11, "ME" : 1)
  aie.flow(%p1, "North" : 0, %21, "ME" : 1)
  aie.flow(%p1, "North" : 0, %31, "ME" : 1)
  aie.flow(%p2, "North" : 0, %02, "ME" : 1)
  aie.flow(%p2, "North" : 0, %12, "ME" : 1)
  aie.flow(%p2, "North" : 0, %22, "ME" : 1)
  aie.flow(%p2, "North" : 0, %32, "ME" : 1)
  aie.flow(%p3, "North" : 0, %03, "ME" : 1)
  aie.flow(%p3, "North" : 0, %13, "ME" : 1)
  aie.flow(%p3, "North" : 0, %23, "ME" : 1)
  aie.flow(%p3, "North" : 0, %33, "ME" : 1)
  // South-west flowing results
  aie.flow(%33, "ME" : 0, %p3, "South" : 0)
  aie.flow(%32, "ME" : 0, %p2, "South" : 0)
  aie.flow(%31, "ME" : 0, %p1, "South" : 0)
  aie.flow(%30, "ME" : 0, %p0, "South" : 0)
}
