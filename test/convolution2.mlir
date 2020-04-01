
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
  %24 = aie.core(2, 4)
  %25 = aie.core(2, 5)
  %26 = aie.core(2, 6)
  %27 = aie.core(2, 7)
  %30 = aie.core(3, 0)
  %31 = aie.core(3, 1)
  %32 = aie.core(3, 2)
  %33 = aie.core(3, 3)
    %40 = aie.core(4, 0)
	   %50 = aie.core(5, 0)
  %p0 = aie.plio(0)
  %p1 = aie.plio(1)
  %p2 = aie.plio(2)
  %p3 = aie.plio(3)

  aie.flow(%20, "DMA" : 0, %22, "DMA" : 0)
  aie.flow(%30, "DMA" : 0, %23, "DMA" : 0)
  aie.flow(%40, "DMA" : 0, %24, "DMA" : 0)
  aie.flow(%50, "DMA" : 0, %25, "DMA" : 0)
}
