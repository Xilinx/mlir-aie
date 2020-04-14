
// arbiter() {
// %1 = masterset(north:1, east:2);
// }
// packetrules(east:1) {
// rule(10000|getRow(), %1);
// }
// for(i: 1:8)
// for(j: 1:50) {
//   out[i][j] = aie.core()
//   }
//   flow(out[i][4], out[i][6])
//   }
module {


  %20 = aie.core(2, 0)
  %21 = aie.core(2, 1)
  %22 = aie.core(2, 2)
  %23 = aie.core(2, 3)
  %24 = aie.core(2, 4)
  %25 = aie.core(2, 5)
  %30 = aie.core(3, 0)
  %31 = aie.core(3, 1)
  %32 = aie.core(3, 2)
  %33 = aie.core(3, 3)
  %34 = aie.core(3, 4)
  %35 = aie.core(3, 5)
    %40 = aie.core(4, 0)
  %41 = aie.core(4, 1)
  %42 = aie.core(4, 2)
  %43 = aie.core(4, 3)
  %44 = aie.core(4, 4)
  %45 = aie.core(4, 5)
	   %50 = aie.core(5, 0)
  %51 = aie.core(5, 1)
  %52 = aie.core(5, 2)
  %53 = aie.core(5, 3)
  %54 = aie.core(5, 4)
  %55 = aie.core(5, 5)
  %p0 = aie.plio(0)
  %p1 = aie.plio(1)
  %p2 = aie.plio(2)
  %p3 = aie.plio(3)

  aie.flow(%20, "DMA" : 0, %22, "DMA" : 1)
  aie.flow(%21, "DMA" : 0, %22, "DMA" : 0)
  aie.flow(%21, "DMA" : 0, %23, "DMA" : 0)
  aie.flow(%21, "DMA" : 0, %24, "DMA" : 0)
  aie.flow(%21, "DMA" : 0, %25, "DMA" : 0)
  aie.flow(%30, "DMA" : 0, %23, "DMA" : 1)
  aie.flow(%31, "DMA" : 0, %32, "DMA" : 0)
  aie.flow(%31, "DMA" : 0, %33, "DMA" : 0)
  aie.flow(%31, "DMA" : 0, %34, "DMA" : 0)
  aie.flow(%31, "DMA" : 0, %35, "DMA" : 0)
  aie.flow(%40, "DMA" : 0, %24, "DMA" : 1)
  aie.flow(%41, "DMA" : 0, %42, "DMA" : 0)
  aie.flow(%41, "DMA" : 0, %43, "DMA" : 0)
  aie.flow(%41, "DMA" : 0, %44, "DMA" : 0)
  aie.flow(%41, "DMA" : 0, %45, "DMA" : 0)
  aie.flow(%50, "DMA" : 0, %25, "DMA" : 1)
  aie.flow(%51, "DMA" : 0, %52, "DMA" : 0)
  aie.flow(%51, "DMA" : 0, %53, "DMA" : 0)
  aie.flow(%51, "DMA" : 0, %54, "DMA" : 0)
  aie.flow(%51, "DMA" : 0, %55, "DMA" : 0)
}
