// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

// arbiter() {
// %1 = masterset(north:1, east:2);
// }
// packetrules(east:1) {
// rule(10000|getRow(), %1);
// }
// for(i: 1:8)
// for(j: 1:50) {
//   out[i][j] = AIE.core()
//   }
//   flow(out[i][4], out[i][6])
//   }
module {


  %20 = AIE.core(2, 0)
  %21 = AIE.core(2, 1)
  %22 = AIE.core(2, 2)
  %23 = AIE.core(2, 3)
  %24 = AIE.core(2, 4)
  %25 = AIE.core(2, 5)
  %30 = AIE.core(3, 0)
  %31 = AIE.core(3, 1)
  %32 = AIE.core(3, 2)
  %33 = AIE.core(3, 3)
  %34 = AIE.core(3, 4)
  %35 = AIE.core(3, 5)
    %40 = AIE.core(4, 0)
  %41 = AIE.core(4, 1)
  %42 = AIE.core(4, 2)
  %43 = AIE.core(4, 3)
  %44 = AIE.core(4, 4)
  %45 = AIE.core(4, 5)
	   %50 = AIE.core(5, 0)
  %51 = AIE.core(5, 1)
  %52 = AIE.core(5, 2)
  %53 = AIE.core(5, 3)
  %54 = AIE.core(5, 4)
  %55 = AIE.core(5, 5)
  %p0 = AIE.plio(0)
  %p1 = AIE.plio(1)
  %p2 = AIE.plio(2)
  %p3 = AIE.plio(3)

  AIE.flow(%20, "DMA" : 0, %22, "DMA" : 1)
  AIE.flow(%21, "DMA" : 0, %22, "DMA" : 0)
  AIE.flow(%21, "DMA" : 0, %23, "DMA" : 0)
  AIE.flow(%21, "DMA" : 0, %24, "DMA" : 0)
  AIE.flow(%21, "DMA" : 0, %25, "DMA" : 0)
  AIE.flow(%30, "DMA" : 0, %23, "DMA" : 1)
  AIE.flow(%31, "DMA" : 0, %32, "DMA" : 0)
  AIE.flow(%31, "DMA" : 0, %33, "DMA" : 0)
  AIE.flow(%31, "DMA" : 0, %34, "DMA" : 0)
  AIE.flow(%31, "DMA" : 0, %35, "DMA" : 0)
  AIE.flow(%40, "DMA" : 0, %24, "DMA" : 1)
  AIE.flow(%41, "DMA" : 0, %42, "DMA" : 0)
  AIE.flow(%41, "DMA" : 0, %43, "DMA" : 0)
  AIE.flow(%41, "DMA" : 0, %44, "DMA" : 0)
  AIE.flow(%41, "DMA" : 0, %45, "DMA" : 0)
  AIE.flow(%50, "DMA" : 0, %25, "DMA" : 1)
  AIE.flow(%51, "DMA" : 0, %52, "DMA" : 0)
  AIE.flow(%51, "DMA" : 0, %53, "DMA" : 0)
  AIE.flow(%51, "DMA" : 0, %54, "DMA" : 0)
  AIE.flow(%51, "DMA" : 0, %55, "DMA" : 0)
}
