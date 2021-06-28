// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
//CHECK: %[[t20:.*]] = AIE.tile(2, 0)
//CHECK: %[[t30:.*]] = AIE.tile(3, 0)
//CHECK: %[[t34:.*]] = AIE.tile(3, 4)
//CHECK: %[[t43:.*]] = AIE.tile(4, 3)
//CHECK: %[[t44:.*]] = AIE.tile(4, 4)
//CHECK: %[[t54:.*]] = AIE.tile(5, 4)
//CHECK: %[[t60:.*]] = AIE.tile(6, 0)
//CHECK: %[[t63:.*]] = AIE.tile(6, 3)
//CHECK: %[[t70:.*]] = AIE.tile(7, 0)
//CHECK: %[[t72:.*]] = AIE.tile(7, 2)
//CHECK: %[[t83:.*]] = AIE.tile(8, 3)
//CHECK: %[[t84:.*]] = AIE.tile(8, 4)

// CHECK: AIE.flow(%[[t20]], DMA : 0, %[[t63]], DMA : 0)
// CHECK: AIE.flow(%[[t20]], DMA : 1, %[[t83]], DMA : 0)
// CHECK: AIE.flow(%[[t30]], DMA : 0, %[[t72]], DMA : 0)
// CHECK: AIE.flow(%[[t30]], DMA : 1, %[[t54]], DMA : 0)

// CHECK: AIE.flow(%[[t34]], Core : 0, %[[t63]], Core : 1)
// CHECK: AIE.flow(%[[t34]], DMA : 1, %[[t70]], DMA : 0)
// CHECK: AIE.flow(%[[t43]], Core : 0, %[[t84]], Core : 1)
// CHECK: AIE.flow(%[[t43]], DMA : 1, %[[t60]], DMA : 1)

// CHECK: AIE.flow(%[[t44]], Core : 0, %[[t54]], Core : 1)
// CHECK: AIE.flow(%[[t44]], DMA : 1, %[[t60]], DMA : 0)
// CHECK: AIE.flow(%[[t54]], Core : 0, %[[t43]], Core : 1)
// CHECK: AIE.flow(%[[t54]], DMA : 1, %[[t30]], DMA : 1)

// CHECK: AIE.flow(%[[t60]], DMA : 0, %[[t44]], DMA : 0)
// CHECK: AIE.flow(%[[t60]], DMA : 1, %[[t43]], DMA : 0)
// CHECK: AIE.flow(%[[t63]], Core : 0, %[[t34]], Core : 1)
// CHECK: AIE.flow(%[[t63]], DMA : 1, %[[t20]], DMA : 1)

// CHECK: AIE.flow(%[[t70]], DMA : 0, %[[t34]], DMA : 0)
// CHECK: AIE.flow(%[[t70]], DMA : 1, %[[t84]], DMA : 0)
// CHECK: AIE.flow(%[[t72]], Core : 0, %[[t83]], Core : 1)
// CHECK: AIE.flow(%[[t72]], DMA : 1, %[[t30]], DMA : 0)

// CHECK: AIE.flow(%[[t83]], Core : 0, %[[t44]], Core : 1)
// CHECK: AIE.flow(%[[t83]], DMA : 1, %[[t20]], DMA : 0)
// CHECK: AIE.flow(%[[t84]], Core : 0, %[[t72]], Core : 1)
// CHECK: AIE.flow(%[[t84]], DMA : 1, %[[t70]], DMA : 1)


module {
%t20 = AIE.tile(2, 0)
%t30 = AIE.tile(3, 0)
%t34 = AIE.tile(3, 4)
%t43 = AIE.tile(4, 3)
%t44 = AIE.tile(4, 4)
%t54 = AIE.tile(5, 4)
%t60 = AIE.tile(6, 0)
%t63 = AIE.tile(6, 3)
%t70 = AIE.tile(7, 0)
%t72 = AIE.tile(7, 2)
%t83 = AIE.tile(8, 3)
%t84 = AIE.tile(8, 4)

AIE.flow(%t20, DMA : 0, %t63, DMA : 0)
AIE.flow(%t20, DMA : 1, %t83, DMA : 0)
AIE.flow(%t30, DMA : 0, %t72, DMA : 0)
AIE.flow(%t30, DMA : 1, %t54, DMA : 0)

AIE.flow(%t34, Core : 0, %t63, Core : 1)
AIE.flow(%t34, DMA : 1, %t70, DMA : 0)
AIE.flow(%t43, Core : 0, %t84, Core : 1)
AIE.flow(%t43, DMA : 1, %t60, DMA : 1)

AIE.flow(%t44, Core : 0, %t54, Core : 1)
AIE.flow(%t44, DMA : 1, %t60, DMA : 0)
AIE.flow(%t54, Core : 0, %t43, Core : 1)
AIE.flow(%t54, DMA : 1, %t30, DMA : 1)

AIE.flow(%t60, DMA : 0, %t44, DMA : 0)
AIE.flow(%t60, DMA : 1, %t43, DMA : 0)
AIE.flow(%t63, Core : 0, %t34, Core : 1)
AIE.flow(%t63, DMA : 1, %t20, DMA : 1)

AIE.flow(%t70, DMA : 0, %t34, DMA : 0)
AIE.flow(%t70, DMA : 1, %t84, DMA : 0)
AIE.flow(%t72, Core : 0, %t83, Core : 1)
AIE.flow(%t72, DMA : 1, %t30, DMA : 0)

AIE.flow(%t83, Core : 0, %t44, Core : 1)
AIE.flow(%t83, DMA : 1, %t20, DMA : 0)
AIE.flow(%t84, Core : 0, %t72, Core : 1)
AIE.flow(%t84, DMA : 1, %t70, DMA : 1)
}