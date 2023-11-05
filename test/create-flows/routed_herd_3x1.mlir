//===- routed_herd_3x1.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T30:.*]] = AIE.tile(3, 0)
// CHECK: %[[T60:.*]] = AIE.tile(6, 0)
// CHECK: %[[T70:.*]] = AIE.tile(7, 0)
// CHECK: %[[T100:.*]] = AIE.tile(10, 0)
// CHECK: %[[T110:.*]] = AIE.tile(11, 0)
// CHECK: %[[T180:.*]] = AIE.tile(18, 0)
// CHECK: %[[T190:.*]] = AIE.tile(19, 0)
// CHECK: %[[T03:.*]] = AIE.tile(0, 3)
// CHECK: %[[T14:.*]] = AIE.tile(1, 4)
// CHECK: %[[T33:.*]] = AIE.tile(3, 3)
// CHECK: %[[T42:.*]] = AIE.tile(4, 2)
// CHECK: %[[T53:.*]] = AIE.tile(5, 3)
// CHECK: %[[T63:.*]] = AIE.tile(6, 3)
// CHECK: %[[T74:.*]] = AIE.tile(7, 4)
// CHECK: %[[T92:.*]] = AIE.tile(9, 2)
// CHECK: %[[T102:.*]] = AIE.tile(10, 2)
// CHECK: %[[T113:.*]] = AIE.tile(11, 3)
//
// CHECK: AIE.flow(%[[T20]], DMA : 0, %[[T14]], DMA : 0)
// CHECK: AIE.flow(%[[T20]], DMA : 1, %[[T63]], DMA : 1)
// CHECK: AIE.flow(%[[T30]], DMA : 0, %[[T33]], DMA : 0)
// CHECK: AIE.flow(%[[T30]], DMA : 1, %[[T74]], DMA : 1)
// CHECK: AIE.flow(%[[T60]], DMA : 0, %[[T03]], DMA : 0)
// CHECK: AIE.flow(%[[T60]], DMA : 1, %[[T42]], DMA : 0)
// CHECK: AIE.flow(%[[T70]], DMA : 0, %[[T03]], DMA : 1)
// CHECK: AIE.flow(%[[T70]], DMA : 1, %[[T53]], DMA : 0)
// CHECK: AIE.flow(%[[T100]], DMA : 0, %[[T102]], DMA : 0)
// CHECK: AIE.flow(%[[T110]], DMA : 0, %[[T113]], DMA : 0)
// CHECK: AIE.flow(%[[T180]], DMA : 0, %[[T63]], DMA : 0)
// CHECK: AIE.flow(%[[T180]], DMA : 1, %[[T92]], DMA : 0)
// CHECK: AIE.flow(%[[T190]], DMA : 0, %[[T74]], DMA : 0)
// CHECK: AIE.flow(%[[T190]], DMA : 1, %[[T113]], DMA : 1)

module {
	AIE.device(xcvc1902) {
		%t00 = AIE.tile(0, 0)
		%t10 = AIE.tile(1, 0)
		%t20 = AIE.tile(2, 0)
		%t30 = AIE.tile(3, 0)
		%t40 = AIE.tile(4, 0)
		%t50 = AIE.tile(5, 0)
		%t60 = AIE.tile(6, 0)
		%t70 = AIE.tile(7, 0)
		%t80 = AIE.tile(8, 0)
		%t90 = AIE.tile(9, 0)
		%t100 = AIE.tile(10, 0)
		%t110 = AIE.tile(11, 0)
		%t180 = AIE.tile(18, 0)
		%t190 = AIE.tile(19, 0)

		%t01 = AIE.tile(0, 1)
		%t02 = AIE.tile(0, 2)
		%t03 = AIE.tile(0, 3)
		%t04 = AIE.tile(0, 4)
		%t11 = AIE.tile(1, 1)
		%t12 = AIE.tile(1, 2)
		%t13 = AIE.tile(1, 3)
		%t14 = AIE.tile(1, 4)
		%t21 = AIE.tile(2, 1)
		%t22 = AIE.tile(2, 2)
		%t23 = AIE.tile(2, 3)
		%t24 = AIE.tile(2, 4)
		%t31 = AIE.tile(3, 1)
		%t32 = AIE.tile(3, 2)
		%t33 = AIE.tile(3, 3)
		%t34 = AIE.tile(3, 4)
		%t41 = AIE.tile(4, 1)
		%t42 = AIE.tile(4, 2)
		%t43 = AIE.tile(4, 3)
		%t44 = AIE.tile(4, 4)
		%t51 = AIE.tile(5, 1)
		%t52 = AIE.tile(5, 2)
		%t53 = AIE.tile(5, 3)
		%t54 = AIE.tile(5, 4)
		%t61 = AIE.tile(6, 1)
		%t62 = AIE.tile(6, 2)
		%t63 = AIE.tile(6, 3)
		%t64 = AIE.tile(6, 4)
		%t71 = AIE.tile(7, 1)
		%t72 = AIE.tile(7, 2)
		%t73 = AIE.tile(7, 3)
		%t74 = AIE.tile(7, 4)
		%t81 = AIE.tile(8, 1)
		%t82 = AIE.tile(8, 2)
		%t83 = AIE.tile(8, 3)
		%t84 = AIE.tile(8, 4)
		%t91 = AIE.tile(9, 1)
		%t92 = AIE.tile(9, 2)
		%t93 = AIE.tile(9, 3)
		%t94 = AIE.tile(9, 4)
		%t101 = AIE.tile(10, 1)
		%t102 = AIE.tile(10, 2)
		%t103 = AIE.tile(10, 3)
		%t104 = AIE.tile(10, 4)
		%t111 = AIE.tile(11, 1)
		%t112 = AIE.tile(11, 2)
		%t113 = AIE.tile(11, 3)
		%t114 = AIE.tile(11, 4)
		%t121 = AIE.tile(12, 1)
		%t122 = AIE.tile(12, 2)
		%t123 = AIE.tile(12, 3)
		%t124 = AIE.tile(12, 4)

		%sb01 = AIE.switchbox(%t01) {
			AIE.connect<South : 0, North : 0>
		}
		%sb02 = AIE.switchbox(%t02) {
			AIE.connect<South : 0, North : 0>
		}
		%sb03 = AIE.switchbox(%t03) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<East : 0, DMA : 1>
		}
		%sb04 = AIE.switchbox(%t04) {
		}
		%sb11 = AIE.switchbox(%t11) {
			AIE.connect<South : 0, North : 0>
		}
		%sb12 = AIE.switchbox(%t12) {
			AIE.connect<South : 0, North : 0>
		}
		%sb13 = AIE.switchbox(%t13) {
			AIE.connect<South : 0, West : 0>
		}
		%sb14 = AIE.switchbox(%t14) {
			AIE.connect<East : 0, DMA : 0>
		}
		%sb21 = AIE.switchbox(%t21) {
			AIE.connect<South : 0, North : 0>
		}
		%sb22 = AIE.switchbox(%t22) {
			AIE.connect<South : 0, North : 0>
		}
		%sb23 = AIE.switchbox(%t23) {
			AIE.connect<South : 0, North : 0>
		}
		%sb24 = AIE.switchbox(%t24) {
			AIE.connect<South : 0, West : 0>
		}
		%sb31 = AIE.switchbox(%t31) {
			AIE.connect<South : 0, North : 0>
		}
		%sb32 = AIE.switchbox(%t32) {
			AIE.connect<South : 0, North : 0>
		}
		%sb33 = AIE.switchbox(%t33) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb34 = AIE.switchbox(%t34) {
		}
		%sb41 = AIE.switchbox(%t41) {
			AIE.connect<South : 0, North : 0>
		}
		%sb42 = AIE.switchbox(%t42) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb43 = AIE.switchbox(%t43) {
		}
		%sb44 = AIE.switchbox(%t44) {
		}
		%sb51 = AIE.switchbox(%t51) {
			AIE.connect<South : 0, North : 0>
		}
		%sb52 = AIE.switchbox(%t52) {
			AIE.connect<South : 0, North : 0>
		}
		%sb53 = AIE.switchbox(%t53) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb54 = AIE.switchbox(%t54) {
		}
		%sb61 = AIE.switchbox(%t61) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb62 = AIE.switchbox(%t62) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb63 = AIE.switchbox(%t63) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<South : 1, DMA : 1>
		}
		%sb64 = AIE.switchbox(%t64) {
		}
		%sb71 = AIE.switchbox(%t71) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb72 = AIE.switchbox(%t72) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb73 = AIE.switchbox(%t73) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb74 = AIE.switchbox(%t74) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<South : 1, DMA : 1>
		}
		%sb81 = AIE.switchbox(%t81) {
		}
		%sb82 = AIE.switchbox(%t82) {
		}
		%sb83 = AIE.switchbox(%t83) {
		}
		%sb84 = AIE.switchbox(%t84) {
		}
		%sb91 = AIE.switchbox(%t91) {
			AIE.connect<South : 0, North : 0>
		}
		%sb92 = AIE.switchbox(%t92) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb93 = AIE.switchbox(%t93) {
		}
		%sb94 = AIE.switchbox(%t94) {
		}
		%sb101 = AIE.switchbox(%t101) {
			AIE.connect<South : 0, North : 0>
		}
		%sb102 = AIE.switchbox(%t102) {
			AIE.connect<South : 0, DMA : 0>
		}
		%sb103 = AIE.switchbox(%t103) {
		}
		%sb104 = AIE.switchbox(%t104) {
		}
		%sb111 = AIE.switchbox(%t111) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb112 = AIE.switchbox(%t112) {
			AIE.connect<South : 0, North : 0>
			AIE.connect<South : 1, North : 1>
		}
		%sb113 = AIE.switchbox(%t113) {
			AIE.connect<South : 0, DMA : 0>
			AIE.connect<South : 1, DMA : 1>
		}
		%sb114 = AIE.switchbox(%t114) {
		}

		AIE.flow(%t20, DMA : 0, %t20, North: 0)
		AIE.flow(%t20, DMA : 1, %t60, North: 1)
		AIE.flow(%t30, DMA : 0, %t30, North: 0)
		AIE.flow(%t30, DMA : 1, %t70, North: 1)
		AIE.flow(%t60, DMA : 0, %t00, North: 0)
		AIE.flow(%t60, DMA : 1, %t40, North: 0)
		AIE.flow(%t70, DMA : 0, %t10, North: 0)
		AIE.flow(%t70, DMA : 1, %t50, North: 0)
		AIE.flow(%t100, DMA : 0, %t100, North: 0)
		AIE.flow(%t110, DMA : 0, %t110, North: 0)
		AIE.flow(%t180, DMA : 0, %t60, North: 0)
		AIE.flow(%t180, DMA : 1, %t90, North: 0)
		AIE.flow(%t190, DMA : 0, %t70, North: 0)
		AIE.flow(%t190, DMA : 1, %t110, North: 1)
	}
}