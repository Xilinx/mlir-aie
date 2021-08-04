//===- routed_herd_3x2.mlir ------------------------------------*- MLIR -*-===//
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
// CHECK: %[[T25:.*]] = AIE.tile(2, 5)
// CHECK: %[[T31:.*]] = AIE.tile(3, 1)
// CHECK: %[[T66:.*]] = AIE.tile(6, 6)
// CHECK: %[[T73:.*]] = AIE.tile(7, 3)
// CHECK: %[[T125:.*]] = AIE.tile(12, 5)
// CHECK: %[[T133:.*]] = AIE.tile(13, 3)
//
// CHECK: AIE.flow(%[[T30]], DMA : 0, %[[T31]], DMA : 0)
// CHECK: AIE.flow(%[[T100]], DMA : 0, %[[T73]], DMA : 0)
// CHECK: AIE.flow(%[[T110]], DMA : 0, %[[T133]], DMA : 0)
//
// CHECK: AIE.flow(%[[T25]], DMA : 0, %[[T60]], DMA : 0)
// CHECK: AIE.flow(%[[T31]], Core : 0, %[[T25]], Core : 0)
// CHECK: AIE.flow(%[[T66]], DMA : 0, %[[T20]], DMA : 0)
//
// CHECK: AIE.flow(%[[T73]], Core : 0, %[[T66]], Core : 0)
// CHECK: AIE.flow(%[[T125]], DMA : 0, %[[T180]], DMA : 0)
// CHECK: AIE.flow(%[[T133]], Core : 0, %[[T125]], Core : 0)


module {
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
	%t05 = AIE.tile(0, 5)
	%t06 = AIE.tile(0, 6)
	%t07 = AIE.tile(0, 7)
	%t08 = AIE.tile(0, 8)
	%t11 = AIE.tile(1, 1)
	%t12 = AIE.tile(1, 2)
	%t13 = AIE.tile(1, 3)
	%t14 = AIE.tile(1, 4)
	%t15 = AIE.tile(1, 5)
	%t16 = AIE.tile(1, 6)
	%t17 = AIE.tile(1, 7)
	%t18 = AIE.tile(1, 8)
	%t21 = AIE.tile(2, 1)
	%t22 = AIE.tile(2, 2)
	%t23 = AIE.tile(2, 3)
	%t24 = AIE.tile(2, 4)
	%t25 = AIE.tile(2, 5)
	%t26 = AIE.tile(2, 6)
	%t27 = AIE.tile(2, 7)
	%t28 = AIE.tile(2, 8)
	%t31 = AIE.tile(3, 1)
	%t32 = AIE.tile(3, 2)
	%t33 = AIE.tile(3, 3)
	%t34 = AIE.tile(3, 4)
	%t35 = AIE.tile(3, 5)
	%t36 = AIE.tile(3, 6)
	%t37 = AIE.tile(3, 7)
	%t38 = AIE.tile(3, 8)
	%t41 = AIE.tile(4, 1)
	%t42 = AIE.tile(4, 2)
	%t43 = AIE.tile(4, 3)
	%t44 = AIE.tile(4, 4)
	%t45 = AIE.tile(4, 5)
	%t46 = AIE.tile(4, 6)
	%t47 = AIE.tile(4, 7)
	%t48 = AIE.tile(4, 8)
	%t51 = AIE.tile(5, 1)
	%t52 = AIE.tile(5, 2)
	%t53 = AIE.tile(5, 3)
	%t54 = AIE.tile(5, 4)
	%t55 = AIE.tile(5, 5)
	%t56 = AIE.tile(5, 6)
	%t57 = AIE.tile(5, 7)
	%t58 = AIE.tile(5, 8)
	%t61 = AIE.tile(6, 1)
	%t62 = AIE.tile(6, 2)
	%t63 = AIE.tile(6, 3)
	%t64 = AIE.tile(6, 4)
	%t65 = AIE.tile(6, 5)
	%t66 = AIE.tile(6, 6)
	%t67 = AIE.tile(6, 7)
	%t68 = AIE.tile(6, 8)
	%t71 = AIE.tile(7, 1)
	%t72 = AIE.tile(7, 2)
	%t73 = AIE.tile(7, 3)
	%t74 = AIE.tile(7, 4)
	%t75 = AIE.tile(7, 5)
	%t76 = AIE.tile(7, 6)
	%t77 = AIE.tile(7, 7)
	%t78 = AIE.tile(7, 8)
	%t81 = AIE.tile(8, 1)
	%t82 = AIE.tile(8, 2)
	%t83 = AIE.tile(8, 3)
	%t84 = AIE.tile(8, 4)
	%t85 = AIE.tile(8, 5)
	%t86 = AIE.tile(8, 6)
	%t87 = AIE.tile(8, 7)
	%t88 = AIE.tile(8, 8)
	%t91 = AIE.tile(9, 1)
	%t92 = AIE.tile(9, 2)
	%t93 = AIE.tile(9, 3)
	%t94 = AIE.tile(9, 4)
	%t95 = AIE.tile(9, 5)
	%t96 = AIE.tile(9, 6)
	%t97 = AIE.tile(9, 7)
	%t98 = AIE.tile(9, 8)
	%t101 = AIE.tile(10, 1)
	%t102 = AIE.tile(10, 2)
	%t103 = AIE.tile(10, 3)
	%t104 = AIE.tile(10, 4)
	%t105 = AIE.tile(10, 5)
	%t106 = AIE.tile(10, 6)
	%t107 = AIE.tile(10, 7)
	%t108 = AIE.tile(10, 8)
	%t111 = AIE.tile(11, 1)
	%t112 = AIE.tile(11, 2)
	%t113 = AIE.tile(11, 3)
	%t114 = AIE.tile(11, 4)
	%t115 = AIE.tile(11, 5)
	%t116 = AIE.tile(11, 6)
	%t117 = AIE.tile(11, 7)
	%t118 = AIE.tile(11, 8)
	%t121 = AIE.tile(12, 1)
	%t122 = AIE.tile(12, 2)
	%t123 = AIE.tile(12, 3)
	%t124 = AIE.tile(12, 4)
	%t125 = AIE.tile(12, 5)
	%t126 = AIE.tile(12, 6)
	%t127 = AIE.tile(12, 7)
	%t128 = AIE.tile(12, 8)
	%t130 = AIE.tile(13, 0)
	%t131 = AIE.tile(13, 1)
	%t132 = AIE.tile(13, 2)
	%t133 = AIE.tile(13, 3)
	%t134 = AIE.tile(13, 4)
	%t135 = AIE.tile(13, 5)
	%t136 = AIE.tile(13, 6)
	%t137 = AIE.tile(13, 7)
	%t138 = AIE.tile(13, 8)
	%t141 = AIE.tile(14, 1)
	%t142 = AIE.tile(14, 2)
	%t143 = AIE.tile(14, 3)
	%t144 = AIE.tile(14, 4)
	%t145 = AIE.tile(14, 5)
	%t146 = AIE.tile(14, 6)
	%t147 = AIE.tile(14, 7)
	%t148 = AIE.tile(14, 8)

	%sb01 = AIE.switchbox(%t01) {
	}
	%sb02 = AIE.switchbox(%t02) {
	}
	%sb03 = AIE.switchbox(%t03) {
	}
	%sb04 = AIE.switchbox(%t04) {
	}
	%sb11 = AIE.switchbox(%t11) {
	}
	%sb12 = AIE.switchbox(%t12) {
	}
	%sb13 = AIE.switchbox(%t13) {
	}
	%sb14 = AIE.switchbox(%t14) {
	}
	%sb21 = AIE.switchbox(%t21) {
	}
	%sb22 = AIE.switchbox(%t22) {
	}
	%sb23 = AIE.switchbox(%t23) {
	}
	%sb24 = AIE.switchbox(%t24) {
		AIE.connect<East : 0, North : 0>
	}
	%sb25 = AIE.switchbox(%t25) {
		AIE.connect<South: 0, Core : 0>
		AIE.connect<DMA : 0, East : 0>
	}
	%sb31 = AIE.switchbox(%t31) {
		AIE.connect<South : 0, DMA : 0>
		AIE.connect<Core : 0, North: 0>
	}
	%sb32 = AIE.switchbox(%t32) {
		AIE.connect<South : 0, North : 0>
	}
	%sb33 = AIE.switchbox(%t33) {
		AIE.connect<South : 0, North : 0>
	}
	%sb34 = AIE.switchbox(%t34) {
		AIE.connect<South : 0, West : 0>
	}
	%sb35 = AIE.switchbox(%t35) {
		AIE.connect<West : 0, East : 0>
	}
	%sb41 = AIE.switchbox(%t41) {
	}
	%sb42 = AIE.switchbox(%t42) {
	}
	%sb43 = AIE.switchbox(%t43) {
	}
	%sb44 = AIE.switchbox(%t44) {
	}
	%sb51 = AIE.switchbox(%t51) {
	}
	%sb52 = AIE.switchbox(%t52) {
	}
	%sb53 = AIE.switchbox(%t53) {
	}
	%sb54 = AIE.switchbox(%t54) {
	}
	%sb55 = AIE.switchbox(%t55) {
	}
	%sb56 = AIE.switchbox(%t56) {
		AIE.connect<East : 0, West : 0>
	}
	%sb61 = AIE.switchbox(%t61) {
	}
	%sb62 = AIE.switchbox(%t62) {
	}
	%sb63 = AIE.switchbox(%t63) {
	}
	%sb64 = AIE.switchbox(%t64) {
	}
	%sb65 = AIE.switchbox(%t65) {
	}
	%sb66 = AIE.switchbox(%t66) {
		AIE.connect<East : 0, Core : 0>
		AIE.connect<DMA : 0, West : 0>
	}
	%sb71 = AIE.switchbox(%t71) {
	}
	%sb72 = AIE.switchbox(%t72) {
	}
	%sb73 = AIE.switchbox(%t73) {
		AIE.connect<East : 0, DMA : 0>
		AIE.connect<Core : 0, North : 0>
	}
	%sb74 = AIE.switchbox(%t74) {
		AIE.connect<South : 0, North : 0>
	}
	%sb75 = AIE.switchbox(%t75) {
		AIE.connect<South : 0, North : 0>
	}
	%sb76 = AIE.switchbox(%t76) {
		AIE.connect<South : 0, West: 0>
	}
	%sb81 = AIE.switchbox(%t81) {
	}
	%sb82 = AIE.switchbox(%t82) {
	}
	%sb83 = AIE.switchbox(%t83) {
		AIE.connect<East : 0, West : 0>
	}
	%sb84 = AIE.switchbox(%t84) {
	}
	%sb91 = AIE.switchbox(%t91) {
	}
	%sb92 = AIE.switchbox(%t92) {
	}
	%sb93 = AIE.switchbox(%t93) {
	}
	%sb94 = AIE.switchbox(%t94) {
	}
	%sb101 = AIE.switchbox(%t101) {
	}
	%sb102 = AIE.switchbox(%t102) {
	}
	%sb103 = AIE.switchbox(%t103) {
	}
	%sb104 = AIE.switchbox(%t104) {
	}
	%sb111 = AIE.switchbox(%t111) {
	}
	%sb112 = AIE.switchbox(%t112) {
	}
	%sb113 = AIE.switchbox(%t113) {
	}
	%sb114 = AIE.switchbox(%t114) {
	}
	%sb121 = AIE.switchbox(%t121) {
	}
	%sb122 = AIE.switchbox(%t122) {
	}
	%sb123 = AIE.switchbox(%t123) {
	}
	%sb124 = AIE.switchbox(%t124) {
	}
	%sb125 = AIE.switchbox(%t125) {
		AIE.connect<East : 0, Core : 0>
		AIE.connect<DMA : 0, East : 0>
	}
	%sb131 = AIE.switchbox(%t131) {
		AIE.connect<South : 0, North : 0>
	}
	%sb132 = AIE.switchbox(%t132) {
		AIE.connect<South : 0, North : 0>
	}
	%sb133 = AIE.switchbox(%t133) {
		AIE.connect<South : 0, DMA : 0>
		AIE.connect<Core : 0, North: 0>
	}
	%sb134 = AIE.switchbox(%t134) {
		AIE.connect<South : 0, North : 0>
	}
	%sb135 = AIE.switchbox(%t135) {
		AIE.connect<South : 0, West : 0>
		AIE.connect<West : 0, East : 0>
	}

  AIE.flow(%t30, DMA : 0, %t30, North: 0)
  AIE.flow(%t45, West: 0, %t60, DMA : 0)

  AIE.flow(%t100, DMA : 0, %t93, West: 0)
  AIE.flow(%t46, East: 0, %t20, DMA : 0)

  AIE.flow(%t110, DMA : 0, %t130, North: 0)
  AIE.flow(%t145, West: 0, %t180, DMA : 0)
}