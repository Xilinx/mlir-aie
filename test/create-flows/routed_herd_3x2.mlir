//===- routed_herd_3x2.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1: %[[T20:.*]] = aie.tile(2, 0)
// CHECK1: %[[T30:.*]] = aie.tile(3, 0)
// CHECK1: %[[T60:.*]] = aie.tile(6, 0)
// CHECK1: %[[T70:.*]] = aie.tile(7, 0)
// CHECK1: %[[T100:.*]] = aie.tile(10, 0)
// CHECK1: %[[T110:.*]] = aie.tile(11, 0)
// CHECK1: %[[T180:.*]] = aie.tile(18, 0)
// CHECK1: %[[T190:.*]] = aie.tile(19, 0)
// CHECK1: %[[T25:.*]] = aie.tile(2, 5)
// CHECK1: %[[T31:.*]] = aie.tile(3, 1)
// CHECK1: %[[T66:.*]] = aie.tile(6, 6)
// CHECK1: %[[T73:.*]] = aie.tile(7, 3)
// CHECK1: %[[T125:.*]] = aie.tile(12, 5)
// CHECK1: %[[T133:.*]] = aie.tile(13, 3)
//
// CHECK1: aie.flow(%[[T30]], DMA : 0, %[[T31]], DMA : 0)
// CHECK1: aie.flow(%[[T100]], DMA : 0, %[[T73]], DMA : 0)
// CHECK1: aie.flow(%[[T110]], DMA : 0, %[[T133]], DMA : 0)
//
// CHECK1: aie.flow(%[[T25]], DMA : 0, %[[T60]], DMA : 0)
// CHECK1: aie.flow(%[[T31]], Core : 0, %[[T25]], Core : 0)
// CHECK1: aie.flow(%[[T66]], DMA : 0, %[[T20]], DMA : 0)
//
// CHECK1: aie.flow(%[[T73]], Core : 0, %[[T66]], Core : 0)
// CHECK1: aie.flow(%[[T125]], DMA : 0, %[[T180]], DMA : 0)
// CHECK1: aie.flow(%[[T133]], Core : 0, %[[T125]], Core : 0)

// CHECK2: "total_path_length": 54

module {
	aie.device(xcvc1902) {
		%t00 = aie.tile(0, 0)
		%t10 = aie.tile(1, 0)
		%t20 = aie.tile(2, 0)
		%t30 = aie.tile(3, 0)
		%t40 = aie.tile(4, 0)
		%t50 = aie.tile(5, 0)
		%t60 = aie.tile(6, 0)
		%t70 = aie.tile(7, 0)
		%t80 = aie.tile(8, 0)
		%t90 = aie.tile(9, 0)
		%t100 = aie.tile(10, 0)
		%t110 = aie.tile(11, 0)
		%t180 = aie.tile(18, 0)
		%t190 = aie.tile(19, 0)

		%t01 = aie.tile(0, 1)
		%t02 = aie.tile(0, 2)
		%t03 = aie.tile(0, 3)
		%t04 = aie.tile(0, 4)
		%t05 = aie.tile(0, 5)
		%t06 = aie.tile(0, 6)
		%t07 = aie.tile(0, 7)
		%t08 = aie.tile(0, 8)
		%t11 = aie.tile(1, 1)
		%t12 = aie.tile(1, 2)
		%t13 = aie.tile(1, 3)
		%t14 = aie.tile(1, 4)
		%t15 = aie.tile(1, 5)
		%t16 = aie.tile(1, 6)
		%t17 = aie.tile(1, 7)
		%t18 = aie.tile(1, 8)
		%t21 = aie.tile(2, 1)
		%t22 = aie.tile(2, 2)
		%t23 = aie.tile(2, 3)
		%t24 = aie.tile(2, 4)
		%t25 = aie.tile(2, 5)
		%t26 = aie.tile(2, 6)
		%t27 = aie.tile(2, 7)
		%t28 = aie.tile(2, 8)
		%t31 = aie.tile(3, 1)
		%t32 = aie.tile(3, 2)
		%t33 = aie.tile(3, 3)
		%t34 = aie.tile(3, 4)
		%t35 = aie.tile(3, 5)
		%t36 = aie.tile(3, 6)
		%t37 = aie.tile(3, 7)
		%t38 = aie.tile(3, 8)
		%t41 = aie.tile(4, 1)
		%t42 = aie.tile(4, 2)
		%t43 = aie.tile(4, 3)
		%t44 = aie.tile(4, 4)
		%t45 = aie.tile(4, 5)
		%t46 = aie.tile(4, 6)
		%t47 = aie.tile(4, 7)
		%t48 = aie.tile(4, 8)
		%t51 = aie.tile(5, 1)
		%t52 = aie.tile(5, 2)
		%t53 = aie.tile(5, 3)
		%t54 = aie.tile(5, 4)
		%t55 = aie.tile(5, 5)
		%t56 = aie.tile(5, 6)
		%t57 = aie.tile(5, 7)
		%t58 = aie.tile(5, 8)
		%t61 = aie.tile(6, 1)
		%t62 = aie.tile(6, 2)
		%t63 = aie.tile(6, 3)
		%t64 = aie.tile(6, 4)
		%t65 = aie.tile(6, 5)
		%t66 = aie.tile(6, 6)
		%t67 = aie.tile(6, 7)
		%t68 = aie.tile(6, 8)
		%t71 = aie.tile(7, 1)
		%t72 = aie.tile(7, 2)
		%t73 = aie.tile(7, 3)
		%t74 = aie.tile(7, 4)
		%t75 = aie.tile(7, 5)
		%t76 = aie.tile(7, 6)
		%t77 = aie.tile(7, 7)
		%t78 = aie.tile(7, 8)
		%t81 = aie.tile(8, 1)
		%t82 = aie.tile(8, 2)
		%t83 = aie.tile(8, 3)
		%t84 = aie.tile(8, 4)
		%t85 = aie.tile(8, 5)
		%t86 = aie.tile(8, 6)
		%t87 = aie.tile(8, 7)
		%t88 = aie.tile(8, 8)
		%t91 = aie.tile(9, 1)
		%t92 = aie.tile(9, 2)
		%t93 = aie.tile(9, 3)
		%t94 = aie.tile(9, 4)
		%t95 = aie.tile(9, 5)
		%t96 = aie.tile(9, 6)
		%t97 = aie.tile(9, 7)
		%t98 = aie.tile(9, 8)
		%t101 = aie.tile(10, 1)
		%t102 = aie.tile(10, 2)
		%t103 = aie.tile(10, 3)
		%t104 = aie.tile(10, 4)
		%t105 = aie.tile(10, 5)
		%t106 = aie.tile(10, 6)
		%t107 = aie.tile(10, 7)
		%t108 = aie.tile(10, 8)
		%t111 = aie.tile(11, 1)
		%t112 = aie.tile(11, 2)
		%t113 = aie.tile(11, 3)
		%t114 = aie.tile(11, 4)
		%t115 = aie.tile(11, 5)
		%t116 = aie.tile(11, 6)
		%t117 = aie.tile(11, 7)
		%t118 = aie.tile(11, 8)
		%t121 = aie.tile(12, 1)
		%t122 = aie.tile(12, 2)
		%t123 = aie.tile(12, 3)
		%t124 = aie.tile(12, 4)
		%t125 = aie.tile(12, 5)
		%t126 = aie.tile(12, 6)
		%t127 = aie.tile(12, 7)
		%t128 = aie.tile(12, 8)
		%t130 = aie.tile(13, 0)
		%t131 = aie.tile(13, 1)
		%t132 = aie.tile(13, 2)
		%t133 = aie.tile(13, 3)
		%t134 = aie.tile(13, 4)
		%t135 = aie.tile(13, 5)
		%t136 = aie.tile(13, 6)
		%t137 = aie.tile(13, 7)
		%t138 = aie.tile(13, 8)
		%t141 = aie.tile(14, 1)
		%t142 = aie.tile(14, 2)
		%t143 = aie.tile(14, 3)
		%t144 = aie.tile(14, 4)
		%t145 = aie.tile(14, 5)
		%t146 = aie.tile(14, 6)
		%t147 = aie.tile(14, 7)
		%t148 = aie.tile(14, 8)

		%sb01 = aie.switchbox(%t01) {
		}
		%sb02 = aie.switchbox(%t02) {
		}
		%sb03 = aie.switchbox(%t03) {
		}
		%sb04 = aie.switchbox(%t04) {
		}
		%sb11 = aie.switchbox(%t11) {
		}
		%sb12 = aie.switchbox(%t12) {
		}
		%sb13 = aie.switchbox(%t13) {
		}
		%sb14 = aie.switchbox(%t14) {
		}
		%sb21 = aie.switchbox(%t21) {
		}
		%sb22 = aie.switchbox(%t22) {
		}
		%sb23 = aie.switchbox(%t23) {
		}
		%sb24 = aie.switchbox(%t24) {
			aie.connect<East : 0, North : 0>
		}
		%sb25 = aie.switchbox(%t25) {
			aie.connect<South: 0, Core : 0>
			aie.connect<DMA : 0, East : 0>
		}
		%sb31 = aie.switchbox(%t31) {
			aie.connect<South : 0, DMA : 0>
			aie.connect<Core : 0, North: 0>
		}
		%sb32 = aie.switchbox(%t32) {
			aie.connect<South : 0, North : 0>
		}
		%sb33 = aie.switchbox(%t33) {
			aie.connect<South : 0, North : 0>
		}
		%sb34 = aie.switchbox(%t34) {
			aie.connect<South : 0, West : 0>
		}
		%sb35 = aie.switchbox(%t35) {
			aie.connect<West : 0, East : 0>
		}
		%sb41 = aie.switchbox(%t41) {
		}
		%sb42 = aie.switchbox(%t42) {
		}
		%sb43 = aie.switchbox(%t43) {
		}
		%sb44 = aie.switchbox(%t44) {
		}
		%sb51 = aie.switchbox(%t51) {
		}
		%sb52 = aie.switchbox(%t52) {
		}
		%sb53 = aie.switchbox(%t53) {
		}
		%sb54 = aie.switchbox(%t54) {
		}
		%sb55 = aie.switchbox(%t55) {
		}
		%sb56 = aie.switchbox(%t56) {
			aie.connect<East : 0, West : 0>
		}
		%sb61 = aie.switchbox(%t61) {
		}
		%sb62 = aie.switchbox(%t62) {
		}
		%sb63 = aie.switchbox(%t63) {
		}
		%sb64 = aie.switchbox(%t64) {
		}
		%sb65 = aie.switchbox(%t65) {
		}
		%sb66 = aie.switchbox(%t66) {
			aie.connect<East : 0, Core : 0>
			aie.connect<DMA : 0, West : 0>
		}
		%sb71 = aie.switchbox(%t71) {
		}
		%sb72 = aie.switchbox(%t72) {
		}
		%sb73 = aie.switchbox(%t73) {
			aie.connect<East : 0, DMA : 0>
			aie.connect<Core : 0, North : 0>
		}
		%sb74 = aie.switchbox(%t74) {
			aie.connect<South : 0, North : 0>
		}
		%sb75 = aie.switchbox(%t75) {
			aie.connect<South : 0, North : 0>
		}
		%sb76 = aie.switchbox(%t76) {
			aie.connect<South : 0, West: 0>
		}
		%sb81 = aie.switchbox(%t81) {
		}
		%sb82 = aie.switchbox(%t82) {
		}
		%sb83 = aie.switchbox(%t83) {
			aie.connect<East : 0, West : 0>
		}
		%sb84 = aie.switchbox(%t84) {
		}
		%sb91 = aie.switchbox(%t91) {
		}
		%sb92 = aie.switchbox(%t92) {
		}
		%sb93 = aie.switchbox(%t93) {
		}
		%sb94 = aie.switchbox(%t94) {
		}
		%sb101 = aie.switchbox(%t101) {
		}
		%sb102 = aie.switchbox(%t102) {
		}
		%sb103 = aie.switchbox(%t103) {
		}
		%sb104 = aie.switchbox(%t104) {
		}
		%sb111 = aie.switchbox(%t111) {
		}
		%sb112 = aie.switchbox(%t112) {
		}
		%sb113 = aie.switchbox(%t113) {
		}
		%sb114 = aie.switchbox(%t114) {
		}
		%sb121 = aie.switchbox(%t121) {
		}
		%sb122 = aie.switchbox(%t122) {
		}
		%sb123 = aie.switchbox(%t123) {
		}
		%sb124 = aie.switchbox(%t124) {
		}
		%sb125 = aie.switchbox(%t125) {
			aie.connect<East : 0, Core : 0>
			aie.connect<DMA : 0, East : 0>
		}
		%sb131 = aie.switchbox(%t131) {
			aie.connect<South : 0, North : 0>
		}
		%sb132 = aie.switchbox(%t132) {
			aie.connect<South : 0, North : 0>
		}
		%sb133 = aie.switchbox(%t133) {
			aie.connect<South : 0, DMA : 0>
			aie.connect<Core : 0, North: 0>
		}
		%sb134 = aie.switchbox(%t134) {
			aie.connect<South : 0, North : 0>
		}
		%sb135 = aie.switchbox(%t135) {
			aie.connect<South : 0, West : 0>
			aie.connect<West : 0, East : 0>
		}

		aie.flow(%t30, DMA : 0, %t30, North: 0)
		aie.flow(%t45, West: 0, %t60, DMA : 0)

		aie.flow(%t100, DMA : 0, %t93, West: 0)
		aie.flow(%t46, East: 0, %t20, DMA : 0)

		aie.flow(%t110, DMA : 0, %t130, North: 0)
		aie.flow(%t145, West: 0, %t180, DMA : 0)
	}
}