//===- routed_herd_3x1.mlir ------------------------------------*- MLIR -*-===//
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
// CHECK1: %[[T03:.*]] = aie.tile(0, 3)
// CHECK1: %[[T14:.*]] = aie.tile(1, 4)
// CHECK1: %[[T33:.*]] = aie.tile(3, 3)
// CHECK1: %[[T42:.*]] = aie.tile(4, 2)
// CHECK1: %[[T53:.*]] = aie.tile(5, 3)
// CHECK1: %[[T63:.*]] = aie.tile(6, 3)
// CHECK1: %[[T74:.*]] = aie.tile(7, 4)
// CHECK1: %[[T92:.*]] = aie.tile(9, 2)
// CHECK1: %[[T102:.*]] = aie.tile(10, 2)
// CHECK1: %[[T113:.*]] = aie.tile(11, 3)
//
// CHECK1: aie.flow(%[[T20]], DMA : 0, %[[T14]], DMA : 0)
// CHECK1: aie.flow(%[[T20]], DMA : 1, %[[T63]], DMA : 1)
// CHECK1: aie.flow(%[[T30]], DMA : 0, %[[T33]], DMA : 0)
// CHECK1: aie.flow(%[[T30]], DMA : 1, %[[T74]], DMA : 1)
// CHECK1: aie.flow(%[[T60]], DMA : 0, %[[T03]], DMA : 0)
// CHECK1: aie.flow(%[[T60]], DMA : 1, %[[T42]], DMA : 0)
// CHECK1: aie.flow(%[[T70]], DMA : 0, %[[T03]], DMA : 1)
// CHECK1: aie.flow(%[[T70]], DMA : 1, %[[T53]], DMA : 0)
// CHECK1: aie.flow(%[[T100]], DMA : 0, %[[T102]], DMA : 0)
// CHECK1: aie.flow(%[[T110]], DMA : 0, %[[T113]], DMA : 0)
// CHECK1: aie.flow(%[[T180]], DMA : 0, %[[T63]], DMA : 0)
// CHECK1: aie.flow(%[[T180]], DMA : 1, %[[T92]], DMA : 0)
// CHECK1: aie.flow(%[[T190]], DMA : 0, %[[T74]], DMA : 0)
// CHECK1: aie.flow(%[[T190]], DMA : 1, %[[T113]], DMA : 1)

// CHECK2: "total_path_length": 109

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
		%t11 = aie.tile(1, 1)
		%t12 = aie.tile(1, 2)
		%t13 = aie.tile(1, 3)
		%t14 = aie.tile(1, 4)
		%t21 = aie.tile(2, 1)
		%t22 = aie.tile(2, 2)
		%t23 = aie.tile(2, 3)
		%t24 = aie.tile(2, 4)
		%t31 = aie.tile(3, 1)
		%t32 = aie.tile(3, 2)
		%t33 = aie.tile(3, 3)
		%t34 = aie.tile(3, 4)
		%t41 = aie.tile(4, 1)
		%t42 = aie.tile(4, 2)
		%t43 = aie.tile(4, 3)
		%t44 = aie.tile(4, 4)
		%t51 = aie.tile(5, 1)
		%t52 = aie.tile(5, 2)
		%t53 = aie.tile(5, 3)
		%t54 = aie.tile(5, 4)
		%t61 = aie.tile(6, 1)
		%t62 = aie.tile(6, 2)
		%t63 = aie.tile(6, 3)
		%t64 = aie.tile(6, 4)
		%t71 = aie.tile(7, 1)
		%t72 = aie.tile(7, 2)
		%t73 = aie.tile(7, 3)
		%t74 = aie.tile(7, 4)
		%t81 = aie.tile(8, 1)
		%t82 = aie.tile(8, 2)
		%t83 = aie.tile(8, 3)
		%t84 = aie.tile(8, 4)
		%t91 = aie.tile(9, 1)
		%t92 = aie.tile(9, 2)
		%t93 = aie.tile(9, 3)
		%t94 = aie.tile(9, 4)
		%t101 = aie.tile(10, 1)
		%t102 = aie.tile(10, 2)
		%t103 = aie.tile(10, 3)
		%t104 = aie.tile(10, 4)
		%t111 = aie.tile(11, 1)
		%t112 = aie.tile(11, 2)
		%t113 = aie.tile(11, 3)
		%t114 = aie.tile(11, 4)
		%t121 = aie.tile(12, 1)
		%t122 = aie.tile(12, 2)
		%t123 = aie.tile(12, 3)
		%t124 = aie.tile(12, 4)

		%sb01 = aie.switchbox(%t01) {
			aie.connect<South : 0, North : 0>
		}
		%sb02 = aie.switchbox(%t02) {
			aie.connect<South : 0, North : 0>
		}
		%sb03 = aie.switchbox(%t03) {
			aie.connect<South : 0, DMA : 0>
			aie.connect<East : 0, DMA : 1>
		}
		%sb04 = aie.switchbox(%t04) {
		}
		%sb11 = aie.switchbox(%t11) {
			aie.connect<South : 0, North : 0>
		}
		%sb12 = aie.switchbox(%t12) {
			aie.connect<South : 0, North : 0>
		}
		%sb13 = aie.switchbox(%t13) {
			aie.connect<South : 0, West : 0>
		}
		%sb14 = aie.switchbox(%t14) {
			aie.connect<East : 0, DMA : 0>
		}
		%sb21 = aie.switchbox(%t21) {
			aie.connect<South : 0, North : 0>
		}
		%sb22 = aie.switchbox(%t22) {
			aie.connect<South : 0, North : 0>
		}
		%sb23 = aie.switchbox(%t23) {
			aie.connect<South : 0, North : 0>
		}
		%sb24 = aie.switchbox(%t24) {
			aie.connect<South : 0, West : 0>
		}
		%sb31 = aie.switchbox(%t31) {
			aie.connect<South : 0, North : 0>
		}
		%sb32 = aie.switchbox(%t32) {
			aie.connect<South : 0, North : 0>
		}
		%sb33 = aie.switchbox(%t33) {
			aie.connect<South : 0, DMA : 0>
		}
		%sb34 = aie.switchbox(%t34) {
		}
		%sb41 = aie.switchbox(%t41) {
			aie.connect<South : 0, North : 0>
		}
		%sb42 = aie.switchbox(%t42) {
			aie.connect<South : 0, DMA : 0>
		}
		%sb43 = aie.switchbox(%t43) {
		}
		%sb44 = aie.switchbox(%t44) {
		}
		%sb51 = aie.switchbox(%t51) {
			aie.connect<South : 0, North : 0>
		}
		%sb52 = aie.switchbox(%t52) {
			aie.connect<South : 0, North : 0>
		}
		%sb53 = aie.switchbox(%t53) {
			aie.connect<South : 0, DMA : 0>
		}
		%sb54 = aie.switchbox(%t54) {
		}
		%sb61 = aie.switchbox(%t61) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb62 = aie.switchbox(%t62) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb63 = aie.switchbox(%t63) {
			aie.connect<South : 0, DMA : 0>
			aie.connect<South : 1, DMA : 1>
		}
		%sb64 = aie.switchbox(%t64) {
		}
		%sb71 = aie.switchbox(%t71) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb72 = aie.switchbox(%t72) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb73 = aie.switchbox(%t73) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb74 = aie.switchbox(%t74) {
			aie.connect<South : 0, DMA : 0>
			aie.connect<South : 1, DMA : 1>
		}
		%sb81 = aie.switchbox(%t81) {
		}
		%sb82 = aie.switchbox(%t82) {
		}
		%sb83 = aie.switchbox(%t83) {
		}
		%sb84 = aie.switchbox(%t84) {
		}
		%sb91 = aie.switchbox(%t91) {
			aie.connect<South : 0, North : 0>
		}
		%sb92 = aie.switchbox(%t92) {
			aie.connect<South : 0, DMA : 0>
		}
		%sb93 = aie.switchbox(%t93) {
		}
		%sb94 = aie.switchbox(%t94) {
		}
		%sb101 = aie.switchbox(%t101) {
			aie.connect<South : 0, North : 0>
		}
		%sb102 = aie.switchbox(%t102) {
			aie.connect<South : 0, DMA : 0>
		}
		%sb103 = aie.switchbox(%t103) {
		}
		%sb104 = aie.switchbox(%t104) {
		}
		%sb111 = aie.switchbox(%t111) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb112 = aie.switchbox(%t112) {
			aie.connect<South : 0, North : 0>
			aie.connect<South : 1, North : 1>
		}
		%sb113 = aie.switchbox(%t113) {
			aie.connect<South : 0, DMA : 0>
			aie.connect<South : 1, DMA : 1>
		}
		%sb114 = aie.switchbox(%t114) {
		}

		aie.flow(%t20, DMA : 0, %t20, North: 0)
		aie.flow(%t20, DMA : 1, %t60, North: 1)
		aie.flow(%t30, DMA : 0, %t30, North: 0)
		aie.flow(%t30, DMA : 1, %t70, North: 1)
		aie.flow(%t60, DMA : 0, %t00, North: 0)
		aie.flow(%t60, DMA : 1, %t40, North: 0)
		aie.flow(%t70, DMA : 0, %t10, North: 0)
		aie.flow(%t70, DMA : 1, %t50, North: 0)
		aie.flow(%t100, DMA : 0, %t100, North: 0)
		aie.flow(%t110, DMA : 0, %t110, North: 0)
		aie.flow(%t180, DMA : 0, %t60, North: 0)
		aie.flow(%t180, DMA : 1, %t90, North: 0)
		aie.flow(%t190, DMA : 0, %t70, North: 0)
		aie.flow(%t190, DMA : 1, %t110, North: 1)
	}
}