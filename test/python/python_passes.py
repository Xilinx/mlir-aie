# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes


import aie
from aie.ir import *
from aie.dialects.aie import *
from aie.passmanager import PassManager
from aie._mlir_libs import _aie_python_passes

from typing import List


def testPathfinderFlowsWithPython():
    print("\nTEST: testPathfinderFlowsWithPython")

    def print_hello(max_col, max_row, is_legal):
        print(f"{max_col=} {max_row=} {is_legal=}")
        print("hello from PythonDynamicTileAnalysis::runAnalysis!")

    module = """
    AIE.device(xcvc1902) {
        %t03 = AIE.tile(0, 3)
        %t02 = AIE.tile(0, 2)
        %t00 = AIE.tile(0, 0)
        %t13 = AIE.tile(1, 3)
        %t11 = AIE.tile(1, 1)
        %t10 = AIE.tile(1, 0)
        %t20 = AIE.tile(2, 0)
        %t30 = AIE.tile(3, 0)
        %t22 = AIE.tile(2, 2)
        %t31 = AIE.tile(3, 1)
        %t60 = AIE.tile(6, 0)
        %t70 = AIE.tile(7, 0)
        %t71 = AIE.tile(7, 1)
        %t72 = AIE.tile(7, 2)
        %t73 = AIE.tile(7, 3)
        %t80 = AIE.tile(8, 0)
        %t82 = AIE.tile(8, 2)
        %t83 = AIE.tile(8, 3)

        AIE.flow(%t20, DMA : 0, %t13, DMA : 0)
        AIE.flow(%t20, DMA : 0, %t31, DMA : 0)
        AIE.flow(%t20, DMA : 0, %t71, DMA : 0)
        AIE.flow(%t20, DMA : 0, %t82, DMA : 0)

        AIE.flow(%t60, DMA : 0, %t02, DMA : 1)
        AIE.flow(%t60, DMA : 0, %t83, DMA : 1)
        AIE.flow(%t60, DMA : 0, %t22, DMA : 1)
        AIE.flow(%t60, DMA : 0, %t31, DMA : 1)
    }
    """

    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        _aie_python_passes.register_pathfinder_flows_with_python(print_hello)
        mlir_module = Module.parse(module)
        device = mlir_module.body.operations[0]
        PassManager.parse("AIE.device(aie-create-pathfinder-flows-with-python)").run(
            device.operation
        )

        print(mlir_module)


# CHECK-LABEL: TEST: testPathfinderFlowsWithPython
# CHECK: max_col=8 max_row=3 is_legal=True
# CHECK: hello from PythonDynamicTileAnalysis::runAnalysis!
# CHECK:  AIE.device(xcvc1902) {
# CHECK:    %tile_0_3 = AIE.tile(0, 3)
# CHECK:    %tile_0_2 = AIE.tile(0, 2)
# CHECK:    %tile_0_0 = AIE.tile(0, 0)
# CHECK:    %tile_1_3 = AIE.tile(1, 3)
# CHECK:    %tile_1_1 = AIE.tile(1, 1)
# CHECK:    %tile_1_0 = AIE.tile(1, 0)
# CHECK:    %tile_2_0 = AIE.tile(2, 0)
# CHECK:    %tile_3_0 = AIE.tile(3, 0)
# CHECK:    %tile_2_2 = AIE.tile(2, 2)
# CHECK:    %tile_3_1 = AIE.tile(3, 1)
# CHECK:    %tile_6_0 = AIE.tile(6, 0)
# CHECK:    %tile_7_0 = AIE.tile(7, 0)
# CHECK:    %tile_7_1 = AIE.tile(7, 1)
# CHECK:    %tile_7_2 = AIE.tile(7, 2)
# CHECK:    %tile_7_3 = AIE.tile(7, 3)
# CHECK:    %tile_8_0 = AIE.tile(8, 0)
# CHECK:    %tile_8_2 = AIE.tile(8, 2)
# CHECK:    %tile_8_3 = AIE.tile(8, 3)
# CHECK:    %switchbox_1_0 = AIE.switchbox(%tile_1_0) {
# CHECK:      AIE.connect<East : 0, North : 0>
# CHECK:      AIE.connect<East : 1, West : 0>
# CHECK:    }
# CHECK:    %switchbox_1_1 = AIE.switchbox(%tile_1_1) {
# CHECK:      AIE.connect<South : 0, North : 0>
# CHECK:    }
# CHECK:    %tile_1_2 = AIE.tile(1, 2)
# CHECK:    %switchbox_1_2 = AIE.switchbox(%tile_1_2) {
# CHECK:      AIE.connect<South : 0, North : 0>
# CHECK:    }
# CHECK:    %switchbox_1_3 = AIE.switchbox(%tile_1_3) {
# CHECK:      AIE.connect<South : 0, DMA : 0>
# CHECK:    }
# CHECK:    %switchbox_2_0 = AIE.switchbox(%tile_2_0) {
# CHECK:      AIE.connect<South : 3, West : 0>
# CHECK:      AIE.connect<South : 3, North : 0>
# CHECK:      AIE.connect<East : 0, West : 1>
# CHECK:      AIE.connect<East : 0, North : 1>
# CHECK:    }
# CHECK:    %shimmux_2_0 = AIE.shimmux(%tile_2_0) {
# CHECK:      AIE.connect<DMA : 0, North : 3>
# CHECK:    }
# CHECK:    %tile_2_1 = AIE.tile(2, 1)
# CHECK:    %switchbox_2_1 = AIE.switchbox(%tile_2_1) {
# CHECK:      AIE.connect<South : 0, North : 0>
# CHECK:      AIE.connect<South : 0, East : 0>
# CHECK:      AIE.connect<South : 1, North : 1>
# CHECK:    }
# CHECK:    %switchbox_2_2 = AIE.switchbox(%tile_2_2) {
# CHECK:      AIE.connect<South : 0, East : 0>
# CHECK:      AIE.connect<South : 1, DMA : 1>
# CHECK:    }
# CHECK:    %switchbox_3_1 = AIE.switchbox(%tile_3_1) {
# CHECK:      AIE.connect<West : 0, DMA : 0>
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:      AIE.connect<South : 0, DMA : 1>
# CHECK:    }
# CHECK:    %tile_3_2 = AIE.tile(3, 2)
# CHECK:    %switchbox_3_2 = AIE.switchbox(%tile_3_2) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %tile_4_1 = AIE.tile(4, 1)
# CHECK:    %switchbox_4_1 = AIE.switchbox(%tile_4_1) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %tile_4_2 = AIE.tile(4, 2)
# CHECK:    %switchbox_4_2 = AIE.switchbox(%tile_4_2) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %tile_5_1 = AIE.tile(5, 1)
# CHECK:    %switchbox_5_1 = AIE.switchbox(%tile_5_1) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %tile_5_2 = AIE.tile(5, 2)
# CHECK:    %switchbox_5_2 = AIE.switchbox(%tile_5_2) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %tile_6_1 = AIE.tile(6, 1)
# CHECK:    %switchbox_6_1 = AIE.switchbox(%tile_6_1) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:      AIE.connect<South : 0, North : 0>
# CHECK:    }
# CHECK:    %tile_6_2 = AIE.tile(6, 2)
# CHECK:    %switchbox_6_2 = AIE.switchbox(%tile_6_2) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:      AIE.connect<South : 0, North : 0>
# CHECK:    }
# CHECK:    %switchbox_7_1 = AIE.switchbox(%tile_7_1) {
# CHECK:      AIE.connect<West : 0, DMA : 0>
# CHECK:    }
# CHECK:    %switchbox_7_2 = AIE.switchbox(%tile_7_2) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %switchbox_8_2 = AIE.switchbox(%tile_8_2) {
# CHECK:      AIE.connect<West : 0, DMA : 0>
# CHECK:    }
# CHECK:    %switchbox_0_0 = AIE.switchbox(%tile_0_0) {
# CHECK:      AIE.connect<East : 0, North : 0>
# CHECK:    }
# CHECK:    %tile_0_1 = AIE.tile(0, 1)
# CHECK:    %switchbox_0_1 = AIE.switchbox(%tile_0_1) {
# CHECK:      AIE.connect<South : 0, North : 0>
# CHECK:    }
# CHECK:    %switchbox_0_2 = AIE.switchbox(%tile_0_2) {
# CHECK:      AIE.connect<South : 0, DMA : 1>
# CHECK:    }
# CHECK:    %switchbox_3_0 = AIE.switchbox(%tile_3_0) {
# CHECK:      AIE.connect<East : 0, West : 0>
# CHECK:      AIE.connect<East : 0, North : 0>
# CHECK:    }
# CHECK:    %tile_4_0 = AIE.tile(4, 0)
# CHECK:    %switchbox_4_0 = AIE.switchbox(%tile_4_0) {
# CHECK:      AIE.connect<East : 0, West : 0>
# CHECK:    }
# CHECK:    %tile_5_0 = AIE.tile(5, 0)
# CHECK:    %switchbox_5_0 = AIE.switchbox(%tile_5_0) {
# CHECK:      AIE.connect<East : 0, West : 0>
# CHECK:    }
# CHECK:    %switchbox_6_0 = AIE.switchbox(%tile_6_0) {
# CHECK:      AIE.connect<South : 3, West : 0>
# CHECK:      AIE.connect<South : 3, North : 0>
# CHECK:    }
# CHECK:    %shimmux_6_0 = AIE.shimmux(%tile_6_0) {
# CHECK:      AIE.connect<DMA : 0, North : 3>
# CHECK:    }
# CHECK:    %tile_6_3 = AIE.tile(6, 3)
# CHECK:    %switchbox_6_3 = AIE.switchbox(%tile_6_3) {
# CHECK:      AIE.connect<South : 0, East : 0>
# CHECK:    }
# CHECK:    %switchbox_7_3 = AIE.switchbox(%tile_7_3) {
# CHECK:      AIE.connect<West : 0, East : 0>
# CHECK:    }
# CHECK:    %switchbox_8_3 = AIE.switchbox(%tile_8_3) {
# CHECK:      AIE.connect<West : 0, DMA : 1>
# CHECK:    }
# CHECK:    AIE.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
# CHECK:    AIE.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
# CHECK:    AIE.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
# CHECK:    AIE.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
# CHECK:    AIE.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
# CHECK:    AIE.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
# CHECK:    AIE.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
# CHECK:    AIE.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
# CHECK:    AIE.wire(%tile_1_1 : Core, %switchbox_1_1 : Core)
# CHECK:    AIE.wire(%tile_1_1 : DMA, %switchbox_1_1 : DMA)
# CHECK:    AIE.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
# CHECK:    AIE.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
# CHECK:    AIE.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
# CHECK:    AIE.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
# CHECK:    AIE.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
# CHECK:    AIE.wire(%tile_1_3 : Core, %switchbox_1_3 : Core)
# CHECK:    AIE.wire(%tile_1_3 : DMA, %switchbox_1_3 : DMA)
# CHECK:    AIE.wire(%switchbox_1_2 : North, %switchbox_1_3 : South)
# CHECK:    AIE.wire(%switchbox_1_0 : East, %switchbox_2_0 : West)
# CHECK:    AIE.wire(%shimmux_2_0 : North, %switchbox_2_0 : South)
# CHECK:    AIE.wire(%tile_2_0 : DMA, %shimmux_2_0 : DMA)
# CHECK:    AIE.wire(%switchbox_1_1 : East, %switchbox_2_1 : West)
# CHECK:    AIE.wire(%tile_2_1 : Core, %switchbox_2_1 : Core)
# CHECK:    AIE.wire(%tile_2_1 : DMA, %switchbox_2_1 : DMA)
# CHECK:    AIE.wire(%switchbox_2_0 : North, %switchbox_2_1 : South)
# CHECK:    AIE.wire(%switchbox_1_2 : East, %switchbox_2_2 : West)
# CHECK:    AIE.wire(%tile_2_2 : Core, %switchbox_2_2 : Core)
# CHECK:    AIE.wire(%tile_2_2 : DMA, %switchbox_2_2 : DMA)
# CHECK:    AIE.wire(%switchbox_2_1 : North, %switchbox_2_2 : South)
# CHECK:    AIE.wire(%switchbox_2_0 : East, %switchbox_3_0 : West)
# CHECK:    AIE.wire(%switchbox_2_1 : East, %switchbox_3_1 : West)
# CHECK:    AIE.wire(%tile_3_1 : Core, %switchbox_3_1 : Core)
# CHECK:    AIE.wire(%tile_3_1 : DMA, %switchbox_3_1 : DMA)
# CHECK:    AIE.wire(%switchbox_3_0 : North, %switchbox_3_1 : South)
# CHECK:    AIE.wire(%switchbox_2_2 : East, %switchbox_3_2 : West)
# CHECK:    AIE.wire(%tile_3_2 : Core, %switchbox_3_2 : Core)
# CHECK:    AIE.wire(%tile_3_2 : DMA, %switchbox_3_2 : DMA)
# CHECK:    AIE.wire(%switchbox_3_1 : North, %switchbox_3_2 : South)
# CHECK:    AIE.wire(%switchbox_3_0 : East, %switchbox_4_0 : West)
# CHECK:    AIE.wire(%switchbox_3_1 : East, %switchbox_4_1 : West)
# CHECK:    AIE.wire(%tile_4_1 : Core, %switchbox_4_1 : Core)
# CHECK:    AIE.wire(%tile_4_1 : DMA, %switchbox_4_1 : DMA)
# CHECK:    AIE.wire(%switchbox_4_0 : North, %switchbox_4_1 : South)
# CHECK:    AIE.wire(%switchbox_3_2 : East, %switchbox_4_2 : West)
# CHECK:    AIE.wire(%tile_4_2 : Core, %switchbox_4_2 : Core)
# CHECK:    AIE.wire(%tile_4_2 : DMA, %switchbox_4_2 : DMA)
# CHECK:    AIE.wire(%switchbox_4_1 : North, %switchbox_4_2 : South)
# CHECK:    AIE.wire(%switchbox_4_0 : East, %switchbox_5_0 : West)
# CHECK:    AIE.wire(%switchbox_4_1 : East, %switchbox_5_1 : West)
# CHECK:    AIE.wire(%tile_5_1 : Core, %switchbox_5_1 : Core)
# CHECK:    AIE.wire(%tile_5_1 : DMA, %switchbox_5_1 : DMA)
# CHECK:    AIE.wire(%switchbox_5_0 : North, %switchbox_5_1 : South)
# CHECK:    AIE.wire(%switchbox_4_2 : East, %switchbox_5_2 : West)
# CHECK:    AIE.wire(%tile_5_2 : Core, %switchbox_5_2 : Core)
# CHECK:    AIE.wire(%tile_5_2 : DMA, %switchbox_5_2 : DMA)
# CHECK:    AIE.wire(%switchbox_5_1 : North, %switchbox_5_2 : South)
# CHECK:    AIE.wire(%switchbox_5_0 : East, %switchbox_6_0 : West)
# CHECK:    AIE.wire(%shimmux_6_0 : North, %switchbox_6_0 : South)
# CHECK:    AIE.wire(%tile_6_0 : DMA, %shimmux_6_0 : DMA)
# CHECK:    AIE.wire(%switchbox_5_1 : East, %switchbox_6_1 : West)
# CHECK:    AIE.wire(%tile_6_1 : Core, %switchbox_6_1 : Core)
# CHECK:    AIE.wire(%tile_6_1 : DMA, %switchbox_6_1 : DMA)
# CHECK:    AIE.wire(%switchbox_6_0 : North, %switchbox_6_1 : South)
# CHECK:    AIE.wire(%switchbox_5_2 : East, %switchbox_6_2 : West)
# CHECK:    AIE.wire(%tile_6_2 : Core, %switchbox_6_2 : Core)
# CHECK:    AIE.wire(%tile_6_2 : DMA, %switchbox_6_2 : DMA)
# CHECK:    AIE.wire(%switchbox_6_1 : North, %switchbox_6_2 : South)
# CHECK:    AIE.wire(%tile_6_3 : Core, %switchbox_6_3 : Core)
# CHECK:    AIE.wire(%tile_6_3 : DMA, %switchbox_6_3 : DMA)
# CHECK:    AIE.wire(%switchbox_6_2 : North, %switchbox_6_3 : South)
# CHECK:    AIE.wire(%switchbox_6_1 : East, %switchbox_7_1 : West)
# CHECK:    AIE.wire(%tile_7_1 : Core, %switchbox_7_1 : Core)
# CHECK:    AIE.wire(%tile_7_1 : DMA, %switchbox_7_1 : DMA)
# CHECK:    AIE.wire(%switchbox_6_2 : East, %switchbox_7_2 : West)
# CHECK:    AIE.wire(%tile_7_2 : Core, %switchbox_7_2 : Core)
# CHECK:    AIE.wire(%tile_7_2 : DMA, %switchbox_7_2 : DMA)
# CHECK:    AIE.wire(%switchbox_7_1 : North, %switchbox_7_2 : South)
# CHECK:    AIE.wire(%switchbox_6_3 : East, %switchbox_7_3 : West)
# CHECK:    AIE.wire(%tile_7_3 : Core, %switchbox_7_3 : Core)
# CHECK:    AIE.wire(%tile_7_3 : DMA, %switchbox_7_3 : DMA)
# CHECK:    AIE.wire(%switchbox_7_2 : North, %switchbox_7_3 : South)
# CHECK:    AIE.wire(%switchbox_7_2 : East, %switchbox_8_2 : West)
# CHECK:    AIE.wire(%tile_8_2 : Core, %switchbox_8_2 : Core)
# CHECK:    AIE.wire(%tile_8_2 : DMA, %switchbox_8_2 : DMA)
# CHECK:    AIE.wire(%switchbox_7_3 : East, %switchbox_8_3 : West)
# CHECK:    AIE.wire(%tile_8_3 : Core, %switchbox_8_3 : Core)
# CHECK:    AIE.wire(%tile_8_3 : DMA, %switchbox_8_3 : DMA)
# CHECK:    AIE.wire(%switchbox_8_2 : North, %switchbox_8_3 : South)
# CHECK:  }
testPathfinderFlowsWithPython()
