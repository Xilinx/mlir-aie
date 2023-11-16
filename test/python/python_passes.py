# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes


import aie
from aie.ir import *
from aie.dialects.aie import *
from aie.passmanager import PassManager
from aie._mlir_libs import _aie_python_passes


def testPathfinderFlowsWithPython():
    print("\nTEST: testPathfinderFlowsWithPython")

    def print_hello(grid, edges, flows):
        print("hello from PythonDynamicTileAnalysis::runAnalysis!")
        # print(f"{grid=}")
        # print(f"{edges=}")
        for f in flows:
            print(f, flush=True)
        exit(0)

    module = """
      AIE.device(xcvc1902) {
        %tile_0_3 = AIE.tile(0, 3)
        %tile_0_2 = AIE.tile(0, 2)
        %tile_0_0 = AIE.tile(0, 0)
        %tile_1_3 = AIE.tile(1, 3)
        %tile_1_1 = AIE.tile(1, 1)
        %tile_1_0 = AIE.tile(1, 0)
        %tile_2_0 = AIE.tile(2, 0)
        %tile_3_0 = AIE.tile(3, 0)
        %tile_2_2 = AIE.tile(2, 2)
        %tile_3_1 = AIE.tile(3, 1)
        %tile_6_0 = AIE.tile(6, 0)
        %tile_7_0 = AIE.tile(7, 0)
        %tile_7_1 = AIE.tile(7, 1)
        %tile_7_2 = AIE.tile(7, 2)
        %tile_7_3 = AIE.tile(7, 3)
        %tile_8_0 = AIE.tile(8, 0)
        %tile_8_2 = AIE.tile(8, 2)
        %tile_8_3 = AIE.tile(8, 3)
        
        AIE.flow(%tile_2_0, DMA : 0, %tile_1_3, DMA : 0)
        AIE.flow(%tile_2_0, DMA : 0, %tile_3_1, DMA : 0)
        AIE.flow(%tile_2_0, DMA : 0, %tile_7_1, DMA : 0)
        AIE.flow(%tile_2_0, DMA : 0, %tile_8_2, DMA : 0)
        AIE.flow(%tile_6_0, DMA : 0, %tile_0_2, DMA : 1)
        AIE.flow(%tile_6_0, DMA : 0, %tile_8_3, DMA : 1)
        AIE.flow(%tile_6_0, DMA : 0, %tile_2_2, DMA : 1)
        AIE.flow(%tile_6_0, DMA : 0, %tile_3_1, DMA : 1)
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
# CHECK-LABEL: hello from PythonDynamicTileAnalysis::runAnalysis!
# CHECK: Flow(PathEndPoint(Switchbox(2, 0): DMA): {PathEndPoint(Switchbox(1, 3): DMA), PathEndPoint(Switchbox(3, 1): DMA), PathEndPoint(Switchbox(7, 1): DMA), PathEndPoint(Switchbox(8, 2): DMA)})
# CHECK: Flow(PathEndPoint(Switchbox(6, 0): DMA): {PathEndPoint(Switchbox(0, 2): DMA), PathEndPoint(Switchbox(8, 3): DMA), PathEndPoint(Switchbox(2, 2): DMA), PathEndPoint(Switchbox(3, 1): DMA)})
testPathfinderFlowsWithPython()
