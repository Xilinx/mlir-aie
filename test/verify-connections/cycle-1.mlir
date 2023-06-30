// RUN: set +o pipefail; aie-opt --aie-create-pathfinder-flows --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: There is a cycle in the route containing this connection

// Note the cycle in this example going clockwise between tiles
// 0, 8 ----> 8, 8
//   ^         |
//   |         |
//   |         .
// 0, 1 <---- 8, 1

module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		%tile08 = AIE.tile(0, 8)
		AIE.flow(%tile01, "East" : 0, %tile08, "East" : 0)
		%tile18 = AIE.tile(1, 8)
		%tile88 = AIE.tile(8, 8)
		AIE.flow(%tile18, "West" : 0, %tile88, "South" : 0)
		%tile87 = AIE.tile(8, 7)
		%tile81 = AIE.tile(8, 1)
		AIE.flow(%tile87, "North" : 0, %tile81, "West" : 0)
		%tile71 = AIE.tile(7, 1)
		%tile11 = AIE.tile(1, 1)
		AIE.flow(%tile71, "East" : 0, %tile11, "West" : 0)
    }
}
