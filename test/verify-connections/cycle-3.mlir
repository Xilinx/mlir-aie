// RUN: set +o pipefail; aie-opt --aie-create-pathfinder-flows --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: There is a cycle in the route containing this connection

// This test should find a cycle in one of the disconnected components.

//                  4, 6 ----> 6, 6
//                   ^          |
//                   |          |
//                   |          .
//                  4, 4 <---- 6, 4
// 0, 3 ----> 3, 3
//   ^         |
//   |         |
//   |         .
// 0, 1      3, 1

module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		%tile03 = AIE.tile(0, 3)
		AIE.flow(%tile01, "Core" : 0, %tile03, "East" : 0)
		%tile13 = AIE.tile(1, 3)
		%tile33 = AIE.tile(3, 3)
		AIE.flow(%tile13, "West" : 0, %tile33, "South" : 0)
		%tile32 = AIE.tile(3, 2)
		%tile31 = AIE.tile(3, 1)
		AIE.flow(%tile32, "North" : 0, %tile31, "Core" : 0)

		%tile44 = AIE.tile(4, 4)
		%tile46 = AIE.tile(4, 6)
		AIE.flow(%tile44, "East" : 0, %tile46, "East" : 1)
		%tile56 = AIE.tile(5, 6)
		%tile66 = AIE.tile(6, 6)
		AIE.flow(%tile56, "West" : 1, %tile66, "South" : 0)
		%tile65 = AIE.tile(6, 5)
		%tile64 = AIE.tile(6, 4)
		AIE.flow(%tile65, "North" : 0, %tile64, "West" : 1)
		%tile54 = AIE.tile(5, 4)
		AIE.flow(%tile54, "East" : 1, %tile54, "West" : 0)
    }
}
