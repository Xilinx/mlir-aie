// RUN: set +o pipefail; aie-opt --aie-create-pathfinder-flows --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: There is a cycle in the route containing this connection

// This test should find a cycle in one of the disconnected components.
// There are some broadcasting tiles.

//                  4, 8       6, 8       8, 8
//                   ^          ^          ^
//                   |          |          |
//                   |          |          |
//                  4, 6 ----> 6, 6 ----> 8, 6
//  0, 6             ^          |          |
//   ^               |          |          |
//   |               |          .          .
//   |              4, 4 <---- 6, 4 ----> 8, 4
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

		%tile06 = AIE.tile(0, 6)
		AIE.flow(%tile03, "Core" : 0, %tile06, "Core" : 0)

		%tile44 = AIE.tile(4, 4)
		%tile46 = AIE.tile(4, 6)
		AIE.flow(%tile44, "East" : 0, %tile46, "East" : 0)
		%tile56 = AIE.tile(5, 6)
		%tile66 = AIE.tile(6, 6)
		AIE.flow(%tile56, "West" : 0, %tile66, "South" : 0)
		%tile65 = AIE.tile(6, 5)
		%tile64 = AIE.tile(6, 4)
		AIE.flow(%tile65, "North" : 0, %tile64, "West" : 0)
		%tile54 = AIE.tile(5, 4)
		AIE.flow(%tile54, "East" : 0, %tile54, "West" : 0)

		%tile48 = AIE.tile(4, 8)
		AIE.flow(%tile46, "South" : 0, %tile48, "Core" : 0)

		%tile68 = AIE.tile(6, 8)
		%tile86 = AIE.tile(8, 6)
		AIE.flow(%tile66, "West" : 0, %tile68, "Core" : 0)
		AIE.flow(%tile66, "West" : 0, %tile86, "Core" : 0)
		AIE.flow(%tile66, "West" : 0, %tile86, "North" : 0)
		%tile88 = AIE.tile(8, 8)
		%tile87 = AIE.tile(8, 7)
		AIE.flow(%tile87, "South" : 0, %tile88, "Core" : 0)
		%tile85 = AIE.tile(8, 5)
		%tile84 = AIE.tile(8, 4)
		AIE.flow(%tile66, "West" : 0, %tile86, "South" : 0)
		AIE.flow(%tile85, "North" : 0, %tile84, "Core" : 0)
		AIE.flow(%tile64, "North" : 0, %tile84, "Core" : 1)
    }
}
