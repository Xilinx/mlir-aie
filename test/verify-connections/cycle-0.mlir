// RUN: set +o pipefail; aie-opt --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: There is a cycle in the route containing this connection
module @test {
    AIE.device(xcvc1902) {
		%tile21 = AIE.tile(2, 1)
		%tile31 = AIE.tile(3, 1)
		%tile22 = AIE.tile(2, 2)
		%tile32 = AIE.tile(3, 2)
		AIE.switchbox(%tile22) {
			AIE.connect<"South" : 0, "East" : 0>
		}
		AIE.switchbox(%tile32) {
			AIE.connect<"West" : 0, "South" : 0>
		}
		AIE.switchbox(%tile31) {
			AIE.connect<"North" : 0, "West" : 0>
		}
		AIE.switchbox(%tile21) {
			AIE.connect<"East" : 0, "North" : 0>
		}
    }
}
