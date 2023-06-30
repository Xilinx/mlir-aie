// RUN: set +o pipefail; aie-opt --aie-create-pathfinder-flows --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: There is no matching incoming connection for <"West" : 1> in tile (1, 1) for this outgoing connection
module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		AIE.switchbox(%tile01) {
			AIE.connect<"DMA" : 1, "East" : 1>
		}
    }
}
