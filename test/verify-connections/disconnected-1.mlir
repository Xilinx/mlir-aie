// RUN: set +o pipefail; aie-opt --aie-create-pathfinder-flows --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: There is no matching outgoing connection for <"East" : 2> in tile (0, 1) for this incoming connection
module @test {
    AIE.device(xcvc1902) {
		%tile11 = AIE.tile(1, 1)
		AIE.switchbox(%tile11) {
			AIE.connect<"West" : 2, "DMA" : 1>
		}
    }
}
