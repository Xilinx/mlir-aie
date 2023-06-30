// RUN: aie-opt --verify-diagnostics --aie-create-pathfinder-flows --aie-verify-connections %s
module @test {
    AIE.device(xcvc1902) {
		%tile11 = AIE.tile(1, 1)
		AIE.switchbox(%tile11) {
			// expected-error@+1 {{There is no matching outgoing connection for <"East" : 2> in tile (0, 1) for this incoming connection}}
			AIE.connect<"West" : 2, "DMA" : 1>
		}
    }
}
