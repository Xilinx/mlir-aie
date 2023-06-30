// RUN: aie-opt --verify-diagnostics --aie-create-pathfinder-flows --aie-verify-connections %s
module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		AIE.switchbox(%tile01) {
			// expected-error@+1 {{There is no matching incoming connection for <"West" : 1> in tile (1, 1) for this outgoing connection}}
			AIE.connect<"DMA" : 1, "East" : 1>
		}
    }
}
