// RUN: aie-opt --verify-diagnostics --aie-verify-connections %s
module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		%lock = AIE.lock(%tile01, 0)
		%buf  = AIE.buffer(%tile01) : memref<256xi32>
		%mem  = AIE.mem(%tile01) {
			// expected-error@+1 {{S2MM DMA defined, but no incoming connections to the DMA are defined}}
			%dma = AIE.dmaStart("S2MM", 0, ^bd, ^end)
			^bd:
				AIE.useLock(%lock, "Acquire", 0)
				AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
				AIE.useLock(%lock, "Release", 1)
				AIE.nextBd ^end
			^end:
				AIE.end
		}
    }
}
