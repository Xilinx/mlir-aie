// RUN: aie-opt --aie-verify-connections %s 
// (Simply checking zero exit code.)

// This test should run through the verify connections pass without any issues,
// ase there are no cycles, and all connections are terminated.
module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		%tile11 = AIE.tile(1, 1)
		%tile12 = AIE.tile(1, 2)

		%lock01 = AIE.lock(%tile01, 0)
		%lock12 = AIE.lock(%tile12, 0)

		%buf01  = AIE.buffer(%tile01) : memref<256xi32>
		%buf12  = AIE.buffer(%tile12) : memref<256xi32>

		%mem01  = AIE.mem(%tile01) {
			%dma = AIE.dmaStart("MM2S", 0, ^bd, ^end)
			^bd:
				AIE.useLock(%lock01, "Acquire", 0)
				AIE.dmaBd(<%buf01 : memref<256xi32>, 0, 256>, 0)
				AIE.useLock(%lock01, "Release", 1)
				AIE.nextBd ^end
			^end:
				AIE.end
		}

		%switchbox01 = AIE.switchbox(%tile01) {
			AIE.connect<"DMA" : 0, "East" : 1>
		}
		%switchbox11 = AIE.switchbox(%tile11) {
			AIE.connect<"West" : 1, "North" : 2>
		}
		%switchbox12 = AIE.switchbox(%tile12) {
			AIE.connect<"South" : 2, "DMA" : 1>
		}

		%mem12 = AIE.mem(%tile12) {
			%dma = AIE.dmaStart("S2MM", 1, ^bd, ^end)
			^bd:
				AIE.useLock(%lock12, "Acquire", 0)
				AIE.dmaBd(<%buf12 : memref<256xi32>, 0, 256>, 0)
				AIE.useLock(%lock12, "Release", 1)
				AIE.nextBd ^end
			^end:
				AIE.end
		}
    }
}
