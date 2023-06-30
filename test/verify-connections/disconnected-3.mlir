// RUN: set +o pipefail; aie-opt --aie-verify-connections %s 2>&1 1>/dev/null | FileCheck %s
// CHECK: MM2S DMA defined, but no outgoing connections out of the DMA are defined
module @test {
    AIE.device(xcvc1902) {
		%tile01 = AIE.tile(0, 1)
		%lock = AIE.lock(%tile01, 0)
		%buf  = AIE.buffer(%tile01) : memref<256xi32>
		%mem  = AIE.mem(%tile01) {
			%dma = AIE.dmaStart("MM2S", 0, ^bd, ^end)
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