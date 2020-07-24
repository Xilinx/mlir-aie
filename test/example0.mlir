// RUN: aie-opt %s | FileCheck %s

// Goal of AIE dialect modeling: ensure the legality of Lock + DMA + Memory accesses
// and StreamSwitch connectivity
// This is a physical netlist of mapping code to multiple AIE cores in the AIE array
//
// The physical netlist will get translated to XAIE* calls (from the ARM host) to configure
// the AIE array (Core, StreamSwitch, Lock, DMMA setups)
// and the code in the CoreModule region will get translated to AIE core instrinsics

// CHECK-LABEL: module @example0 {
// CHECK:       }

module @example0 {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 4) (3, 4) (4, 4) (5, 4)
  // (2, 3) (3, 3) (4, 3) (5, 3)
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %c33 = AIE.core(3, 3)
  %c34 = AIE.core(4, 3)
  %c42 = AIE.core(4, 2)
  %c44 = AIE.core(4, 4)

  %m33 = AIE.mem(3, 3) {
    %buf = alloc() { id=0, bank=1 } : memref<256xi32> 
    %l0 = AIE.lock<0>()
    %l1 = AIE.lock<1>()

    %dmaSt0 = AIE.dmaStart("MM2S0")
    %dmaSt1 = AIE.dmaStart("MM2S1")
    AIE.terminator(^dma0, ^dma1, ^end)
    ^dma0:
      cond_br %dmaSt0, ^bd0, ^end
    ^bd0:
      AIE.useLock(%l0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l0, "Release", 0, 0)
      br ^end
    ^dma1:
      cond_br %dmaSt1, ^bd1, ^end
    ^bd1:
      AIE.useLock(%l1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l1, "Release", 1, 0)
      br ^end
    ^end:
      AIE.end
  }

  %m42 = AIE.mem(4, 2) {
    %buf = alloc() { id=0, bank=0 } : memref<256xi32> 
    %l0 = AIE.lock<0>()

    %dmaSt = AIE.dmaStart("S2MM0")
    AIE.terminator(^dma0, ^end)
    ^dma0:
      cond_br %dmaSt, ^bd0, ^end
    ^bd0:
      AIE.useLock(%l0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l0, "Release", 0, 0)
      br ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(4, 4) {
    %buf = alloc() { id=0, bank=0 } : memref<256xi32> 
    %l0 = AIE.lock<0>()

    %dmaSt = AIE.dmaStart("S2MM0")
    AIE.terminator(^dma0, ^end)
    ^dma0:
      cond_br %dmaSt, ^bd0, ^end
    ^bd0:
      AIE.useLock(%l0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l0, "Release", 0, 0)
      br ^end
    ^end:
      AIE.end
  }

  %s33 = AIE.switchbox(3, 3) {
    AIE.connect<"DMA": 0, "East": 0>
    AIE.connect<"DMA": 1, "East": 1>
  }

  %s34 = AIE.switchbox(4, 3) {
    AIE.connect<"West": 0, "South": 0>
    AIE.connect<"West": 1, "North": 0>
  }

  %s42 = AIE.switchbox(4, 2) {
    AIE.connect<"North": 0, "DMA": 0>
  }

  %s44 = AIE.switchbox(4, 4) {
    AIE.connect<"South":0, "DMA": 0>
  }

  AIE.coreModule(%c33, %m33) {
    %l0 = AIE.lock<0>(%m33)
    %buf = AIE.buffer(%m33, 0) : memref<256xi32>

    AIE.useLock(%l0, "Acquire", 0, 0)
    // code
    AIE.useLock(%l0, "Release", 1, 0)
    AIE.end
  }

  AIE.coreModule(%c42, %m42) {
    %l0 = AIE.lock<0>(%m42)
    %buf = AIE.buffer(%m42, 0) : memref<256xi32>

    AIE.useLock(%l0, "Acquire", 1, 0)
    // code
    AIE.useLock(%l0, "Release", 0, 0)
    AIE.end
  }

  AIE.coreModule(%c44, %m44) {
    %l0 = AIE.lock<0>(%m44)
    %buf = AIE.buffer(%m44, 0) : memref<256xi32>

    AIE.useLock(%l0, "Acquire", 1, 0)
    // code
    AIE.useLock(%l0, "Release", 0, 0)
    AIE.end
  }
}
