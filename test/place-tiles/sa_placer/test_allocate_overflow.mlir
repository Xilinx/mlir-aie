// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=42' %s | FileCheck %s

// Test: SA placer handles core tiles whose intratile ObjectFifos overflow
// local memory (64KB). The penalty model accounts for relocatable fifos
// and accessible neighbor spare capacity.

// CHECK: aie.device(npu2)
// Verify placement produces physical tiles and allocate ops for overflow
// CHECK: aie.tile
// CHECK: aie.objectfifo.allocate

module {
  aie.device(npu2) {
    // Two core tiles and I/O
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)

    // Input/output through mem tile
    aie.objectfifo @input(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @to_core(%mem, {%core0}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@input] -> [@to_core]([] [0])

    // Inter-core connection
    aie.objectfifo @core_to_core(%core0, {%core1}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    // Intratile ObjectFifos on core0 that overflow memory (3 x depth3 x 8KB = 72KB > 64KB)
    aie.objectfifo @intra0(%core0, {%core0}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @intra1(%core0, {%core0}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<8192xi8>>
    aie.objectfifo @intra2(%core0, {%core0}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<8192xi8>>

    // Output
    aie.objectfifo @output(%core1, {%shim}, 2 : i32) : !aie.objectfifo<memref<256xi32>>

    %c0 = aie.core(%core0) {
      aie.end
    }
    %c1 = aie.core(%core1) {
      aie.end
    }
  }
}
