// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// True-negative / replay-guard test. A weight objectFIFO fans out as a coupled
// broadcast to a wide row of eight compute cores (array_fan = 8) that feed a
// back-pressure cycle, but it is a SINGLE-TRIP launch: no repeat_count, so the
// SDF trip multiplicity T == 1. A single-trip broadcast fans out once and drains
// monotonically -- the producer never has to re-acquire a slot while prior-trip
// tokens are still outstanding, so no re-acquire cycle forms regardless of how
// wide the fan-out is. This mirrors the known-good whole-array matmul examples
// (array-wide broadcast, low depth, T == 1) that run on device. The pass MUST
// stay silent here; flagging it would be a false positive (and, per mlir-air
// #1694, forcing a reset on the safe multi-trip cascade path costs ~42% perf).
module {
  aie.device(npu2) {
    %shim  = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem   = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    %c5 = aie.logical_tile<CoreTile>(?, ?)
    %c6 = aie.logical_tile<CoreTile>(?, ?)
    %c7 = aie.logical_tile<CoreTile>(?, ?)
    // Activations into the row (point-to-point), no repeat_count -> T == 1.
    aie.objectfifo @memH(%shim, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, 4 : i32) : !aie.objectfifo<memref<16x32xbf16>>
    // Coupled weight broadcast, wide fan-out, low depth, still single-trip.
    aie.objectfifo @memW(%mem, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    // Outputs back to the mem tile close the back-pressure cycle.
    aie.objectfifo @memC(%c0, {%mem}, 4 : i32) : !aie.objectfifo<memref<16x32xbf16>>
  }
}
