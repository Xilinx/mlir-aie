// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// Two independent feed-forward objectFIFOs from one producer (shim) to two
// distinct consumer cores. No tile is both a producer and a consumer, so the
// dependency graph has no back-edge and no cycle -> the pass must be silent.
module {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @ofA(%shim, {%core0}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>
    aie.objectfifo @ofB(%shim, {%core1}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>
  }
}
