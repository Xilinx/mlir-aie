// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
//
// True-negative test for the no-multicast case. Two cores form a cyclic
// dependency (core0 -> core1 -> core0) through point-to-point objectFIFOs, and
// the fifos are replayed (repeat_count = 2, T == 2), so the T >= 2 guard is
// satisfied. But every fifo is point-to-point (fan-out 1): array_fan = 0, so
// there is no coupled-broadcast demand to outrun the buffer slack. The circular
// wait that this pass models is specifically the wide-multicast re-acquire
// cycle; a point-to-point loop is a different (temporal, single-consumer) class
// that the min-buffer/cyclostatic depth math handles elsewhere and this pass
// deliberately does not claim. The pass MUST stay silent.
module {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @fwd(%c0, {%c1}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>
    aie.objectfifo @bwd(%c1, {%c0}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>
  }
}
