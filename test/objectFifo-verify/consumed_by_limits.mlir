//===- consumed_by_limits.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Where the bytes an output receives are not all the input's, or not as many
// as the pools' objects hold, the proof must say no more than the hardware
// guarantees.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// Two packet routes merge into @host, so what it receives may all be @in2's,
// relayed through the memtile: neither input is proven consumed.

// CHECK-LABEL: module @merged_output
// CHECK-NOT: consumed_by
module @merged_output {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @in1(%shim) DMA
    aie.route_endpoint @in2(%shim) DMA
    aie.route_endpoint @host(%shim1) DMA
    aie.objectfifo.pool @p(%mt) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @fill(%mt) fills @p
    aie.objectfifo.dma_endpoint @drain(%mt) drains @p
    aie.route from @in2 to [@fill]
    aie.route from @in1 to [@host] {packet = #aie.packet_info<pkt_id = 1>}
    aie.route from @drain to [@host] {packet = #aie.packet_info<pkt_id = 2>}
  }
}

// -----

// The output keeps each packet's header, so it receives more bytes than the
// input sent.

// CHECK-LABEL: module @kept_header
// CHECK-NOT: consumed_by
module @kept_header {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    aie.route_endpoint @in(%shim) DMA
    aie.route_endpoint @host(%shim1) DMA
    aie.route from @in to [@host] {keep_pkt_header, packet = #aie.packet_info<pkt_id = 1>}
  }
}

// -----

// The core tile's DMA fills half of each `in` object, so the input sends 32
// bytes for every 64 the output receives.

// CHECK-LABEL: module @fill_half
// CHECK: aie.route_endpoint @in__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 2>], fifoName = "in"}
module @fill_half {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core dimensionsFromStream [<size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The core tile's DMA sends each `out` object twice, so the output receives
// 128 bytes for every 64 the input sends.

// CHECK-LABEL: module @drain_twice
// CHECK: aie.route_endpoint @in__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 2>], fifoName = "in"}
module @drain_twice {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core dimensionsToStream [<size = 2, stride = 0>, <size = 16, stride = 1>], {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// Two paths reach @d: through @b one input byte for every output byte, through
// @c one for every two. Only the larger share is recorded; the smaller
// follows from it.

// CHECK-LABEL: module @two_paths
// CHECK: aie.route_endpoint @a__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @d__cons__dma, input_bytes = 1, output_bytes = 1>], fifoName = "a"}
module @two_paths {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core2 = aie.tile(0, 2)
    %core4 = aie.tile(0, 4)
    aie.objectfifo @a(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%core2, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%core2, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%core4, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %x = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
        %z:2 = aie.objectfifo.acquire @c(Produce, 2) : memref<16xi32>, memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @b(Produce, 1)
        aie.objectfifo.release @c(Produce, 2)
        scf.yield
      }
      aie.end
    }
    aie.core(%core4) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %x = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %y = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        %z = aie.objectfifo.acquire @d(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
        aie.objectfifo.release @d(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The core's own output DMA sends each object twice (the pool's
// repeat_count), so the output receives 128 bytes for every 64 the input
// sends: the input is not proven consumed.

// CHECK-LABEL: module @core_output_repeats
// CHECK: aie.route_endpoint @in__prod__dma({{.*}}) DMA {fifoName = "in"}
module @core_output_repeats {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The trip count is the RTP word @seq writes, but @other may rewrite it with a
// raw write32, so a row may be any length and angle is not proven consumed.

// CHECK-LABEL: module @rtp_raw_write
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_raw_write {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @x(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @angle(%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    %rtp = aie.buffer(%core) {sym_name = "rtp"} : memref<1xi32>
    %barrier = aie.lock(%core)
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aie.use_lock(%barrier, AcquireGreaterEqual, %one)
        %n = memref.load %rtp[%c0] : memref<1xi32>
        aie.use_lock(%barrier, Release, %one)
        %cols = arith.index_cast %n : i32 to index
        %a = aie.objectfifo.acquire @angle(Consume, 1) : memref<128xbf16>
        scf.for %j = %c0 to %cols step %c1 {
          %i = aie.objectfifo.acquire @x(Consume, 1) : memref<128xbf16>
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<128xbf16>
          aie.objectfifo.release @x(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @angle(Consume, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>) {
      %c0 = arith.constant 0 : i32
      %c96 = arith.constant 96 : i32
      aiex.npu.write32(%c0, %c96) {buffer = @rtp} : i32, i32
    }
  }
}
