// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.core(%tile) {
    // expected-error @+1 {{expected buffer to reference an aie.buffer or aie.external_buffer declaration}}
    %buffer = aie.get_buffer @missing : memref<16xi32>
    aie.end
  }
}

// -----

aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.lock @not_a_buffer(%tile, 0)
  aie.core(%tile) {
    // expected-error @+1 {{expected buffer to reference an aie.buffer or aie.external_buffer declaration}}
    %buffer = aie.get_buffer @not_a_buffer : memref<16xi32>
    aie.end
  }
}

// -----

aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.buffer @buffer(%tile) : memref<16xi32>
  aie.core(%tile) {
    // expected-error @+1 {{result type must match the declared buffer type}}
    %buffer = aie.get_buffer @buffer : memref<8xi32>
    aie.end
  }
}

// -----

aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.external_buffer @external : memref<16xi32>
  aie.core(%tile) {
    // expected-error @+1 {{result type must match the declared buffer type}}
    %buffer = aie.get_buffer @external : memref<16xf32>
    aie.end
  }
}

// -----

// Shape and element type alone are insufficient: layouts must also match.
aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.buffer @buffer(%tile) : memref<4xi32, strided<[2]>>
  aie.core(%tile) {
    // expected-error @+1 {{result type must match the declared buffer type}}
    %buffer = aie.get_buffer @buffer : memref<4xi32>
    aie.end
  }
}

// -----

aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.core(%tile) {
    %value = arith.constant 1 : i32
    // expected-error @+1 {{expected lock to reference an aie.lock declaration}}
    aie.use_lock(@missing, AcquireGreaterEqual, %value)
    aie.end
  }
}

// -----

aie.device(npu1) {
  %tile = aie.tile(0, 2)
  aie.buffer @not_a_lock(%tile) : memref<16xi32>
  aie.core(%tile) {
    %value = arith.constant 1 : i32
    // expected-error @+1 {{expected lock to reference an aie.lock declaration}}
    aie.use_lock(@not_a_lock, Release, %value)
    aie.end
  }
}

// -----

aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %tile = aie.tile(0, 2)
  aie.objectfifo @fifo(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error @+1 {{externalBuffers must reference aie.external_buffer declarations}}
  aie.objectfifo.register_external_buffers @fifo (%shim, [@missing])
}

// -----

// Unlike get_buffer, external-buffer registration cannot name a local buffer.
aie.device(npu1) {
  %shim = aie.tile(0, 0)
  %tile = aie.tile(0, 2)
  aie.buffer @not_external(%tile) : memref<16xi32>
  aie.objectfifo @fifo(%shim, {%tile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  // expected-error @+1 {{externalBuffers must reference aie.external_buffer declarations}}
  aie.objectfifo.register_external_buffers @fifo (%shim, [@not_external])
}

// -----

aie.device(npu1) {
  %shim = aie.tile(0, 0)
  aie.external_buffer @not_a_fifo : memref<16xi32>
  // expected-error @+1 {{objectFIFO must reference an aie.objectfifo declaration}}
  aie.objectfifo.register_external_buffers @not_a_fifo (%shim, [@not_a_fifo])
}

// -----

aie.device(npu1) {
  aie.runtime_sequence() {
    %value = arith.constant 1 : i32
    // expected-error @+1 {{lock must reference an aie.lock declaration}}
    aiex.set_lock(@missing, %value)
  }
}

// -----

aie.device(npu1) {
  aie.external_buffer @not_a_lock : memref<16xi32>
  aie.runtime_sequence() {
    %value = arith.constant 1 : i32
    // expected-error @+1 {{lock must reference an aie.lock declaration}}
    aiex.set_lock(@not_a_lock, %value)
  }
}

// -----

// Local-only memtile channels must diagnose unresolved buffer accessors rather
// than dereferencing a missing declaration while verifying the enclosing DMA.
aie.device(npu2) {
  %tile = aie.tile(0, 1)
  aie.memtile_dma(%tile) {
    aie.dma_start(MM2S, 4, ^bd, ^end)
  ^bd:
    %buffer = aie.get_buffer @missing : memref<16xi32>
    // expected-error @+1 {{BDs only support BufferOp or ExternalBufferOp operands.}}
    aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}

// -----

// A resolved external buffer is still invalid for an on-chip memory tile DMA.
aie.device(npu2) {
  %tile = aie.tile(0, 1)
  aie.external_buffer @external : memref<16xi32>
  aie.memtile_dma(%tile) {
    aie.dma_start(MM2S, 4, ^bd, ^end)
  ^bd:
    %buffer = aie.get_buffer @external : memref<16xi32>
    // expected-error @+1 {{core and memory tile DMAs require an aie.buffer declaration}}
    aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16)
    aie.next_bd ^bd
  ^end:
    aie.end
  }
}
