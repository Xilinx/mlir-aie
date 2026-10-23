//===- npu_buffer_write.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s -o /dev/null
// RUN: aie-opt --split-input-file --verify-diagnostics --verify-each --pass-pipeline='builtin.module(canonicalize,canonicalize)' %s -o /dev/null

// Exact fit without a pin; partial writes end exactly at the buffer boundary.
// Runtime values and masks are independent of the constant symbolic offset.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    memref.global "private" @zeros8 : memref<8xi32> = dense<0>
    memref.global "private" constant @words2 : memref<2xi32> = dense<[5, 6]>
    aie.runtime_sequence(%value: i32, %mask: i32) {
      %z = memref.get_global @zeros8 : memref<8xi32>
      %two = memref.get_global @words2 : memref<2xi32>
      %last = arith.constant 7 : i32
      aiex.npu.blockwrite(%z) {buffer = @acc, address = 0 : ui32} : memref<8xi32>
      aiex.npu.blockwrite(%two) {buffer = @acc, address = 6 : ui32} : memref<2xi32>
      aiex.npu.write32(%last, %value) {buffer = @acc} : i32, i32
      aiex.npu.maskwrite32(%last, %value, %mask) {buffer = @acc} : i32, i32, i32
    }
  }
}

// -----

// Symbolic offsets 1 and 2 are word-aligned. Destination type does not change
// the offset unit, and source provenance remains the backend's responsibility.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %bytes = aie.buffer(%tile) {sym_name = "bytes", address = 0 : i32} : memref<8xi8>
    %wide = aie.buffer(%tile) {sym_name = "wide"} : memref<4xi64>
    aie.runtime_sequence(%data: memref<2xi32>, %value: i32, %mask: i32) {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      aiex.npu.blockwrite(%data) {buffer = @bytes, address = 0 : ui32} : memref<2xi32>
      aiex.npu.write32(%one, %value) {buffer = @bytes} : i32, i32
      aiex.npu.maskwrite32(%two, %value, %mask) {buffer = @wide} : i32, i32, i32
      aiex.npu.blockwrite(%data) {buffer = @wide, address = 6 : ui32} : memref<2xi32>
    }
  }
}

// -----

// Fixed-vector destination storage follows DataLayout, including packed i1
// vectors and power-of-two padding of their innermost dimension.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %wide = aie.buffer(%tile) {sym_name = "wide"} : memref<2xvector<4xi16>>
    %packed = aie.buffer(%tile) {sym_name = "packed"} : memref<2xvector<3x3xi1>>
    aie.runtime_sequence(%four: memref<4xi32>, %one: memref<1xi32>, %value: i32) {
      %last = arith.constant 3 : i32
      aiex.npu.blockwrite(%four) {buffer = @wide, address = 0 : ui32} : memref<4xi32>
      aiex.npu.write32(%last, %value) {buffer = @wide} : i32, i32
      aiex.npu.blockwrite(%one) {buffer = @packed, address = 0 : ui32} : memref<1xi32>
    }
  }
}

// -----

// Model-derived final words: core 65536-4, memtile 524288-4 bytes.
module {
  aie.device(npu2) {
    %core = aie.tile(0, 2)
    %mem = aie.tile(0, 1)
    %core_tail = aie.buffer(%core) {sym_name = "core_tail", address = 65532 : i32, aligned = false} : memref<1xi32>
    %mem_tail = aie.buffer(%mem) {sym_name = "mem_tail", address = 524284 : i32, aligned = false} : memref<1xi32>
    aie.runtime_sequence(%data: memref<1xi32>, %value: i32, %mask: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.blockwrite(%data) {buffer = @core_tail, address = 0 : ui32} : memref<1xi32>
      aiex.npu.blockwrite(%data) {buffer = @mem_tail, address = 0 : ui32} : memref<1xi32>
      aiex.npu.write32(%zero, %value) {buffer = @core_tail} : i32, i32
      aiex.npu.maskwrite32(%zero, %value, %mask) {buffer = @mem_tail} : i32, i32, i32
    }
  }
}

// -----

// Logical tiles supply memory kind without physical placement. A forward named
// buffer reference must not depend on an SSA use or MLIR SymbolOpInterface.
module {
  aie.device(npu2) {
    %tile = aie.logical_tile<CoreTile>(?, ?)
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    %later = aie.buffer(%tile) {sym_name = "later"} : memref<1xi32>
  }
}

// -----

// Valid forward buffers and their physical/logical tile producers remain
// accepted after raw representation checks, including an assigned i32 base.
module {
  aie.device(npu2) {
    aie.runtime_sequence(%value: i32, %mask: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
      aiex.npu.maskwrite32(%zero, %value, %mask) {buffer = @later_mem} : i32, i32, i32
    }
    %tile = "aie.tile"() {col = 0 : i32, row = 2 : i32} : () -> index
    %mem = aie.logical_tile<MemTile>(?, ?)
    %later = "aie.buffer"(%tile) {sym_name = "later", address = 0 : i32} : (index) -> memref<1xi32>
    %later_mem = "aie.buffer"(%mem) {sym_name = "later_mem"} : (index) -> memref<1xi32>
  }
}

// -----

// Queryability is not a scalar-only restriction: default complex layouts and
// custom element types implementing DataLayoutTypeInterface remain supported.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %complex = aie.buffer(%tile) {sym_name = "complex"} : memref<1xcomplex<f32>>
    %block = aie.buffer(%tile) {sym_name = "block"} : memref<1x!aiex.bfp<"v8bfp16ebs8">>
    aie.runtime_sequence(%data: memref<2xi32>) {
      aiex.npu.blockwrite(%data) {buffer = @complex, address = 0 : ui32} : memref<2xi32>
      aiex.npu.blockwrite(%data) {buffer = @block, address = 0 : ui32} : memref<2xi32>
    }
  }
}

// -----

// Layout queries remain scoped to buffer references. The former nested-memref
// payload crash has a raw preservation control and a symbolic good-fit control.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xi32>
    aie.runtime_sequence(%nested: memref<1xmemref<1xi32>>, %word: memref<1xi32>) {
      aiex.npu.blockwrite(%nested) {address = 0 : ui32} : memref<1xmemref<1xi32>>
      aiex.npu.blockwrite(%word) {buffer = @acc, address = 0 : ui32} : memref<1xi32>
    }
  }
}

// -----

// These verifiers check the written interval, not the buffer's unused tail.
// Empty payloads are not newly forbidden, even with a large zero-sized shape.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc", address = 0 : i32} : memref<16385xi32>
    aie.runtime_sequence(%value: i32, %empty: memref<4611686018427387904x0xi32>) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @acc} : i32, i32
      aiex.npu.blockwrite(%empty) {buffer = @acc, address = 0 : ui32} : memref<4611686018427387904x0xi32>
    }
  }
}

// -----

// No new architecture ban on existing writes (AIE1 core data is 32768 bytes).
module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 3)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xi32>
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Raw addresses, including shim register writes and arith-derived addresses,
// retain their existing contract. Payload provenance/shape checks stay scoped
// to symbolic blockwrites, not every raw backend configuration write.
module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence(%address: i32, %value: i32, %mask: i32, %data: memref<?xi16>) {
      %four = arith.constant 4 : i32
      %next = arith.addi %address, %four : i32
      aiex.npu.write32(%address, %value) : i32, i32
      aiex.npu.maskwrite32(%next, %value, %mask) : i32, i32, i32
      aiex.npu.write32(%next, %value) {column = 0 : i32, row = 0 : i32} : i32, i32
      aiex.npu.blockwrite(%data) {address = 0 : ui32, column = 0 : i32, row = 0 : i32} : memref<?xi16>
    }
  }
}

// -----

// Walk through isolated modules/functions after their verification has joined.
// Resolve names in the device, not the nearest nested symbol table, and retain
// valid forward references to buffers that follow the isolated child.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    builtin.module @nested_scope {
      func.func private @acc()
      func.func @write(%value: i32) {
        %zero = arith.constant 0 : i32
        aiex.npu.write32(%zero, %value) {buffer = @acc} : i32, i32
        return
      }
    }
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xi32>
  }
}

// -----

// Identical buffer names in separate devices belong to separate tables.
module {
  aie.device(npu2) @small {
    %tile = aie.tile(0, 2)
    %shared = aie.buffer(%tile) {sym_name = "shared"} : memref<1xi32>
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @shared} : i32, i32
    }
  }
  aie.device(npu2) @large {
    %tile = aie.tile(0, 2)
    %shared = aie.buffer(%tile) {sym_name = "shared"} : memref<2xi32>
    aie.runtime_sequence(%value: i32) {
      %one = arith.constant 1 : i32
      aiex.npu.write32(%one, %value) {buffer = @shared} : i32, i32
    }
  }
}

// -----

// A nested device is verified by its own region hook, with its own names.
module {
  aie.device(npu2) @outer {
    %tile = aie.tile(0, 2)
    %shared = aie.buffer(%tile) {sym_name = "shared"} : memref<1xi32>
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @shared} : i32, i32
    }
    builtin.module @nested {
      aie.device(npu2) @inner {
        %inner_tile = aie.tile(0, 2)
        %inner_buffer = aie.buffer(%inner_tile) {sym_name = "shared"} : memref<2xi32>
        aie.runtime_sequence(%inner_value: i32) {
          %one = arith.constant 1 : i32
          aiex.npu.write32(%one, %inner_value) {buffer = @shared} : i32, i32
        }
      }
    }
  }
}

// -----

// The local scope guard does not impose a device parent on raw writes.
module {
  func.func @raw(%address: i32, %value: i32, %mask: i32, %data: memref<?xi16>) {
    aiex.npu.write32(%address, %value) : i32, i32
    aiex.npu.maskwrite32(%address, %value, %mask) : i32, i32, i32
    aiex.npu.blockwrite(%data) {address = 0 : ui32} : memref<?xi16>
    return
  }
}
