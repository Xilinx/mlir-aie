// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{has address space for bank 0 or 2 but is pinned to bank 1}}
    aie.buffer @conflict(%t) {mem_bank = 1 : i32} : memref<64xi8, 10>
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{address attribute places the buffer in bank 1, which is excluded by its memory space}}
    aie.buffer @wrong_address(%t) {address = 16384 : i32} : memref<64xi8, 10>
  }
}

// -----

// Checking only the start bank would overlook the forbidden bank B.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{address attribute places the buffer in bank 1, which is excluded by its memory space}}
    aie.buffer @crossing_address(%t) {address = 16352 : i32, mem_bank = 0 : i32} : memref<64xi8, 10>
  }
}

// -----

// Nor may the allocator span a forbidden bank between a disjoint pair.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{requires 32768 bytes, but no contiguous aligned space remains within allowed banks 0 and 2}}
    aie.buffer @too_large(%t) : memref<32768xi8, 10>
  }
}

// -----

// Free banks outside the pair do not make a full pair usable.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @a(%t) {address = 0 : i32} : memref<16384xi8>
    aie.buffer @c(%t) {address = 32768 : i32} : memref<16384xi8>
    // expected-error @+1 {{requires 64 bytes, but no contiguous aligned space remains within allowed banks 0 and 2}}
    aie.buffer @full_pair(%t) : memref<64xi8, 10>
  }
}

// -----

// A single-bank resource must also reject a straddling explicit address.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{address attribute places the buffer in bank 1, which is excluded by its memory space}}
    aie.buffer @single_crossing(%t) {address = 16352 : i32} : memref<64xi8, 5>
  }
}

// -----

// A conflicting single-bank address used to overwrite the address-space pin.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{address attribute places the buffer in bank 2, which is excluded by its memory space}}
    aie.buffer @single_address(%t) {address = 32768 : i32} : memref<64xi8, 6>
  }
}

// -----

// A compatible mem_bank narrows the pair: free bank A is not a fallback.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @c(%t) {address = 32768 : i32} : memref<16384xi8>
    // expected-error @+1 {{requires 64 bytes in bank 2, but only 0 of 16384 bytes are free there}}
    aie.buffer @narrowed(%t) {mem_bank = 2 : i32} : memref<64xi8, 10>
  }
}
