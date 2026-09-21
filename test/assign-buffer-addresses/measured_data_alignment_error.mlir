// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file --verify-diagnostics %s

// Disabling vector alignment does not disable the linker's section alignment.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{data reservation address must be aligned to 256 bytes to satisfy its measured section alignment}}
    %data = aie.buffer(%t) {sym_name = "data", core_data, address = 1088 : i32, aligned = false} : memref<64xi8>
    aie.core(%t) {
      aie.end
    } {data_size = 64 : i32, measured_data_alignment = 256 : i32}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{data reservation address must be aligned to 4096 bytes to satisfy its measured section alignment}}
    %data = aie.buffer(%t) {sym_name = "bank_reserved_0_2_0", bank_reserved, mem_bank = 0 : i32, address = 1024 : i32, aligned = false} : memref<4096xi8>
    aie.core(%t) {
      aie.end
    } {measured_bank_alignments = array<i32: 4096, 1, 1, 1>}
  }
}
