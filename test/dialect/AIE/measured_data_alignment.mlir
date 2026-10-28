// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {measured_data_alignment = 4096 : i32, measured_bank_alignments = array<i32: 1, 4096, 16384, 65536>}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @below {{measured_data_alignment must be a power of two between 1 and 65536 bytes}}
    %core = aie.core(%t) { aie.end } {measured_data_alignment = 3 : i32}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @below {{measured_data_alignment must be a power of two between 1 and 65536 bytes}}
    %core = aie.core(%t) { aie.end } {measured_data_alignment = 1073741824 : i32}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @below {{measured_bank_alignments must contain only powers of two between 1 and 65536 bytes}}
    %core = aie.core(%t) { aie.end } {measured_bank_alignments = array<i32: 0>}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @below {{measured_bank_alignments must contain only powers of two between 1 and 65536 bytes}}
    %core = aie.core(%t) { aie.end } {measured_bank_alignments = array<i32: -1>}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @below {{measured_bank_alignments must contain only powers of two between 1 and 65536 bytes}}
    %core = aie.core(%t) { aie.end } {measured_bank_alignments = array<i32: 1, 3>}
  }
}

// -----

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    // expected-error @below {{measured_bank_alignments must contain only powers of two between 1 and 65536 bytes}}
    %core = aie.core(%t) { aie.end } {measured_bank_alignments = array<i32: 1073741824>}
  }
}
