// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each device a load_pdi names becomes a same-named cert.section in the main
// device. That includes a device loaded only from another absorbed device's
// configuration, whether that one was loaded or configured. A symbol of another
// kind that already holds the name is an error, not a reason to skip the
// section.

// RUN: aie-opt --aie-npu-to-cert --split-input-file --verify-diagnostics %s | FileCheck %s

// @seg_b is loaded only from @seg_a's configuration.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_a)
// CHECK: aiex.cert.section @seg_a {
// CHECK: aiex.cert.write32(20480, 1)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_b)
// CHECK: aiex.cert.section @seg_b {
// CHECK: aiex.cert.write32(20480, 2)
// CHECK-NOT: aie.device(xcve3858) @
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg_a}
    }
  }
  aie.device(xcve3858) @seg_a {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      aiex.npu.load_pdi {device_ref = @seg_b}
    }
  }
  aie.device(xcve3858) @seg_b {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 2 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// The same, with @seg_a reached through aiex.configure.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_a)
// CHECK-NEXT: aiex.cert.write32(24576, 9)
// CHECK: aiex.cert.section @seg_a {
// CHECK: aiex.cert.write32(20480, 1)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_b)
// CHECK: aiex.cert.section @seg_b {
// CHECK: aiex.cert.write32(20480, 2)
// CHECK-NOT: aie.device(xcve3858) @
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @seq() {
      aiex.configure @seg_a {
        %a = arith.constant 0x6000 : i32
        %v = arith.constant 9 : i32
        aiex.npu.write32(%a, %v) : i32, i32
      }
    }
  }
  aie.device(xcve3858) @seg_a {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      aiex.npu.load_pdi {device_ref = @seg_b}
    }
  }
  aie.device(xcve3858) @seg_b {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 2 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// A runtime sequence of the main device is named after the loaded device.
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @seg_a() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 3 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @main_seq() {
      // expected-error@+1 {{cannot absorb device @seg_a as a cert.section: @seg_a is already defined by 'aie.runtime_sequence'}}
      aiex.npu.load_pdi {device_ref = @seg_a}
    }
  }
  aie.device(xcve3858) @seg_a {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// A mixed chain: @seg_a is configured, its configuration loads @seg_b,
// @seg_b's configuration configures @seg_c, and @seg_c's configuration
// configures @seg_d and runs one of its sequences. Every configure is lowered,
// including the ones cloned in by absorbing @seg_b and by lowering the
// configure of @seg_c, and the run is inlined before its configure is lowered.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_a)
// CHECK-NEXT: aiex.cert.write32(24576, 8)
// CHECK: aiex.cert.section @seg_a {
// CHECK: aiex.cert.write32(20480, 1)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_b)
// CHECK: aiex.cert.section @seg_b {
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_c)
// CHECK-NEXT: aiex.cert.write32(24576, 7)
// CHECK: aiex.cert.section @seg_c {
// CHECK: aiex.cert.write32(20480, 3)
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_d)
// CHECK-NEXT: aiex.cert.write32(28672, 5)
// CHECK: aiex.cert.section @seg_d {
// CHECK: aiex.cert.write32(20480, 4)
// CHECK-NOT: aiex.configure
// CHECK-NOT: aiex.run
// CHECK-NOT: aie.device(xcve3858) @
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @seq() {
      aiex.configure @seg_a {
        %a = arith.constant 0x6000 : i32
        %v = arith.constant 8 : i32
        aiex.npu.write32(%a, %v) : i32, i32
      }
    }
  }
  aie.device(xcve3858) @seg_a {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      aiex.npu.load_pdi {device_ref = @seg_b}
    }
  }
  aie.device(xcve3858) @seg_b {
    aie.runtime_sequence @configure() {
      aiex.configure @seg_c {
        %a = arith.constant 0x6000 : i32
        %v = arith.constant 7 : i32
        aiex.npu.write32(%a, %v) : i32, i32
      }
    }
  }
  aie.device(xcve3858) @seg_c {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 3 : i32
      aiex.npu.write32(%a, %v) : i32, i32
      aiex.configure @seg_d {
        %w = arith.constant 0 : i32
        aiex.run @work(%w) : (i32)
      }
    }
  }
  aie.device(xcve3858) @seg_d {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 4 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
    aie.runtime_sequence @work(%unused: i32) {
      %a = arith.constant 0x7000 : i32
      %v = arith.constant 5 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}
