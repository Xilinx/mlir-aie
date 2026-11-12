//===- aiecc_options.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Rules between aiecc's reconfiguration options.

// RUN: not %aiecc --expand-load-pdis --get-ctrlpkt %s 2>&1 | FileCheck %s --check-prefix=EXPAND-CTRLPKT
// EXPAND-CTRLPKT: aiecc: --expand-load-pdis and --get-ctrlpkt are mutually exclusive

// RUN: not %aiecc --expand-load-pdis --load-pdi-to-ctrl-pkt %s 2>&1 | FileCheck %s --check-prefix=EXPAND-LDPDI-CTRLPKT
// EXPAND-LDPDI-CTRLPKT: aiecc: --expand-load-pdis and --load-pdi-to-ctrl-pkt are mutually exclusive

// RUN: not %aiecc --get-full-elf --get-ctrlpkt %s 2>&1 | FileCheck %s --check-prefix=FULL-ELF-CTRLPKT
// FULL-ELF-CTRLPKT: aiecc: --generate-full-elf and --aie-generate-ctrlpkt together also requires --load-pdi-to-ctrl-pkt

// RUN: not %aiecc --expand-load-pdis --reconfig-teardown %s 2>&1 | FileCheck %s --check-prefix=TEARDOWN-FULL-ELF
// TEARDOWN-FULL-ELF: aiecc: --reconfig-teardown requires full-ELF output

// RUN: not %aiecc --get-full-elf --reconfig-teardown %s 2>&1 | FileCheck %s --check-prefix=TEARDOWN-EXPAND
// TEARDOWN-EXPAND: aiecc: --reconfig-teardown requires --expand-load-pdis or --load-pdi-to-ctrl-pkt

// RUN: not %aiecc --get-full-elf --expand-load-pdis --generate-ctrl-pkt-overlay --reconfig-teardown %s 2>&1 | FileCheck %s --check-prefix=TEARDOWN-OVERLAY
// TEARDOWN-OVERLAY: aiecc: --reconfig-teardown requires that --generate-ctrl-pkt-overlay is not set

// RUN: not %aiecc --get-full-elf --expand-load-pdis --reconfig-init-entry %s 2>&1 | FileCheck %s --check-prefix=INIT-TEARDOWN
// INIT-TEARDOWN: aiecc: --reconfig-init-entry requires --reconfig-teardown


// RUN: not %aiecc --ctrlpkt-adaptive-overlay %s 2>&1 | FileCheck %s --check-prefix=ADAPTIVE-LDPDI-CTRLPKT
// ADAPTIVE-LDPDI-CTRLPKT: aiecc: --ctrlpkt-adaptive-overlay requires --generate-ctrl-pkt-overlay or --load-pdi-to-ctrl-pkt

// RUN: not %aiecc --ctrlpkt-parallel-shims %s 2>&1 | FileCheck %s --check-prefix=PARALLEL-LDPDI-CTRLPKT
// PARALLEL-LDPDI-CTRLPKT: aiecc: --ctrlpkt-parallel-shims requires --load-pdi-to-ctrl-pkt

// RUN: not %aiecc --ctrlpkt-multicast-span=2 %s 2>&1 | FileCheck %s --check-prefix=SPAN-LDPDI-CTRLPKT
// SPAN-LDPDI-CTRLPKT: aiecc: --ctrlpkt-multicast-span requires --load-pdi-to-ctrl-pkt

// Under --load-pdi-to-ctrl-pkt the teardown is off unless asked for.
// RUN: not %aiecc --get-full-elf --load-pdi-to-ctrl-pkt --reconfig-init-entry %s 2>&1 | FileCheck %s --check-prefix=INIT-TEARDOWN
// RUN: %aiecc -n --get-full-elf --load-pdi-to-ctrl-pkt --reconfig-teardown --reconfig-init-entry %s

// The full ELF can also be requested by its output name.
// RUN: %aiecc -n --get=aie.elf --expand-load-pdis --reconfig-teardown --reconfig-init-entry %s

// Turning an option off explicitly is never an error.
// RUN: %aiecc -n --reconfig-teardown=false --reconfig-init-entry=false --ctrlpkt-adaptive-overlay=false --ctrlpkt-parallel-shims=false --ctrlpkt-multicast-span=0 %s

module {
  aie.device(npu2) @main {
  }
}
