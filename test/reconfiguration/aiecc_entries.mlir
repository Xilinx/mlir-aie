//===- aiecc_entries.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Full-ELF entries of a two-configuration program under the reconfiguration
// options. --sequence-name keeps the init entry. The configurations have no
// cores, so nothing needs compiling.

// REQUIRES: aiebu

// RUN: rm -rf %t && mkdir -p %t/whole %t/init %t/select
// RUN: cd %t/whole && %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --tmpdir=prj %s
// RUN: FileCheck %s --check-prefix=WHOLE < prj/full_elf_config.json
// RUN: cd %t/init && %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --reconfig-init-entry --tmpdir=prj %s
// RUN: FileCheck %s --check-prefix=INIT < prj/full_elf_config.json
// RUN: cd %t/select && %aiecc --get-full-elf --expand-load-pdis --reconfig-teardown --reconfig-init-entry --sequence-name=sequence --tmpdir=prj %s
// RUN: FileCheck %s --check-prefix=SELECT < prj/full_elf_config.json

// WHOLE:          "instance": [
// WHOLE-NOT:      "id"
// WHOLE:          "id": "sequence"
// WHOLE-NEXT:   }
// WHOLE-NEXT:   ],
// WHOLE-NEXT:   "name": "main"

// INIT:           "instance": [
// INIT-NOT:       "id"
// INIT:           "id": "reconfig_init"
// INIT-NOT:       "id"
// INIT:           "id": "sequence"
// INIT-NEXT:    }
// INIT-NEXT:    ],
// INIT-NEXT:    "name": "main"

// SELECT:         "instance": [
// SELECT-NOT:     "id"
// SELECT:         "id": "reconfig_init"
// SELECT-NOT:     "id"
// SELECT:         "id": "sequence"
// SELECT-NEXT:  }
// SELECT-NEXT:  ],
// SELECT-NEXT:  "name": "main"

module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @sequence(%x: memref<1xi32>) {
      aiex.configure @a {
        aiex.run @run(%x) : (memref<1xi32>)
      }
      aiex.configure @b {
        aiex.run @run(%x) : (memref<1xi32>)
      }
    }
  }

  aie.device(npu2_1col) @a {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 1, DMA : 0>
    }
    aie.runtime_sequence @run(%x: memref<1xi32>) {
    }
  }

  aie.device(npu2_1col) @b {
    %t = aie.tile(0, 2)
    aie.switchbox(%t) {
      aie.connect<South : 2, DMA : 1>
    }
    aie.runtime_sequence @run(%x: memref<1xi32>) {
    }
  }
}
