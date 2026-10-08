//===- cpp_full_elf_pdi_lists.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Each xrt-kernel of the full-ELF config lists its own device's PDI and the
// PDIs its sequences load, not every device's: aiebu emits a section per PDI
// per kernel.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t/all %t/other
// RUN: cd %t/all && aiecc --get-full-elf --tmpdir=%t/all %s 2>&1
// RUN: cat %t/all/full_elf_config.json | FileCheck %s
// RUN: cd %t/other && aiecc --get-full-elf --sequence-name=other --tmpdir=%t/other %s 2>&1
// RUN: cat %t/other/full_elf_config.json | FileCheck %s --check-prefix=OTHER

// PDI IDs follow the devices' order: main=1, add_one=2, add_two=3.

// CHECK:      "xrt-kernels": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "PDIs": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "PDI_file": "{{.*}}main.pdi",
// CHECK-NEXT:         "id": 1
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:         "PDI_file": "{{.*}}add_one.pdi",
// CHECK-NEXT:         "id": 2
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:         "PDI_file": "{{.*}}add_two.pdi",
// CHECK-NEXT:         "id": 3
// CHECK-NEXT:       }
// CHECK-NEXT:     ],
// CHECK:          "name": "main"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "PDIs": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "PDI_file": "{{.*}}add_one.pdi",
// CHECK-NEXT:         "id": 2
// CHECK-NEXT:       }
// CHECK-NEXT:     ],
// CHECK:          "name": "add_one"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "PDIs": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "PDI_file": "{{.*}}add_two.pdi",
// CHECK-NEXT:         "id": 3
// CHECK-NEXT:       }
// CHECK-NEXT:     ],
// CHECK:          "name": "add_two"

// Only the selected sequence's loads count: @other configures add_two alone.

// OTHER:      "xrt-kernels": [
// OTHER-NEXT:   {
// OTHER-NEXT:     "PDIs": [
// OTHER-NEXT:       {
// OTHER-NEXT:         "PDI_file": "{{.*}}main.pdi",
// OTHER-NEXT:         "id": 1
// OTHER-NEXT:       },
// OTHER-NEXT:       {
// OTHER-NEXT:         "PDI_file": "{{.*}}add_two.pdi",
// OTHER-NEXT:         "id": 3
// OTHER-NEXT:       }
// OTHER-NEXT:     ],
// OTHER:          "name": "main"
// OTHER-NEXT:   }
// OTHER-NEXT: ]

module {

    aie.device(npu2) @main {
        aie.runtime_sequence @sequence(%arg : memref<16xi32>) {
            aiex.configure @add_one {
                aiex.run @add_one_seq (%arg) : (memref<16xi32>)
            }
            aiex.configure @add_two {
                aiex.run @add_two_seq (%arg) : (memref<16xi32>)
            }
        }
        aie.runtime_sequence @other(%arg : memref<16xi32>) {
            aiex.configure @add_two {
                aiex.run @add_two_seq (%arg) : (memref<16xi32>)
            }
        }
    }

    aie.device(npu2) @add_one {
        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @of_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c1_i32 = arith.constant 1 : i32
            %elem_in = aie.objectfifo.acquire @of_in (Consume, 1) : memref<16xi32>
            %elem_out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
            scf.for %i = %c0 to %c16 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<16xi32>
                %1 = arith.addi %0, %c1_i32 : i32
                memref.store %1, %elem_out[%i] : memref<16xi32>
            }
            aie.objectfifo.release @of_in (Consume, 1)
            aie.objectfifo.release @of_out(Produce, 1)
            aie.end
        }

        aie.runtime_sequence @add_one_seq(%a : memref<16xi32>) {
            %t_in = aiex.dma_configure_task_for @of_in {
                aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @of_out {
                aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }

    aie.device(npu2) @add_two {
        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @of_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @of_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c16 = arith.constant 16 : index
            %c2_i32 = arith.constant 2 : i32
            %elem_in = aie.objectfifo.acquire @of_in (Consume, 1) : memref<16xi32>
            %elem_out = aie.objectfifo.acquire @of_out(Produce, 1) : memref<16xi32>
            scf.for %i = %c0 to %c16 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<16xi32>
                %1 = arith.addi %0, %c2_i32 : i32
                memref.store %1, %elem_out[%i] : memref<16xi32>
            }
            aie.objectfifo.release @of_in (Consume, 1)
            aie.objectfifo.release @of_out(Produce, 1)
            aie.end
        }

        aie.runtime_sequence @add_two_seq(%a : memref<16xi32>) {
            %t_in = aiex.dma_configure_task_for @of_in {
                aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @of_out {
                aie.dma_bd(%a : memref<16xi32> offset = 0 len = 16)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }
}
