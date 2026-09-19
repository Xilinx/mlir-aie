// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Reuses the ELF prebaked_elf_producer.mlir built, and asks for a buffer big
// enough to collide with it. See prebaked_elf_reserved.test.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %scratch = aie.buffer(%t) {sym_name = "scratch"} : memref<16384xi8>
    aie.core(%t) { aie.end } {elf_file = "prebaked.elf", stack_size = 1024 : i32}
  }
}
