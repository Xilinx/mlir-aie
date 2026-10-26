// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Built only to produce a core ELF for prebaked_elf_reserved.test.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "owned"} : memref<4096xi8>
    func.func private @prebaked_kernel(memref<4096xi8>) attributes {link_with = "prebaked_elf_kernel.o"}
    aie.core(%t) {
      func.call @prebaked_kernel(%buf) : (memref<4096xi8>) -> ()
      aie.end
    } {stack_size = 1024 : i32}
  }
}
