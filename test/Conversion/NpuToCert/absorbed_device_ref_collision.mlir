// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Absorbed devices become same-named cert.sections in the main device. Payload
// globals that collide with a device name must be renamed, wherever they live,
// and only payload references may follow the rename.
// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-npu-to-cert --split-input-file %s | FileCheck %s --check-prefix=REFS

// No load_pdi may follow a payload rename.
// REFS-NOT: load_pdi({{.*}}_{{[0-9]+}})

// An absorbed device holds a global named after a device it also loads. The
// global is renamed and its blockwrite follows; both load_pdi refs stay on the
// device, which becomes a section.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @[[G:seg_empty_[0-9]+]] : memref<2xi32> = dense<[11, 12]>
// CHECK-DAG: aiex.cert.uc_dma_bd @[[G]], 20480
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_a)
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @seg_empty)
// CHECK: aiex.cert.section @seg_a {
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_empty)
// CHECK: aiex.cert.section @seg_empty {
// CHECK-NOT: aie.device(xcve3858) @
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg_a}
      aiex.npu.load_pdi {device_ref = @seg_empty}
    }
  }
  aie.device(xcve3858) @seg_a {
    memref.global "private" constant @seg_empty : memref<2xi32> = dense<[11, 12]>
    aie.runtime_sequence @configure() {
      %g = memref.get_global @seg_empty : memref<2xi32>
      aiex.npu.blockwrite(%g) {address = 20480 : ui32} : memref<2xi32>
      aiex.npu.load_pdi {device_ref = @seg_empty}
    }
  }
  aie.device(xcve3858) @seg_empty {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// The main device's own global collides with a device name. It is renamed too,
// along with its blockwrite, and the load still gets its section.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @[[M:seg_cfg_[0-9]+]] : memref<2xi32> = dense<[1, 2]>
// CHECK-DAG: aiex.cert.uc_dma_bd @[[M]], 16384
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_cfg)
// CHECK: aiex.cert.section @seg_cfg {
// CHECK-NOT: aie.device(xcve3858) @
module {
  aie.device(xcve3858) @main {
    memref.global "private" constant @seg_cfg : memref<2xi32> = dense<[1, 2]>
    aie.runtime_sequence @main_seq() {
      %g = memref.get_global @seg_cfg : memref<2xi32>
      aiex.npu.blockwrite(%g) {address = 16384 : ui32} : memref<2xi32>
      aiex.npu.load_pdi {device_ref = @seg_cfg}
    }
  }
  aie.device(xcve3858) @seg_cfg {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// Ordinary hoisting collisions: @seg_a's @x is renamed past its own @x_0, and
// device @x's own @x is renamed past both. Every payload keeps its blockwrite,
// and no rename reaches a load_pdi of device @x.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: memref.global "private" constant @[[A:x_[0-9]+]] : memref<2xi32> = dense<[1, 2]>
// CHECK-DAG: memref.global "private" constant @x_0 : memref<2xi32> = dense<[3, 4]>
// CHECK-DAG: memref.global "private" constant @[[C:x_[0-9]+]] : memref<2xi32> = dense<[5, 6]>
// CHECK-DAG: aiex.cert.uc_dma_bd @[[A]], 4096
// CHECK-DAG: aiex.cert.uc_dma_bd @x_0, 8192
// CHECK-DAG: aiex.cert.uc_dma_bd @[[C]], 12288
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @seg_a)
// CHECK-NEXT: aiex.cert.load_pdi({{[0-9]+}}, @x)
// CHECK: aiex.cert.section @seg_a {
// CHECK: aiex.cert.load_pdi({{[0-9]+}}, @x)
// CHECK: aiex.cert.section @x {
// CHECK-NOT: aie.device(xcve3858) @
module {
  aie.device(xcve3858) @main {
    aie.runtime_sequence @main_seq() {
      aiex.npu.load_pdi {device_ref = @seg_a}
      aiex.npu.load_pdi {device_ref = @x}
    }
  }
  aie.device(xcve3858) @seg_a {
    memref.global "private" constant @x : memref<2xi32> = dense<[1, 2]>
    memref.global "private" constant @x_0 : memref<2xi32> = dense<[3, 4]>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @x : memref<2xi32>
      aiex.npu.blockwrite(%a) {address = 4096 : ui32} : memref<2xi32>
      %b = memref.get_global @x_0 : memref<2xi32>
      aiex.npu.blockwrite(%b) {address = 8192 : ui32} : memref<2xi32>
      aiex.npu.load_pdi {device_ref = @x}
    }
  }
  aie.device(xcve3858) @x {
    memref.global "private" constant @x : memref<2xi32> = dense<[5, 6]>
    aie.runtime_sequence @configure() {
      %a = memref.get_global @x : memref<2xi32>
      aiex.npu.blockwrite(%a) {address = 12288 : ui32} : memref<2xi32>
    }
  }
}

// -----

// A replacement name must also avoid SSA ops carrying a sym_name, which the
// device's symbol table does not index but its verifier still rejects.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK-DAG: aie.buffer({{.*}}sym_name = "seg_buf_0"
// CHECK-DAG: memref.global "private" constant @seg_buf_1 : memref<2xi32> = dense<[1, 2]>
// CHECK-DAG: aiex.cert.uc_dma_bd @seg_buf_1, 16384
// CHECK: aiex.cert.section @seg_buf {
module {
  aie.device(xcve3858) @main {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) {sym_name = "seg_buf_0", address = 1024 : i32} : memref<1xi32>
    memref.global "private" constant @seg_buf : memref<2xi32> = dense<[1, 2]>
    aie.runtime_sequence @main_seq() {
      %g = memref.get_global @seg_buf : memref<2xi32>
      aiex.npu.blockwrite(%g) {address = 16384 : ui32} : memref<2xi32>
      aiex.npu.load_pdi {device_ref = @seg_buf}
    }
  }
  aie.device(xcve3858) @seg_buf {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}

// -----

// A nested symbol table resolves its own names: renaming the main device's
// @seg_nest must not redirect the nested module's use of its own @seg_nest.
//
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: memref.global "private" constant @seg_nest_{{[0-9]+}} : memref<1xi32> = dense<1>
// CHECK: builtin.module @nested {
// CHECK: memref.global "private" constant @seg_nest : memref<1xi32> = dense<2>
// CHECK: memref.get_global @seg_nest : memref<1xi32>
// CHECK: aiex.cert.section @seg_nest {
module {
  aie.device(xcve3858) @main {
    memref.global "private" constant @seg_nest : memref<1xi32> = dense<1>
    builtin.module @nested {
      memref.global "private" constant @seg_nest : memref<1xi32> = dense<2>
      memref.global "private" constant @seg_nest_0 : memref<1xi32> = dense<3>
      func.func @f() -> memref<1xi32> {
        %g = memref.get_global @seg_nest : memref<1xi32>
        return %g : memref<1xi32>
      }
    }
    aie.runtime_sequence @main_seq() {
      %g = memref.get_global @seg_nest : memref<1xi32>
      aiex.npu.blockwrite(%g) {address = 16384 : ui32} : memref<1xi32>
      aiex.npu.load_pdi {device_ref = @seg_nest}
    }
  }
  aie.device(xcve3858) @seg_nest {
    aie.runtime_sequence @configure() {
      %a = arith.constant 0x5000 : i32
      %v = arith.constant 1 : i32
      aiex.npu.write32(%a, %v) : i32, i32
    }
  }
}
