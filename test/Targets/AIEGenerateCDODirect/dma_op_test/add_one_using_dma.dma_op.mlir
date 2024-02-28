// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename -s .dma_op.mlir %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.dma_op.prj $BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_start.prj && pushd $BASENAME.dma_start.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %S/$BASENAME.dma_start && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_start.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_op.prj && pushd $BASENAME.dma_op.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %s && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_op.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_op.prj

// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_error_handling.bin $BASENAME.dma_start.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_init.bin $BASENAME.dma_start.prj/aie_cdo_init.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_elfs.bin $BASENAME.dma_start.prj/aie_cdo_elfs.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_enable.bin $BASENAME.dma_start.prj/aie_cdo_enable.bin

module {
  aie.device(ipu) {
    memref.global "public" @objFifo_in0 : memref<16xi32>
    memref.global "public" @objFifo_in0_cons : memref<16xi32>
    memref.global "public" @objFifo_in1 : memref<8xi32>
    memref.global "public" @objFifo_in1_cons : memref<8xi32>
    memref.global "public" @objFifo_out0 : memref<16xi32>
    memref.global "public" @objFifo_out0_cons : memref<16xi32>
    memref.global "public" @objFifo_out1 : memref<8xi32>
    memref.global "public" @objFifo_out1_cons : memref<8xi32>

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    %objFifo_in0_cons_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_0"} : memref<16xi32>
    %objFifo_in0_cons_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_in0_cons_buff_1"} : memref<16xi32>
    %objFifo_out0_buff_0 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_0"} : memref<16xi32>
    %objFifo_out0_buff_1 = aie.buffer(%tile_0_1) {sym_name = "objFifo_out0_buff_1"} : memref<16xi32>

    %objFifo_in1_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_0"} : memref<8xi32>
    %objFifo_in1_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_in1_cons_buff_1"} : memref<8xi32>
    %objFifo_out1_buff_0 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_0"} : memref<8xi32>
    %objFifo_out1_buff_1 = aie.buffer(%tile_0_2) {sym_name = "objFifo_out1_buff_1"} : memref<8xi32>

    %objFifo_in0_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "objFifo_in0_prod_lock"}
    %objFifo_in0_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_lock"}
    %objFifo_out0_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "objFifo_out0_cons_prod_lock"}
    %objFifo_out0_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_cons_lock"}

    %objFifo_in0_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "objFifo_in0_cons_prod_lock"}
    %objFifo_in0_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "objFifo_in0_cons_cons_lock"}
    %objFifo_out0_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "objFifo_out0_prod_lock"}
    %objFifo_out0_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "objFifo_out0_cons_lock"}

    %objFifo_in1_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "objFifo_in1_cons_prod_lock"}
    %objFifo_in1_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "objFifo_in1_cons_cons_lock"}
    %objFifo_out1_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "objFifo_out1_prod_lock"}
    %objFifo_out1_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "objFifo_out1_cons_lock"}

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)

    %core_0_2 = aie.core(%tile_0_2) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c8 step %c2 {
        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)

        scf.for %arg1 = %c0 to %c8 step %c1 {
          %0 = memref.load %objFifo_in1_cons_buff_0[%arg1] : memref<8xi32>
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %objFifo_out1_buff_0[%arg1] : memref<8xi32>
        }

        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)

        aie.use_lock(%objFifo_in1_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%objFifo_out1_prod_lock, AcquireGreaterEqual, 1)

        scf.for %arg1 = %c0 to %c8 step %c1 {
          %0 = memref.load %objFifo_in1_cons_buff_1[%arg1] : memref<8xi32>
          %1 = arith.addi %0, %c1_i32 : i32
          memref.store %1, %objFifo_out1_buff_1[%arg1] : memref<8xi32>
        }

        aie.use_lock(%objFifo_in1_cons_prod_lock, Release, 1)
        aie.use_lock(%objFifo_out1_cons_lock, Release, 1)
      }
      aie.end
    }

    aie.shim_dma_allocation @objFifo_in0(MM2S, 0, 0)

    func.func @bobsyouruncle(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c64_i64 = arith.constant 64 : i64
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64]) {id = 0 : i64, metadata = @objFifo_in0} : memref<64xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64] [%c1_i64, %c1_i64, %c1_i64, %c64_i64] [%c0_i64, %c0_i64, %c0_i64]) {id = 1 : i64, metadata = @objFifo_out0} : memref<64xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_in0_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_cons_lock, Release, 1)
      }]
      aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_in0_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in0_cons_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_in0_cons_prod_lock, Release, 1)
      }]
      aie.dma(MM2S, 1) [{
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_out0_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_prod_lock, Release, 1)
      }]
      aie.dma(S2MM, 1) [{
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_0 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_out0_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out0_buff_1 : memref<16xi32>, 0, 16)
        aie.use_lock(%objFifo_out0_cons_lock, Release, 1)
      }]
      aie.end
    }

    aie.shim_dma_allocation @objFifo_out0(S2MM, 0, 0)

    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.dma(S2MM, 0) [{
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_0 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_in1_cons_prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_in1_cons_buff_1 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_in1_cons_cons_lock, Release, 1)
      }]
      aie.dma(MM2S, 0) [{
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_0 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }, {
        aie.use_lock(%objFifo_out1_cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%objFifo_out1_buff_1 : memref<8xi32>, 0, 8)
        aie.use_lock(%objFifo_out1_prod_lock, Release, 1)
      }]
      aie.end
    }
  }
}

