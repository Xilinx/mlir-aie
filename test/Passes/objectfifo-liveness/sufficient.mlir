// RUN: aie-opt --aie-objectfifo-liveness --verify-diagnostics %s
module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_1 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_2 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_3 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_4 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_5 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_6 = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_7 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_8 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_9 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_10 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_11 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_12 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_13 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_14 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_15 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_16 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_17 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_18 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_19 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_20 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_21 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_22 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_23 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_24 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_25 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_26 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_27 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_28 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @inA_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_0(%logical_mem dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_0] -> [@memA_0]([] [0])
    aie.objectfifo @inA_1(%logical_shim_noc_7, {%logical_mem_8}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_1(%logical_mem_8 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_1] -> [@memA_1]([] [0])
    aie.objectfifo @inA_2(%logical_shim_noc_9, {%logical_mem_10}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_2(%logical_mem_10 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_3}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_2] -> [@memA_2]([] [0])
    aie.objectfifo @inA_3(%logical_shim_noc_11, {%logical_mem_12}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_3(%logical_mem_12 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_5}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_3] -> [@memA_3]([] [0])
    aie.objectfifo @inW1_0(%logical_shim_noc_13, {%logical_mem_14}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo @memW1_0(%logical_mem_14 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_core, %logical_core_1, %logical_core_3, %logical_core_5}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo.link [@inW1_0] -> [@memW1_0]([] [0])
    aie.objectfifo @inW2_0(%logical_shim_noc_15, {%logical_mem_16}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo @memW2_0(%logical_mem_16 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_core_0, %logical_core_2, %logical_core_4, %logical_core_6}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo.link [@inW2_0] -> [@memW2_0]([] [0])
    aie.objectfifo @memC_0(%logical_core_0, {%logical_mem_17}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_0(%logical_mem_17 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_25}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_0] -> [@outC_0]([] [0])
    aie.objectfifo @memC_1(%logical_core_2, {%logical_mem_18}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_1(%logical_mem_18 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_26}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_1] -> [@outC_1]([] [0])
    aie.objectfifo @memC_2(%logical_core_4, {%logical_mem_19}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_2(%logical_mem_19 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_27}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_2] -> [@outC_2]([] [0])
    aie.objectfifo @memC_3(%logical_core_6, {%logical_mem_20}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_3(%logical_mem_20 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_28}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_3] -> [@outC_3]([] [0])
    aie.objectfifo @memH_0(%logical_mem_21 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_0}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_0(%logical_core, {%logical_mem_21 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_0] -> [@memH_0]([] [0])
    aie.objectfifo @memH_1(%logical_mem_22 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_2}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_1(%logical_core_1, {%logical_mem_22 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_1] -> [@memH_1]([] [0])
    aie.objectfifo @memH_2(%logical_mem_23 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_4}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_2(%logical_core_3, {%logical_mem_23 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_2] -> [@memH_2]([] [0])
    aie.objectfifo @memH_3(%logical_mem_24 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_6}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_3(%logical_core_5, {%logical_mem_24 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_3] -> [@memH_3]([] [0])
    func.func private @zero_bf16(memref<16x32xbf16>) attributes {link_with = "mm_16x32x32.o"}
    func.func private @matmul_bf16_bf16(memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) attributes {link_with = "mm_16x32x32.o"}
    func.func private @zero_f32(memref<16x32xf32>) attributes {link_with = "mm_16x32x32.o"}
    func.func private @matmul_bf16_f32(memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) attributes {link_with = "mm_16x32x32.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %8 = aie.objectfifo.acquire @ofH_0(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%9) : (memref<16x32xbf16>) -> ()
          %c0_31 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_32 = arith.constant 1 : index
          scf.for %arg2 = %c0_31 to %c4 step %c1_32 {
            %10 = aie.objectfifo.acquire @memA_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %12 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%11, %13, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_0(Consume, 1)
            aie.objectfifo.release @memW1_0(Consume, 1)
          }
          aie.objectfifo.release @ofH_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %1 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %8 = aie.objectfifo.acquire @memC_0(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%9) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%10) : (memref<16x32xf32>) -> ()
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %11 = aie.objectfifo.acquire @memH_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %13 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%12, %14, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_0(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %15 = aie.objectfifo.acquire @memH_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %17 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%16, %18, %10) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_0(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
        }
        aie.objectfifo.release @memC_0(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %2 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %8 = aie.objectfifo.acquire @ofH_1(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%9) : (memref<16x32xbf16>) -> ()
          %c0_31 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_32 = arith.constant 1 : index
          scf.for %arg2 = %c0_31 to %c4 step %c1_32 {
            %10 = aie.objectfifo.acquire @memA_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %12 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%11, %13, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_1(Consume, 1)
            aie.objectfifo.release @memW1_0(Consume, 1)
          }
          aie.objectfifo.release @ofH_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %3 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %8 = aie.objectfifo.acquire @memC_1(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%9) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%10) : (memref<16x32xf32>) -> ()
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %11 = aie.objectfifo.acquire @memH_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %13 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%12, %14, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_1(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %15 = aie.objectfifo.acquire @memH_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %17 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%16, %18, %10) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_1(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
        }
        aie.objectfifo.release @memC_1(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %4 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %8 = aie.objectfifo.acquire @ofH_2(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%9) : (memref<16x32xbf16>) -> ()
          %c0_31 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_32 = arith.constant 1 : index
          scf.for %arg2 = %c0_31 to %c4 step %c1_32 {
            %10 = aie.objectfifo.acquire @memA_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %12 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%11, %13, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_2(Consume, 1)
            aie.objectfifo.release @memW1_0(Consume, 1)
          }
          aie.objectfifo.release @ofH_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %5 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %8 = aie.objectfifo.acquire @memC_2(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%9) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%10) : (memref<16x32xf32>) -> ()
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %11 = aie.objectfifo.acquire @memH_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %13 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%12, %14, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_2(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %15 = aie.objectfifo.acquire @memH_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %17 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%16, %18, %10) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_2(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
        }
        aie.objectfifo.release @memC_2(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %6 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %8 = aie.objectfifo.acquire @ofH_3(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%9) : (memref<16x32xbf16>) -> ()
          %c0_31 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_32 = arith.constant 1 : index
          scf.for %arg2 = %c0_31 to %c4 step %c1_32 {
            %10 = aie.objectfifo.acquire @memA_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %12 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%11, %13, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_3(Consume, 1)
            aie.objectfifo.release @memW1_0(Consume, 1)
          }
          aie.objectfifo.release @ofH_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %7 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %8 = aie.objectfifo.acquire @memC_3(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %10 = aie.objectfifo.subview.access %8[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%9) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%10) : (memref<16x32xf32>) -> ()
        %c0_29 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_30 = arith.constant 1 : index
        scf.for %arg1 = %c0_29 to %c8 step %c1_30 {
          %11 = aie.objectfifo.acquire @memH_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %13 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %14 = aie.objectfifo.subview.access %13[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%12, %14, %9) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_3(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %15 = aie.objectfifo.acquire @memH_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %16 = aie.objectfifo.subview.access %15[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %17 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %18 = aie.objectfifo.subview.access %17[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%16, %18, %10) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_3(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
        }
        aie.objectfifo.release @memC_3(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<32768xbf16>, %arg2: memref<16384xbf16>, %arg3: memref<4096xf32>) {
      %8 = aiex.dma_configure_task_for @outC_0 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @outC_0 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 32, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @outC_1 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 1024, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @outC_1 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 1056, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @outC_2 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 2048, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @outC_2 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 2080, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @outC_3 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 3072, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @outC_3 {
        aie.dma_bd(%arg3 : memref<4096xf32>, 3104, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @inA_0 {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @inA_1 {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 2048, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @inA_2 {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 4096, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @inA_3 {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 6144, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @inW1_0 {
        aie.dma_bd(%arg1 : memref<32768xbf16>, 0, 4096, [<size = 8, stride = 32>, <size = 4, stride = 8192>, <size = 32, stride = 256>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 8192, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 10240, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 14336, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%28)
      aiex.dma_await_task(%8)
      aiex.dma_await_task(%9)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%11)
      aiex.dma_await_task(%12)
      aiex.dma_await_task(%13)
      aiex.dma_await_task(%14)
      aiex.dma_await_task(%15)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%20)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%23)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%26)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
    }
  }
}

