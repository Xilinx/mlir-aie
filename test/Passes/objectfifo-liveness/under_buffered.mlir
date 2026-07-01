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
    %logical_core_7 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_8 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_9 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_10 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_11 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_12 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_13 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_14 = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_15 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_16 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_17 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_18 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_19 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_20 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_21 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_22 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_23 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_24 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_25 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_26 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_27 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_28 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_29 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_30 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_31 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_32 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_33 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_34 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_35 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_36 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_37 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_38 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_39 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_40 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_41 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_42 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_43 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_44 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_45 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_46 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_47 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_48 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_49 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_50 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_51 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_52 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_53 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_54 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_55 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_56 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_57 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_58 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_59 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_60 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @inA_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_0(%logical_mem dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_0] -> [@memA_0]([] [0])
    aie.objectfifo @inA_1(%logical_shim_noc_15, {%logical_mem_16}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_1(%logical_mem_16 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_1] -> [@memA_1]([] [0])
    aie.objectfifo @inA_2(%logical_shim_noc_17, {%logical_mem_18}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_2(%logical_mem_18 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_3}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_2] -> [@memA_2]([] [0])
    aie.objectfifo @inA_3(%logical_shim_noc_19, {%logical_mem_20}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_3(%logical_mem_20 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_5}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_3] -> [@memA_3]([] [0])
    aie.objectfifo @inA_4(%logical_shim_noc_21, {%logical_mem_22}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_4(%logical_mem_22 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_7}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_4] -> [@memA_4]([] [0])
    aie.objectfifo @inA_5(%logical_shim_noc_23, {%logical_mem_24}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_5(%logical_mem_24 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_9}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_5] -> [@memA_5]([] [0])
    aie.objectfifo @inA_6(%logical_shim_noc_25, {%logical_mem_26}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_6(%logical_mem_26 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_11}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_6] -> [@memA_6]([] [0])
    aie.objectfifo @inA_7(%logical_shim_noc_27, {%logical_mem_28}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @memA_7(%logical_mem_28 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_13}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@inA_7] -> [@memA_7]([] [0])
    aie.objectfifo @inW1_0(%logical_shim_noc_29, {%logical_mem_30}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    // expected-error @below {{objectFIFO @memW1 in a cyclic dependency requires depth >= 8 for deadlock-free execution; allocated depth = 4}}
    aie.objectfifo @memW1_0(%logical_mem_30 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_core, %logical_core_1, %logical_core_3, %logical_core_5}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo.link [@inW1_0] -> [@memW1_0]([] [0])
    aie.objectfifo @inW1_1(%logical_shim_noc_31, {%logical_mem_32}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo @memW1_1(%logical_mem_32 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_core_7, %logical_core_9, %logical_core_11, %logical_core_13}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo.link [@inW1_1] -> [@memW1_1]([] [0])
    aie.objectfifo @inW2_0(%logical_shim_noc_33, {%logical_mem_34}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo @memW2_0(%logical_mem_34 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_core_0, %logical_core_2, %logical_core_4, %logical_core_6}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo.link [@inW2_0] -> [@memW2_0]([] [0])
    aie.objectfifo @inW2_1(%logical_shim_noc_35, {%logical_mem_36}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo @memW2_1(%logical_mem_36 dimensionsToStream [<size = 4, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%logical_core_8, %logical_core_10, %logical_core_12, %logical_core_14}, 4 : i32) : !aie.objectfifo<memref<32x32xbf16>>  
    aie.objectfifo.link [@inW2_1] -> [@memW2_1]([] [0])
    aie.objectfifo @memC_0(%logical_core_0, {%logical_mem_37}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_0(%logical_mem_37 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_53}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_0] -> [@outC_0]([] [0])
    aie.objectfifo @memC_1(%logical_core_2, {%logical_mem_38}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_1(%logical_mem_38 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_54}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_1] -> [@outC_1]([] [0])
    aie.objectfifo @memC_2(%logical_core_4, {%logical_mem_39}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_2(%logical_mem_39 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_55}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_2] -> [@outC_2]([] [0])
    aie.objectfifo @memC_3(%logical_core_6, {%logical_mem_40}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_3(%logical_mem_40 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_56}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_3] -> [@outC_3]([] [0])
    aie.objectfifo @memC_4(%logical_core_8, {%logical_mem_41}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_4(%logical_mem_41 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_57}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_4] -> [@outC_4]([] [0])
    aie.objectfifo @memC_5(%logical_core_10, {%logical_mem_42}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_5(%logical_mem_42 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_58}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_5] -> [@outC_5]([] [0])
    aie.objectfifo @memC_6(%logical_core_12, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_6(%logical_mem_43 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_59}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_6] -> [@outC_6]([] [0])
    aie.objectfifo @memC_7(%logical_core_14, {%logical_mem_44}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo @outC_7(%logical_mem_44 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_60}, 2 : i32) : !aie.objectfifo<memref<16x32xf32>>  
    aie.objectfifo.link [@memC_7] -> [@outC_7]([] [0])
    aie.objectfifo @memH_0(%logical_mem_45 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_0}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_0(%logical_core, {%logical_mem_45 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_0] -> [@memH_0]([] [0])
    aie.objectfifo @memH_1(%logical_mem_46 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_2}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_1(%logical_core_1, {%logical_mem_46 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_1] -> [@memH_1]([] [0])
    aie.objectfifo @memH_2(%logical_mem_47 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_4}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_2(%logical_core_3, {%logical_mem_47 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_2] -> [@memH_2]([] [0])
    aie.objectfifo @memH_3(%logical_mem_48 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_6}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_3(%logical_core_5, {%logical_mem_48 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_3] -> [@memH_3]([] [0])
    aie.objectfifo @memH_4(%logical_mem_49 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_8}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_4(%logical_core_7, {%logical_mem_49 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_4] -> [@memH_4]([] [0])
    aie.objectfifo @memH_5(%logical_mem_50 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_10}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_5(%logical_core_9, {%logical_mem_50 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_5] -> [@memH_5]([] [0])
    aie.objectfifo @memH_6(%logical_mem_51 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_12}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_6(%logical_core_11, {%logical_mem_51 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_6] -> [@memH_6]([] [0])
    aie.objectfifo @memH_7(%logical_mem_52 dimensionsToStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_14}, 8 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo @ofH_7(%logical_core_13, {%logical_mem_52 dimensionsFromStream [<size = 4, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<16x32xbf16>>  
    aie.objectfifo.link [@ofH_7] -> [@memH_7]([] [0])
    func.func private @zero_bf16(memref<16x32xbf16>) attributes {link_with = "mm_16x32x32.o"}
    func.func private @matmul_bf16_bf16(memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) attributes {link_with = "mm_16x32x32.o"}
    func.func private @zero_f32(memref<16x32xf32>) attributes {link_with = "mm_16x32x32.o"}
    func.func private @matmul_bf16_f32(memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) attributes {link_with = "mm_16x32x32.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_0(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
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
        %16 = aie.objectfifo.acquire @memC_0(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_0(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_0(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
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
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_1(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
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
        %16 = aie.objectfifo.acquire @memC_1(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_1(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_1(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
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
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_2(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
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
        %16 = aie.objectfifo.acquire @memC_2(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_2(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_2(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
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
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_3(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
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
        %16 = aie.objectfifo.acquire @memC_3(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_3(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_3(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_3(Consume, 1)
          aie.objectfifo.release @memW2_0(Consume, 1)
        }
        aie.objectfifo.release @memC_3(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %8 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_4(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_4(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_4(Consume, 1)
            aie.objectfifo.release @memW1_1(Consume, 1)
          }
          aie.objectfifo.release @ofH_4(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %9 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @memC_4(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_4(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_4(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_4(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_4(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
        }
        aie.objectfifo.release @memC_4(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %10 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_5(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_5(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_5(Consume, 1)
            aie.objectfifo.release @memW1_1(Consume, 1)
          }
          aie.objectfifo.release @ofH_5(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %11 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @memC_5(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_5(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_5(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_5(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_5(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
        }
        aie.objectfifo.release @memC_5(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %12 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_6(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_6(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_6(Consume, 1)
            aie.objectfifo.release @memW1_1(Consume, 1)
          }
          aie.objectfifo.release @ofH_6(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %13 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @memC_6(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_6(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_6(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_6(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_6(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
        }
        aie.objectfifo.release @memC_6(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    %14 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %16 = aie.objectfifo.acquire @ofH_7(Produce, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          func.call @zero_bf16(%17) : (memref<16x32xbf16>) -> ()
          %c0_63 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_64 = arith.constant 1 : index
          scf.for %arg2 = %c0_63 to %c4 step %c1_64 {
            %18 = aie.objectfifo.acquire @memA_7(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
            %19 = aie.objectfifo.subview.access %18[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
            %20 = aie.objectfifo.acquire @memW1_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
            %21 = aie.objectfifo.subview.access %20[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
            func.call @matmul_bf16_bf16(%19, %21, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xbf16>) -> ()
            aie.objectfifo.release @memA_7(Consume, 1)
            aie.objectfifo.release @memW1_1(Consume, 1)
          }
          aie.objectfifo.release @ofH_7(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %15 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %16 = aie.objectfifo.acquire @memC_7(Produce, 2) : !aie.objectfifosubview<memref<16x32xf32>>
        %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        %18 = aie.objectfifo.subview.access %16[1] : !aie.objectfifosubview<memref<16x32xf32>> -> memref<16x32xf32>
        func.call @zero_f32(%17) : (memref<16x32xf32>) -> ()
        func.call @zero_f32(%18) : (memref<16x32xf32>) -> ()
        %c0_61 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c1_62 = arith.constant 1 : index
        scf.for %arg1 = %c0_61 to %c8 step %c1_62 {
          %19 = aie.objectfifo.acquire @memH_7(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %21 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%20, %22, %17) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_7(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
          %23 = aie.objectfifo.acquire @memH_7(Consume, 1) : !aie.objectfifosubview<memref<16x32xbf16>>
          %24 = aie.objectfifo.subview.access %23[0] : !aie.objectfifosubview<memref<16x32xbf16>> -> memref<16x32xbf16>
          %25 = aie.objectfifo.acquire @memW2_1(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %26 = aie.objectfifo.subview.access %25[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          func.call @matmul_bf16_f32(%24, %26, %18) : (memref<16x32xbf16>, memref<32x32xbf16>, memref<16x32xf32>) -> ()
          aie.objectfifo.release @memH_7(Consume, 1)
          aie.objectfifo.release @memW2_1(Consume, 1)
        }
        aie.objectfifo.release @memC_7(Produce, 2)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: memref<32768xbf16>, %arg2: memref<16384xbf16>, %arg3: memref<8192xf32>) {
      %16 = aiex.dma_configure_task_for @outC_0 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 0, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @outC_0 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 32, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @outC_1 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 1024, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @outC_1 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 1056, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @outC_2 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 2048, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @outC_2 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 2080, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @outC_3 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 3072, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @outC_3 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 3104, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @outC_4 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 4096, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @outC_4 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 4128, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @outC_5 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 5120, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @outC_5 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 5152, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @outC_6 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 6144, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @outC_6 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 6176, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @outC_7 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 7168, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @outC_7 {
        aie.dma_bd(%arg3 : memref<8192xf32>, 7200, 512, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 16, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @inA_0 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 0, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @inA_1 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 2048, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @inA_2 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 4096, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @inA_3 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 6144, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @inA_4 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 8192, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @inA_5 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 10240, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @inA_6 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 12288, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @inA_7 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 14336, 2048, [<size = 8, stride = 0>, <size = 4, stride = 32>, <size = 16, stride = 128>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @inW1_0 {
        aie.dma_bd(%arg1 : memref<32768xbf16>, 0, 4096, [<size = 8, stride = 32>, <size = 4, stride = 8192>, <size = 32, stride = 256>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 8192, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 10240, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @inW2_0 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 14336, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @inW1_1 {
        aie.dma_bd(%arg1 : memref<32768xbf16>, 0, 4096, [<size = 8, stride = 32>, <size = 4, stride = 8192>, <size = 32, stride = 256>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 2048, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 4096, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 6144, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 8192, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 10240, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 12288, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @inW2_1 {
        aie.dma_bd(%arg2 : memref<16384xbf16>, 14336, 2048, [<size = 1, stride = 0>, <size = 2, stride = 32>, <size = 32, stride = 64>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%57)
      aiex.dma_await_task(%16)
      aiex.dma_await_task(%17)
      aiex.dma_await_task(%18)
      aiex.dma_await_task(%19)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%21)
      aiex.dma_await_task(%22)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%24)
      aiex.dma_await_task(%25)
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%27)
      aiex.dma_await_task(%28)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%30)
      aiex.dma_await_task(%31)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%42)
      aiex.dma_free_task(%43)
      aiex.dma_free_task(%44)
      aiex.dma_free_task(%45)
      aiex.dma_free_task(%46)
      aiex.dma_free_task(%47)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%52)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%55)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
    }
  }
}

