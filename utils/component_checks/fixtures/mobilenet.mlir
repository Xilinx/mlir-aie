// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// mobilenet (programming_examples/ml/mobilenet) before placement: 32 cores,
// 17 memtiles, 4 shims. From programming_examples/ml,
//   python -m mobilenet.aie2_mobilenet_iron --emit-mlir
// with every dense<...> weight replaced by dense<0>; placement reads the
// buffers' types, not their contents, and places it the same.

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
    %logical_core_15 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_16 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_17 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_18 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_19 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_20 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_21 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_22 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_23 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_24 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_25 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_26 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_27 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_28 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_29 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_30 = aie.logical_tile<CoreTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_31 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_32 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_33 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_34 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_35 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_36 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_37 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_38 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_39 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_40 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_41 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_42 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_43 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_44 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_45 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_46 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_47 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_48 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_49 = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @bn13_l1_get_wts(%logical_mem, {%logical_core_17}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn13_l1_put_wts(%logical_mem_31, {%logical_core_16}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn13_l3_get_wts(%logical_mem_32, {%logical_core_20}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn13_l3_put_wts(%logical_mem_33, {%logical_core_19}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn14_l1_get_wts(%logical_mem_34, {%logical_core_22}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn14_l1_put_wts(%logical_mem_35, {%logical_core_21}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn14_l3_get_wts(%logical_mem_36, {%logical_core_25}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @bn14_l3_put_wts(%logical_mem_37, {%logical_core_24}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<38400xi8>>  -> !aie.objectfifo<memref<19200xi8>> = [dense<0> : memref<38400xi8>]
    aie.objectfifo @of0(%logical_shim_noc, {%logical_core}, [1 : i32, 5 : i32]) : !aie.objectfifo<memref<224x1x8xi8>>
    aie.objectfifo @of1(%logical_core, {%logical_core_0}, [5 : i32, 3 : i32]) : !aie.objectfifo<memref<112x1x16xui8>>
    aie.objectfifo @of10(%logical_core_3, {%logical_core_4}, 2 : i32) : !aie.objectfifo<memref<28x1x40xi8>>
    aie.objectfifo @of11(%logical_core_3, {%logical_core_3}, 1 : i32) : !aie.objectfifo<memref<28x1x72xui8>>
    aie.objectfifo @of12(%logical_core_3, {%logical_core_4}, [2 : i32, 3 : i32]) : !aie.objectfifo<memref<28x1x120xui8>>
    aie.objectfifo @of13(%logical_core_3, {%logical_core_3}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<56x1x24xi8>>
    aie.objectfifo @of14(%logical_core_4, {%logical_core_5}, 2 : i32) : !aie.objectfifo<memref<28x1x40xi8>>
    aie.objectfifo @of15(%logical_core_4, {%logical_core_4}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<28x1x120xui8>>
    aie.objectfifo @of16(%logical_core_4, {%logical_core_4}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<28x1x40xi8>>
    aie.objectfifo @of17(%logical_core_4, {%logical_core_4}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<28x1x120xui8>>
    aie.objectfifo @of18(%logical_core_4, {%logical_core_4}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<28x1x120xui8>>
    aie.objectfifo @of19(%logical_core_5, {%logical_core_5}, 3 : i32) : !aie.objectfifo<memref<28x1x240xui8>>
    aie.objectfifo @of2(%logical_core_0, {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<112x1x16xi8>>
    aie.objectfifo @of20(%logical_core_5, {%logical_core_6}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @of21(%logical_core_5, {%logical_core_5}, 1 : i32) : !aie.objectfifo<memref<14x1x240xui8>>
    aie.objectfifo @of22(%logical_core_6, {%logical_core_6}, 3 : i32) : !aie.objectfifo<memref<14x1x200xui8>>
    aie.objectfifo @of23(%logical_core_6, {%logical_core_7}, 2 : i32) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @of24(%logical_core_6, {%logical_core_6}, 1 : i32) : !aie.objectfifo<memref<14x1x200xui8>>
    aie.objectfifo @of25(%logical_core_7, {%logical_core_8}, [1 : i32, 2 : i32]) : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @of26(%logical_core_7, {%logical_core_7}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<14x1x184xui8>>
    aie.objectfifo @of27(%logical_core_7, {%logical_core_7}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<14x1x184xui8>>
    aie.objectfifo @of28(%logical_core_7, {%logical_core_7}, 2 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<14x1x80xi8>>
    aie.objectfifo @of29(%logical_core_7, {%logical_core_7}, 3 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<14x1x184xui8>>
    aie.objectfifo @of3(%logical_core_0, {%logical_core_0}, 1 : i32) : !aie.objectfifo<memref<112x1x16xui8>>
    aie.objectfifo @of30(%logical_core_7, {%logical_core_7}, 1 : i32) {disable_synchronization = true} : !aie.objectfifo<memref<14x1x184xui8>>
    aie.objectfifo @of31(%logical_core_8, {%logical_core_9}, 4 : i32) : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @of32(%logical_core_9, {%logical_core_10}, 2 : i32) : !aie.objectfifo<memref<14x1x480xui8>>
    aie.objectfifo @of33(%logical_core_10, {%logical_mem_38, %logical_core_11}, [2 : i32, 6 : i32, 2 : i32]) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo @of33_fwd(%logical_mem_38, {%logical_core_13}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo.link [@of33] -> [@of33_fwd]([] [0])
    aie.objectfifo @of34(%logical_core_11, {%logical_core_12}, 4 : i32) : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @of35(%logical_core_12, {%logical_core_13}, 2 : i32) : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @of36(%logical_core_13, {%logical_core_14}, 2 : i32) : !aie.objectfifo<memref<14x1x112xi8>>
    aie.objectfifo @of37(%logical_core_14, {%logical_core_15}, 4 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<14x1x336xui8>>
    aie.objectfifo @of38(%logical_core_15, {%logical_core_15}, 1 : i32) : !aie.objectfifo<memref<7x1x336xui8>>
    aie.objectfifo @of39(%logical_core_15, {%logical_core_16, %logical_core_17, %logical_mem_39}, [2 : i32, 2 : i32, 2 : i32, 6 : i32]) : !aie.objectfifo<memref<7x1x80xi8>>
    aie.objectfifo @of39_fwd(%logical_mem_39, {%logical_core_20}, 2 : i32) : !aie.objectfifo<memref<7x1x80xi8>>
    aie.objectfifo.link [@of39] -> [@of39_fwd]([] [0])
    aie.objectfifo @of4(%logical_core_1, {%logical_core_1}, 3 : i32) : !aie.objectfifo<memref<112x1x64xui8>>
    aie.objectfifo @of40(%logical_core_17, {%logical_core_18}, 4 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<7x1x960xui8>>
    aie.objectfifo @of41(%logical_core_18, {%logical_core_19}, 2 : i32) : !aie.objectfifo<memref<7x1x480xui8>>
    aie.objectfifo @of42(%logical_core_18, {%logical_core_20}, 2 : i32) : !aie.objectfifo<memref<7x1x480xui8>>
    aie.objectfifo @of43(%logical_core_20, {%logical_core_21, %logical_core_22, %logical_mem_40}, [2 : i32, 2 : i32, 2 : i32, 6 : i32]) : !aie.objectfifo<memref<7x1x80xi8>>
    aie.objectfifo @of43_fwd(%logical_mem_40, {%logical_core_25}, 2 : i32) : !aie.objectfifo<memref<7x1x80xi8>>
    aie.objectfifo.link [@of43] -> [@of43_fwd]([] [0])
    aie.objectfifo @of44(%logical_core_22, {%logical_core_23}, 4 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<7x1x960xui8>>
    aie.objectfifo @of45(%logical_core_23, {%logical_core_24}, 2 : i32) : !aie.objectfifo<memref<7x1x480xui8>>
    aie.objectfifo @of46(%logical_core_23, {%logical_core_25}, 2 : i32) : !aie.objectfifo<memref<7x1x480xui8>>
    aie.objectfifo @of47(%logical_core_25, {%logical_core_26}, 2 : i32) : !aie.objectfifo<memref<7x1x80xi8>>
    aie.objectfifo @of48(%logical_core_26, {%logical_shim_noc_41}, 2 : i32) : !aie.objectfifo<memref<1280xui16>>
    aie.objectfifo @of49(%logical_shim_noc_42, {%logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<1280xui16>>
    aie.objectfifo @of5(%logical_core_1, {%logical_core_2}, 4 : i32) : !aie.objectfifo<memref<56x1x64xui8>>
    aie.objectfifo @of50(%logical_mem_43, {%logical_shim_noc_44}, 2 : i32) : !aie.objectfifo<memref<1280xui16>>
    aie.objectfifo @of50_join0(%logical_core_27, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<8xui16>>
    aie.objectfifo @of50_join1(%logical_core_28, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<8xui16>>
    aie.objectfifo @of50_join2(%logical_core_29, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<8xui16>>
    aie.objectfifo @of50_join3(%logical_core_30, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<8xui16>>
    aie.objectfifo.link [@of50_join0, @of50_join1, @of50_join2, @of50_join3] -> [@of50]([0, 320, 640, 960] [])
    aie.objectfifo @of6(%logical_core_2, {%logical_core_2}, 3 : i32) : !aie.objectfifo<memref<56x1x72xui8>>
    aie.objectfifo @of7(%logical_core_2, {%logical_core_3}, 4 : i32) : !aie.objectfifo<memref<56x1x72xui8>>
    aie.objectfifo @of8(%logical_core_2, {%logical_core_3}, 4 : i32) : !aie.objectfifo<memref<56x1x24xi8>>
    aie.objectfifo @of9(%logical_core_3, {%logical_core_3}, 3 : i32) : !aie.objectfifo<memref<56x1x72xui8>>
    aie.objectfifo @post_L1_wts(%logical_mem_45, {%logical_core_26}, [1 : i32, 2 : i32]) {repeat_count = 7 : i32} : !aie.objectfifo<memref<76800xi8>>  -> !aie.objectfifo<memref<9600xi8>> = [dense<0> : memref<76800xi8>]
    aie.objectfifo @post_L2_wts_1(%logical_mem_46, {%logical_core_27}, 2 : i32) : !aie.objectfifo<memref<409600xi8>>  -> !aie.objectfifo<memref<10240xi8>> = [dense<0> : memref<409600xi8>, dense<0> : memref<409600xi8>]
    aie.objectfifo @post_L2_wts_2(%logical_mem_47, {%logical_core_28}, 2 : i32) : !aie.objectfifo<memref<409600xi8>>  -> !aie.objectfifo<memref<10240xi8>> = [dense<0> : memref<409600xi8>, dense<0> : memref<409600xi8>]
    aie.objectfifo @post_L2_wts_3(%logical_mem_48, {%logical_core_29}, 2 : i32) : !aie.objectfifo<memref<409600xi8>>  -> !aie.objectfifo<memref<10240xi8>> = [dense<0> : memref<409600xi8>, dense<0> : memref<409600xi8>]
    aie.objectfifo @post_L2_wts_4(%logical_mem_49, {%logical_core_30}, 2 : i32) : !aie.objectfifo<memref<409600xi8>>  -> !aie.objectfifo<memref<10240xi8>> = [dense<0> : memref<409600xi8>, dense<0> : memref<409600xi8>]
    %buf_0 = aie.buffer(%logical_core) {sym_name = "buf_0"} : memref<1152xi8> = dense<0>
    func.func private @"1b12b3d8_conv2dk3_stride2_i8"(memref<1792xi8>, memref<1792xi8>, memref<1792xi8>, memref<1152xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_stride2_i8_1b12b3d8.o"}
    %buf_1 = aie.buffer(%logical_core_0) {sym_name = "buf_1"} : memref<448xi8> = dense<0>
    func.func private @"77adb2a9_conv2dk3_dw_stride1_relu_ui8_ui8"(memref<1792xui8>, memref<1792xui8>, memref<1792xui8>, memref<144xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_77adb2a9.o"}
    func.func private @"36e6bbc5_conv2dk1_skip_ui8_ui8_i8"(memref<1792xui8>, memref<256xi8>, memref<1792xi8>, memref<1792xui8>, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_skip_ui8_ui8_i8_36e6bbc5.o"}
    %buf_2 = aie.buffer(%logical_core_1) {sym_name = "buf_2"} : memref<1600xi8> = dense<0>
    func.func private @"0d52fac7_conv2dk1_relu_i8_ui8"(memref<1792xi8>, memref<1024xi8>, memref<7168xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_0d52fac7.o"}
    func.func private @"167332cb_conv2dk3_dw_stride2_relu_ui8_ui8"(memref<7168xui8>, memref<7168xui8>, memref<7168xui8>, memref<576xi8>, memref<3584xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride2_relu_ui8_ui8_167332cb.o"}
    %buf_3 = aie.buffer(%logical_core_2) {sym_name = "buf_3"} : memref<3968xi8> = dense<0>
    func.func private @b33ddc7d_conv2dk1_relu_i8_ui8(memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_b33ddc7d.o"}
    func.func private @a63dee7f_conv2dk3_dw_stride1_relu_ui8_ui8(memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<4032xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_a63dee7f.o"}
    func.func private @f7ee2b1f_conv2dk1_ui8_i8(memref<3584xui8>, memref<1536xi8>, memref<1344xi8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_ui8_i8_f7ee2b1f.o"}
    %buf_4 = aie.buffer(%logical_core_3) {sym_name = "buf_4"} : memref<11840xi8> = dense<0>
    func.func private @c374361d_conv2dk3_dw_stride2_relu_ui8_ui8(memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<2016xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride2_relu_ui8_ui8_c374361d.o"}
    func.func private @aff12032_conv2dk1_ui8_i8(memref<2016xui8>, memref<2880xi8>, memref<1120xi8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_ui8_i8_aff12032.o"}
    func.func private @"453c6c71_conv2dk1_relu_i8_ui8"(memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_453c6c71.o"}
    func.func private @"8d516248_conv2dk1_skip_ui8_i8_i8"(memref<4032xui8>, memref<1728xi8>, memref<1344xi8>, memref<1344xi8>, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_skip_ui8_i8_i8_8d516248.o"}
    %buf_5 = aie.buffer(%logical_core_4) {sym_name = "buf_5"} : memref<16576xi8> = dense<0>
    func.func private @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_3e7b1aaa.o"}
    func.func private @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_skip_ui8_i8_i8_6d1fcc7d.o"}
    %buf_6 = aie.buffer(%logical_core_5) {sym_name = "buf_6"} : memref<30976xi8> = dense<0>
    func.func private @b8f588ff_conv2dk1_relu_i8_ui8(memref<1120xi8>, memref<9600xi8>, memref<6720xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_b8f588ff.o"}
    func.func private @"5a8aa48c_conv2dk3_dw_stride2_relu_ui8_ui8"(memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<2160xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride2_relu_ui8_ui8_5a8aa48c.o"}
    func.func private @a7cf1c86_conv2dk1_ui8_i8(memref<3360xui8>, memref<19200xi8>, memref<1120xi8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_ui8_i8_a7cf1c86.o"}
    %buf_7 = aie.buffer(%logical_core_6) {sym_name = "buf_7"} : memref<33856xi8> = dense<0>
    func.func private @ee5a41d0_conv2dk1_relu_i8_ui8(memref<1120xi8>, memref<16000xi8>, memref<2800xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_ee5a41d0.o"}
    func.func private @"4cb11a8b_conv2dk3_dw_stride1_relu_ui8_ui8"(memref<2800xui8>, memref<2800xui8>, memref<2800xui8>, memref<1800xi8>, memref<2800xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_4cb11a8b.o"}
    func.func private @f245367d_conv2dk1_skip_ui8_i8_i8(memref<2800xui8>, memref<16000xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_skip_ui8_i8_i8_f245367d.o"}
    %buf_8 = aie.buffer(%logical_core_7) {sym_name = "buf_8"} : memref<62208xi8> = dense<0>
    func.func private @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_a74d4d9a.o"}
    func.func private @a7eba513_conv2dk1_skip_ui8_i8_i8(memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_skip_ui8_i8_i8_a7eba513.o"}
    func.func private @"14a4ba36_conv2dk1_relu_i8_ui8"(memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_14a4ba36.o"}
    %buf_9 = aie.buffer(%logical_core_8) {sym_name = "buf_9"} : memref<38400xi8> = dense<0>
    func.func private @"62fc3ee5_conv2dk1_relu_i8_ui8"(memref<1120xi8>, memref<38400xi8>, memref<6720xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_62fc3ee5.o"}
    %buf_10 = aie.buffer(%logical_core_9) {sym_name = "buf_10"} : memref<4320xi8> = dense<0>
    func.func private @"61f9b950_conv2dk3_dw_stride1_relu_ui8_ui8"(memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<4320xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_61f9b950.o"}
    %buf_11 = aie.buffer(%logical_core_10) {sym_name = "buf_11"} : memref<53760xi8> = dense<0>
    func.func private @c261f057_conv2dk1_ui8_i8(memref<6720xui8>, memref<53760xi8>, memref<1568xi8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_ui8_i8_c261f057.o"}
    %buf_12 = aie.buffer(%logical_core_11) {sym_name = "buf_12"} : memref<37632xi8> = dense<0>
    func.func private @"9c08375a_conv2dk1_relu_i8_ui8"(memref<1568xi8>, memref<37632xi8>, memref<4704xui8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_relu_i8_ui8_9c08375a.o"}
    %buf_13 = aie.buffer(%logical_core_12) {sym_name = "buf_13"} : memref<3024xi8> = dense<0>
    func.func private @"1461e5eb_conv2dk3_dw_stride1_relu_ui8_ui8"(memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<4704xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride1_relu_ui8_ui8_1461e5eb.o"}
    %buf_14 = aie.buffer(%logical_core_13) {sym_name = "buf_14"} : memref<37632xi8> = dense<0>
    func.func private @"0de09bc1_conv2dk1_skip_ui8_i8_i8"(memref<4704xui8>, memref<37632xi8>, memref<1568xi8>, memref<1568xi8>, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_skip_ui8_i8_i8_0de09bc1.o"}
    %buf_15 = aie.buffer(%logical_core_14) {sym_name = "buf_15"} : memref<37632xi8> = dense<0>
    %buf_16 = aie.buffer(%logical_core_15) {sym_name = "buf_16"} : memref<29952xi8> = dense<0>
    func.func private @"2aac7224_conv2dk3_dw_stride2_relu_ui8_ui8"(memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<2352xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk3_dw_stride2_relu_ui8_ui8_2aac7224.o"}
    func.func private @"6a5d1c89_conv2dk1_ui8_i8"(memref<2352xui8>, memref<26880xi8>, memref<560xi8>, i32, i32, i32, i32) attributes {link_with = "conv2dk1_ui8_i8_6a5d1c89.o"}
    func.func private @e10f2832_bn13_1_conv2dk1_i8_ui8_partial_width_put_new(memref<560xi8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn13_1_conv2dk1_i8_ui8_partial_width_put_new_e10f2832.o"}
    func.func private @ef8b7952_bn13_1_conv2dk1_i8_ui8_partial_width_get_new(memref<560xi8>, memref<19200xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn13_1_conv2dk1_i8_ui8_partial_width_get_new_ef8b7952.o"}
    %bn13_2_wts_static = aie.buffer(%logical_core_18) {sym_name = "bn13_2_wts_static"} : memref<8640xi8> = dense<0>
    func.func private @db3aae29_bn13_conv2dk3_ui8_out_split(memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn13_conv2dk3_ui8_out_split_db3aae29.o"}
    func.func private @"612b8218_bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new"(memref<3360xui8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new_612b8218.o"}
    func.func private @f8d47821_bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(memref<3360xui8>, memref<19200xi8>, memref<560xi8>, memref<560xi8>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new_f8d47821.o"}
    func.func private @d244edb6_bn14_1_conv2dk1_i8_ui8_partial_width_put_new(memref<560xi8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn14_1_conv2dk1_i8_ui8_partial_width_put_new_d244edb6.o"}
    func.func private @"7a24c72f_bn14_1_conv2dk1_i8_ui8_partial_width_get_new"(memref<560xi8>, memref<19200xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn14_1_conv2dk1_i8_ui8_partial_width_get_new_7a24c72f.o"}
    %bn14_2_wts_static = aie.buffer(%logical_core_23) {sym_name = "bn14_2_wts_static"} : memref<8640xi8> = dense<0>
    func.func private @a1fb6ef1_bn14_conv2dk3_ui8_out_split(memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn14_conv2dk3_ui8_out_split_a1fb6ef1.o"}
    func.func private @aa85880e_bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new(memref<3360xui8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new_aa85880e.o"}
    func.func private @"9336eff4_bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new"(memref<3360xui8>, memref<19200xi8>, memref<560xi8>, memref<560xi8>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new_9336eff4.o"}
    func.func private @"613bef8c_conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8"(memref<560xi8>, memref<9600xi8>, memref<1280xui16>, i32, i32, i32, i32, i32, i32, i32, i32) attributes {link_with = "conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8_613bef8c.o"}
    func.func private @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) attributes {link_with = "post_L2_conv2dk1_relu_i16_ui16_pad_c7b77a1e.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32:2 = aie.objectfifo.acquire @of0(Consume, 2) : memref<224x1x8xi8>, memref<224x1x8xi8>
        %33 = aie.objectfifo.acquire @of1(Produce, 1) : memref<112x1x16xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<224x1x8xi8> into memref<1792xi8>
        %collapse_shape_50 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<224x1x8xi8> into memref<1792xi8>
        %collapse_shape_51 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<224x1x8xi8> into memref<1792xi8>
        %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %c224_i32 = arith.constant 224 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_53 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32_54 = arith.constant 8 : i32
        %c0_i32_55 = arith.constant 0 : i32
        func.call @"1b12b3d8_conv2dk3_stride2_i8"(%collapse_shape, %collapse_shape_50, %collapse_shape_51, %buf_0, %collapse_shape_52, %c224_i32, %c8_i32, %c16_i32, %c3_i32, %c3_i32_53, %c0_i32, %c8_i32_54, %c0_i32_55) : (memref<1792xi8>, memref<1792xi8>, memref<1792xi8>, memref<1152xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of1(Produce, 1)
        aie.objectfifo.release @of0(Consume, 1)
        %c0_56 = arith.constant 0 : index
        %c111 = arith.constant 111 : index
        %c1_57 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c111 step %c1_57 {
          %34:3 = aie.objectfifo.acquire @of0(Consume, 3) : memref<224x1x8xi8>, memref<224x1x8xi8>, memref<224x1x8xi8>
          %35 = aie.objectfifo.acquire @of1(Produce, 1) : memref<112x1x16xui8>
          %collapse_shape_58 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<224x1x8xi8> into memref<1792xi8>
          %collapse_shape_59 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<224x1x8xi8> into memref<1792xi8>
          %collapse_shape_60 = memref.collapse_shape %34#2 [[0, 1, 2]] : memref<224x1x8xi8> into memref<1792xi8>
          %collapse_shape_61 = memref.collapse_shape %35 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %c224_i32_62 = arith.constant 224 : i32
          %c8_i32_63 = arith.constant 8 : i32
          %c16_i32_64 = arith.constant 16 : i32
          %c3_i32_65 = arith.constant 3 : i32
          %c3_i32_66 = arith.constant 3 : i32
          %c1_i32 = arith.constant 1 : i32
          %c8_i32_67 = arith.constant 8 : i32
          %c0_i32_68 = arith.constant 0 : i32
          func.call @"1b12b3d8_conv2dk3_stride2_i8"(%collapse_shape_58, %collapse_shape_59, %collapse_shape_60, %buf_0, %collapse_shape_61, %c224_i32_62, %c8_i32_63, %c16_i32_64, %c3_i32_65, %c3_i32_66, %c1_i32, %c8_i32_67, %c0_i32_68) : (memref<1792xi8>, memref<1792xi8>, memref<1792xi8>, memref<1152xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of0(Consume, 2)
          aie.objectfifo.release @of1(Produce, 1)
        }
        aie.objectfifo.release @of0(Consume, 1)
      }
      aie.end
    }
    %1 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_1[%c0_50][] : memref<448xi8> to memref<144xi8>
        %c192 = arith.constant 192 : index
        %view_51 = memref.view %buf_1[%c192][] : memref<448xi8> to memref<256xi8>
        %32:2 = aie.objectfifo.acquire @of1(Consume, 2) : memref<112x1x16xui8>, memref<112x1x16xui8>
        %33 = aie.objectfifo.acquire @of3(Produce, 1) : memref<112x1x16xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_52 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_53 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %c112_i32 = arith.constant 112 : i32
        %c1_i32 = arith.constant 1 : i32
        %c16_i32 = arith.constant 16 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_55 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c9_i32 = arith.constant 9 : i32
        %c0_i32_56 = arith.constant 0 : i32
        func.call @"77adb2a9_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape, %collapse_shape_52, %collapse_shape_53, %view, %collapse_shape_54, %c112_i32, %c1_i32, %c16_i32, %c3_i32, %c3_i32_55, %c0_i32, %c9_i32, %c0_i32_56) : (memref<1792xui8>, memref<1792xui8>, memref<1792xui8>, memref<144xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of3(Produce, 1)
        %34 = aie.objectfifo.acquire @of3(Consume, 1) : memref<112x1x16xui8>
        %35 = aie.objectfifo.acquire @of2(Produce, 1) : memref<112x1x16xi8>
        %collapse_shape_57 = memref.collapse_shape %34 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_58 = memref.collapse_shape %35 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
        %collapse_shape_59 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %c112_i32_60 = arith.constant 112 : i32
        %c16_i32_61 = arith.constant 16 : i32
        %c16_i32_62 = arith.constant 16 : i32
        %c8_i32 = arith.constant 8 : i32
        %c2_i32 = arith.constant 2 : i32
        func.call @"36e6bbc5_conv2dk1_skip_ui8_ui8_i8"(%collapse_shape_57, %view_51, %collapse_shape_58, %collapse_shape_59, %c112_i32_60, %c16_i32_61, %c16_i32_62, %c8_i32, %c2_i32) : (memref<1792xui8>, memref<256xi8>, memref<1792xi8>, memref<1792xui8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of3(Consume, 1)
        aie.objectfifo.release @of2(Produce, 1)
        %c0_63 = arith.constant 0 : index
        %c110 = arith.constant 110 : index
        %c1_64 = arith.constant 1 : index
        scf.for %arg1 = %c0_63 to %c110 step %c1_64 {
          %40:3 = aie.objectfifo.acquire @of1(Consume, 3) : memref<112x1x16xui8>, memref<112x1x16xui8>, memref<112x1x16xui8>
          %41 = aie.objectfifo.acquire @of3(Produce, 1) : memref<112x1x16xui8>
          %collapse_shape_85 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %collapse_shape_86 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %collapse_shape_87 = memref.collapse_shape %40#2 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %collapse_shape_88 = memref.collapse_shape %41 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %c112_i32_89 = arith.constant 112 : i32
          %c1_i32_90 = arith.constant 1 : i32
          %c16_i32_91 = arith.constant 16 : i32
          %c3_i32_92 = arith.constant 3 : i32
          %c3_i32_93 = arith.constant 3 : i32
          %c1_i32_94 = arith.constant 1 : i32
          %c9_i32_95 = arith.constant 9 : i32
          %c0_i32_96 = arith.constant 0 : i32
          func.call @"77adb2a9_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_85, %collapse_shape_86, %collapse_shape_87, %view, %collapse_shape_88, %c112_i32_89, %c1_i32_90, %c16_i32_91, %c3_i32_92, %c3_i32_93, %c1_i32_94, %c9_i32_95, %c0_i32_96) : (memref<1792xui8>, memref<1792xui8>, memref<1792xui8>, memref<144xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of3(Produce, 1)
          %42 = aie.objectfifo.acquire @of3(Consume, 1) : memref<112x1x16xui8>
          %43 = aie.objectfifo.acquire @of2(Produce, 1) : memref<112x1x16xi8>
          %collapse_shape_97 = memref.collapse_shape %42 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %collapse_shape_98 = memref.collapse_shape %43 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
          %collapse_shape_99 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
          %c112_i32_100 = arith.constant 112 : i32
          %c16_i32_101 = arith.constant 16 : i32
          %c16_i32_102 = arith.constant 16 : i32
          %c8_i32_103 = arith.constant 8 : i32
          %c2_i32_104 = arith.constant 2 : i32
          func.call @"36e6bbc5_conv2dk1_skip_ui8_ui8_i8"(%collapse_shape_97, %view_51, %collapse_shape_98, %collapse_shape_99, %c112_i32_100, %c16_i32_101, %c16_i32_102, %c8_i32_103, %c2_i32_104) : (memref<1792xui8>, memref<256xi8>, memref<1792xi8>, memref<1792xui8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of3(Consume, 1)
          aie.objectfifo.release @of2(Produce, 1)
          aie.objectfifo.release @of1(Consume, 1)
        }
        %36:2 = aie.objectfifo.acquire @of1(Consume, 2) : memref<112x1x16xui8>, memref<112x1x16xui8>
        %37 = aie.objectfifo.acquire @of3(Produce, 1) : memref<112x1x16xui8>
        %collapse_shape_65 = memref.collapse_shape %36#0 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_66 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_67 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_68 = memref.collapse_shape %37 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %c112_i32_69 = arith.constant 112 : i32
        %c1_i32_70 = arith.constant 1 : i32
        %c16_i32_71 = arith.constant 16 : i32
        %c3_i32_72 = arith.constant 3 : i32
        %c3_i32_73 = arith.constant 3 : i32
        %c2_i32_74 = arith.constant 2 : i32
        %c9_i32_75 = arith.constant 9 : i32
        %c0_i32_76 = arith.constant 0 : i32
        func.call @"77adb2a9_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_65, %collapse_shape_66, %collapse_shape_67, %view, %collapse_shape_68, %c112_i32_69, %c1_i32_70, %c16_i32_71, %c3_i32_72, %c3_i32_73, %c2_i32_74, %c9_i32_75, %c0_i32_76) : (memref<1792xui8>, memref<1792xui8>, memref<1792xui8>, memref<144xi8>, memref<1792xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of3(Produce, 1)
        %38 = aie.objectfifo.acquire @of3(Consume, 1) : memref<112x1x16xui8>
        %39 = aie.objectfifo.acquire @of2(Produce, 1) : memref<112x1x16xi8>
        %collapse_shape_77 = memref.collapse_shape %38 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %collapse_shape_78 = memref.collapse_shape %39 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
        %collapse_shape_79 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<112x1x16xui8> into memref<1792xui8>
        %c112_i32_80 = arith.constant 112 : i32
        %c16_i32_81 = arith.constant 16 : i32
        %c16_i32_82 = arith.constant 16 : i32
        %c8_i32_83 = arith.constant 8 : i32
        %c2_i32_84 = arith.constant 2 : i32
        func.call @"36e6bbc5_conv2dk1_skip_ui8_ui8_i8"(%collapse_shape_77, %view_51, %collapse_shape_78, %collapse_shape_79, %c112_i32_80, %c16_i32_81, %c16_i32_82, %c8_i32_83, %c2_i32_84) : (memref<1792xui8>, memref<256xi8>, memref<1792xi8>, memref<1792xui8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of3(Consume, 1)
        aie.objectfifo.release @of2(Produce, 1)
        aie.objectfifo.release @of1(Consume, 2)
      }
      aie.end
    }
    %2 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_2[%c0_50][] : memref<1600xi8> to memref<1024xi8>
        %c1024 = arith.constant 1024 : index
        %view_51 = memref.view %buf_2[%c1024][] : memref<1600xi8> to memref<576xi8>
        %32:2 = aie.objectfifo.acquire @of2(Consume, 2) : memref<112x1x16xi8>, memref<112x1x16xi8>
        %33:2 = aie.objectfifo.acquire @of4(Produce, 2) : memref<112x1x64xui8>, memref<112x1x64xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
        %collapse_shape_52 = memref.collapse_shape %33#0 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
        %c112_i32 = arith.constant 112 : i32
        %c16_i32 = arith.constant 16 : i32
        %c64_i32 = arith.constant 64 : i32
        %c8_i32 = arith.constant 8 : i32
        func.call @"0d52fac7_conv2dk1_relu_i8_ui8"(%collapse_shape, %view, %collapse_shape_52, %c112_i32, %c16_i32, %c64_i32, %c8_i32) : (memref<1792xi8>, memref<1024xi8>, memref<7168xui8>, i32, i32, i32, i32) -> ()
        %collapse_shape_53 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
        %collapse_shape_54 = memref.collapse_shape %33#1 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
        %c112_i32_55 = arith.constant 112 : i32
        %c16_i32_56 = arith.constant 16 : i32
        %c64_i32_57 = arith.constant 64 : i32
        %c8_i32_58 = arith.constant 8 : i32
        func.call @"0d52fac7_conv2dk1_relu_i8_ui8"(%collapse_shape_53, %view, %collapse_shape_54, %c112_i32_55, %c16_i32_56, %c64_i32_57, %c8_i32_58) : (memref<1792xi8>, memref<1024xi8>, memref<7168xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of4(Produce, 2)
        aie.objectfifo.release @of2(Consume, 2)
        %34:2 = aie.objectfifo.acquire @of4(Consume, 2) : memref<112x1x64xui8>, memref<112x1x64xui8>
        %35 = aie.objectfifo.acquire @of5(Produce, 1) : memref<56x1x64xui8>
        %collapse_shape_59 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
        %collapse_shape_60 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
        %collapse_shape_61 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
        %collapse_shape_62 = memref.collapse_shape %35 [[0, 1, 2]] : memref<56x1x64xui8> into memref<3584xui8>
        %c112_i32_63 = arith.constant 112 : i32
        %c1_i32 = arith.constant 1 : i32
        %c64_i32_64 = arith.constant 64 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_65 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_66 = arith.constant 0 : i32
        func.call @"167332cb_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape_59, %collapse_shape_60, %collapse_shape_61, %view_51, %collapse_shape_62, %c112_i32_63, %c1_i32, %c64_i32_64, %c3_i32, %c3_i32_65, %c0_i32, %c7_i32, %c0_i32_66) : (memref<7168xui8>, memref<7168xui8>, memref<7168xui8>, memref<576xi8>, memref<3584xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of4(Consume, 1)
        aie.objectfifo.release @of5(Produce, 1)
        %c0_67 = arith.constant 0 : index
        %c55 = arith.constant 55 : index
        %c1_68 = arith.constant 1 : index
        scf.for %arg1 = %c0_67 to %c55 step %c1_68 {
          %36:2 = aie.objectfifo.acquire @of2(Consume, 2) : memref<112x1x16xi8>, memref<112x1x16xi8>
          %37:2 = aie.objectfifo.acquire @of4(Produce, 2) : memref<112x1x64xui8>, memref<112x1x64xui8>
          %collapse_shape_69 = memref.collapse_shape %36#0 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
          %collapse_shape_70 = memref.collapse_shape %37#0 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
          %c112_i32_71 = arith.constant 112 : i32
          %c16_i32_72 = arith.constant 16 : i32
          %c64_i32_73 = arith.constant 64 : i32
          %c8_i32_74 = arith.constant 8 : i32
          func.call @"0d52fac7_conv2dk1_relu_i8_ui8"(%collapse_shape_69, %view, %collapse_shape_70, %c112_i32_71, %c16_i32_72, %c64_i32_73, %c8_i32_74) : (memref<1792xi8>, memref<1024xi8>, memref<7168xui8>, i32, i32, i32, i32) -> ()
          %collapse_shape_75 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<112x1x16xi8> into memref<1792xi8>
          %collapse_shape_76 = memref.collapse_shape %37#1 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
          %c112_i32_77 = arith.constant 112 : i32
          %c16_i32_78 = arith.constant 16 : i32
          %c64_i32_79 = arith.constant 64 : i32
          %c8_i32_80 = arith.constant 8 : i32
          func.call @"0d52fac7_conv2dk1_relu_i8_ui8"(%collapse_shape_75, %view, %collapse_shape_76, %c112_i32_77, %c16_i32_78, %c64_i32_79, %c8_i32_80) : (memref<1792xi8>, memref<1024xi8>, memref<7168xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of4(Produce, 2)
          aie.objectfifo.release @of2(Consume, 2)
          %38:3 = aie.objectfifo.acquire @of4(Consume, 3) : memref<112x1x64xui8>, memref<112x1x64xui8>, memref<112x1x64xui8>
          %39 = aie.objectfifo.acquire @of5(Produce, 1) : memref<56x1x64xui8>
          %collapse_shape_81 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
          %collapse_shape_82 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
          %collapse_shape_83 = memref.collapse_shape %38#2 [[0, 1, 2]] : memref<112x1x64xui8> into memref<7168xui8>
          %collapse_shape_84 = memref.collapse_shape %39 [[0, 1, 2]] : memref<56x1x64xui8> into memref<3584xui8>
          %c112_i32_85 = arith.constant 112 : i32
          %c1_i32_86 = arith.constant 1 : i32
          %c64_i32_87 = arith.constant 64 : i32
          %c3_i32_88 = arith.constant 3 : i32
          %c3_i32_89 = arith.constant 3 : i32
          %c1_i32_90 = arith.constant 1 : i32
          %c7_i32_91 = arith.constant 7 : i32
          %c0_i32_92 = arith.constant 0 : i32
          func.call @"167332cb_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape_81, %collapse_shape_82, %collapse_shape_83, %view_51, %collapse_shape_84, %c112_i32_85, %c1_i32_86, %c64_i32_87, %c3_i32_88, %c3_i32_89, %c1_i32_90, %c7_i32_91, %c0_i32_92) : (memref<7168xui8>, memref<7168xui8>, memref<7168xui8>, memref<576xi8>, memref<3584xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of4(Consume, 2)
          aie.objectfifo.release @of5(Produce, 1)
        }
        aie.objectfifo.release @of4(Consume, 1)
      }
      aie.end
    }
    %3 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_3[%c0_50][] : memref<3968xi8> to memref<1728xi8>
        %c1728 = arith.constant 1728 : index
        %view_51 = memref.view %buf_3[%c1728][] : memref<3968xi8> to memref<648xi8>
        %c2432 = arith.constant 2432 : index
        %view_52 = memref.view %buf_3[%c2432][] : memref<3968xi8> to memref<1536xi8>
        %32 = aie.objectfifo.acquire @of5(Consume, 1) : memref<56x1x64xui8>
        %33 = aie.objectfifo.acquire @of8(Produce, 1) : memref<56x1x24xi8>
        %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<56x1x64xui8> into memref<3584xui8>
        %collapse_shape_53 = memref.collapse_shape %33 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %c56_i32 = arith.constant 56 : i32
        %c64_i32 = arith.constant 64 : i32
        %c24_i32 = arith.constant 24 : i32
        %c7_i32 = arith.constant 7 : i32
        func.call @f7ee2b1f_conv2dk1_ui8_i8(%collapse_shape, %view_52, %collapse_shape_53, %c56_i32, %c64_i32, %c24_i32, %c7_i32) : (memref<3584xui8>, memref<1536xi8>, memref<1344xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of5(Consume, 1)
        %34 = aie.objectfifo.acquire @of6(Produce, 1) : memref<56x1x72xui8>
        %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %collapse_shape_55 = memref.collapse_shape %34 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %c56_i32_56 = arith.constant 56 : i32
        %c24_i32_57 = arith.constant 24 : i32
        %c72_i32 = arith.constant 72 : i32
        %c8_i32 = arith.constant 8 : i32
        func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_54, %view, %collapse_shape_55, %c56_i32_56, %c24_i32_57, %c72_i32, %c8_i32) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of6(Produce, 1)
        aie.objectfifo.release @of8(Produce, 1)
        %35 = aie.objectfifo.acquire @of5(Consume, 1) : memref<56x1x64xui8>
        %36 = aie.objectfifo.acquire @of8(Produce, 1) : memref<56x1x24xi8>
        %collapse_shape_58 = memref.collapse_shape %35 [[0, 1, 2]] : memref<56x1x64xui8> into memref<3584xui8>
        %collapse_shape_59 = memref.collapse_shape %36 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %c56_i32_60 = arith.constant 56 : i32
        %c64_i32_61 = arith.constant 64 : i32
        %c24_i32_62 = arith.constant 24 : i32
        %c7_i32_63 = arith.constant 7 : i32
        func.call @f7ee2b1f_conv2dk1_ui8_i8(%collapse_shape_58, %view_52, %collapse_shape_59, %c56_i32_60, %c64_i32_61, %c24_i32_62, %c7_i32_63) : (memref<3584xui8>, memref<1536xi8>, memref<1344xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of5(Consume, 1)
        %37 = aie.objectfifo.acquire @of6(Produce, 1) : memref<56x1x72xui8>
        %collapse_shape_64 = memref.collapse_shape %36 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %collapse_shape_65 = memref.collapse_shape %37 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %c56_i32_66 = arith.constant 56 : i32
        %c24_i32_67 = arith.constant 24 : i32
        %c72_i32_68 = arith.constant 72 : i32
        %c8_i32_69 = arith.constant 8 : i32
        func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_64, %view, %collapse_shape_65, %c56_i32_66, %c24_i32_67, %c72_i32_68, %c8_i32_69) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of6(Produce, 1)
        aie.objectfifo.release @of8(Produce, 1)
        %38:2 = aie.objectfifo.acquire @of6(Consume, 2) : memref<56x1x72xui8>, memref<56x1x72xui8>
        %39 = aie.objectfifo.acquire @of7(Produce, 1) : memref<56x1x72xui8>
        %collapse_shape_70 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_71 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_72 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_73 = memref.collapse_shape %39 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %c56_i32_74 = arith.constant 56 : i32
        %c1_i32 = arith.constant 1 : i32
        %c72_i32_75 = arith.constant 72 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_76 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32_77 = arith.constant 8 : i32
        %c0_i32_78 = arith.constant 0 : i32
        func.call @a63dee7f_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_70, %collapse_shape_71, %collapse_shape_72, %view_51, %collapse_shape_73, %c56_i32_74, %c1_i32, %c72_i32_75, %c3_i32, %c3_i32_76, %c0_i32, %c8_i32_77, %c0_i32_78) : (memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<4032xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of7(Produce, 1)
        %c0_79 = arith.constant 0 : index
        %c54 = arith.constant 54 : index
        %c1_80 = arith.constant 1 : index
        scf.for %arg1 = %c0_79 to %c54 step %c1_80 {
          %42 = aie.objectfifo.acquire @of5(Consume, 1) : memref<56x1x64xui8>
          %43 = aie.objectfifo.acquire @of8(Produce, 1) : memref<56x1x24xi8>
          %collapse_shape_92 = memref.collapse_shape %42 [[0, 1, 2]] : memref<56x1x64xui8> into memref<3584xui8>
          %collapse_shape_93 = memref.collapse_shape %43 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %c56_i32_94 = arith.constant 56 : i32
          %c64_i32_95 = arith.constant 64 : i32
          %c24_i32_96 = arith.constant 24 : i32
          %c7_i32_97 = arith.constant 7 : i32
          func.call @f7ee2b1f_conv2dk1_ui8_i8(%collapse_shape_92, %view_52, %collapse_shape_93, %c56_i32_94, %c64_i32_95, %c24_i32_96, %c7_i32_97) : (memref<3584xui8>, memref<1536xi8>, memref<1344xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of5(Consume, 1)
          %44 = aie.objectfifo.acquire @of6(Produce, 1) : memref<56x1x72xui8>
          %collapse_shape_98 = memref.collapse_shape %43 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %collapse_shape_99 = memref.collapse_shape %44 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %c56_i32_100 = arith.constant 56 : i32
          %c24_i32_101 = arith.constant 24 : i32
          %c72_i32_102 = arith.constant 72 : i32
          %c8_i32_103 = arith.constant 8 : i32
          func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_98, %view, %collapse_shape_99, %c56_i32_100, %c24_i32_101, %c72_i32_102, %c8_i32_103) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of6(Produce, 1)
          aie.objectfifo.release @of8(Produce, 1)
          %45:3 = aie.objectfifo.acquire @of6(Consume, 3) : memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>
          %46 = aie.objectfifo.acquire @of7(Produce, 1) : memref<56x1x72xui8>
          %collapse_shape_104 = memref.collapse_shape %45#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_105 = memref.collapse_shape %45#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_106 = memref.collapse_shape %45#2 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_107 = memref.collapse_shape %46 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %c56_i32_108 = arith.constant 56 : i32
          %c1_i32_109 = arith.constant 1 : i32
          %c72_i32_110 = arith.constant 72 : i32
          %c3_i32_111 = arith.constant 3 : i32
          %c3_i32_112 = arith.constant 3 : i32
          %c1_i32_113 = arith.constant 1 : i32
          %c8_i32_114 = arith.constant 8 : i32
          %c0_i32_115 = arith.constant 0 : i32
          func.call @a63dee7f_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_104, %collapse_shape_105, %collapse_shape_106, %view_51, %collapse_shape_107, %c56_i32_108, %c1_i32_109, %c72_i32_110, %c3_i32_111, %c3_i32_112, %c1_i32_113, %c8_i32_114, %c0_i32_115) : (memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<4032xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of6(Consume, 1)
          aie.objectfifo.release @of7(Produce, 1)
        }
        %40:2 = aie.objectfifo.acquire @of6(Consume, 2) : memref<56x1x72xui8>, memref<56x1x72xui8>
        %41 = aie.objectfifo.acquire @of7(Produce, 1) : memref<56x1x72xui8>
        %collapse_shape_81 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_82 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_83 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_84 = memref.collapse_shape %41 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %c56_i32_85 = arith.constant 56 : i32
        %c1_i32_86 = arith.constant 1 : i32
        %c72_i32_87 = arith.constant 72 : i32
        %c3_i32_88 = arith.constant 3 : i32
        %c3_i32_89 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c8_i32_90 = arith.constant 8 : i32
        %c0_i32_91 = arith.constant 0 : i32
        func.call @a63dee7f_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_81, %collapse_shape_82, %collapse_shape_83, %view_51, %collapse_shape_84, %c56_i32_85, %c1_i32_86, %c72_i32_87, %c3_i32_88, %c3_i32_89, %c2_i32, %c8_i32_90, %c0_i32_91) : (memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<4032xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of6(Consume, 2)
        aie.objectfifo.release @of7(Produce, 1)
      }
      aie.end
    }
    %4 = aie.core(%logical_core_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_4[%c0_50][] : memref<11840xi8> to memref<1728xi8>
        %c1728 = arith.constant 1728 : index
        %view_51 = memref.view %buf_4[%c1728][] : memref<11840xi8> to memref<648xi8>
        %c2432 = arith.constant 2432 : index
        %view_52 = memref.view %buf_4[%c2432][] : memref<11840xi8> to memref<2880xi8>
        %c5312 = arith.constant 5312 : index
        %view_53 = memref.view %buf_4[%c5312][] : memref<11840xi8> to memref<4800xi8>
        %c10112 = arith.constant 10112 : index
        %view_54 = memref.view %buf_4[%c10112][] : memref<11840xi8> to memref<1728xi8>
        %32 = aie.objectfifo.acquire @of7(Consume, 1) : memref<56x1x72xui8>
        %33 = aie.objectfifo.acquire @of13(Produce, 1) : memref<56x1x24xi8>
        %34 = aie.objectfifo.acquire @of8(Consume, 1) : memref<56x1x24xi8>
        %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_55 = memref.collapse_shape %33 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %collapse_shape_56 = memref.collapse_shape %34 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %c56_i32 = arith.constant 56 : i32
        %c72_i32 = arith.constant 72 : i32
        %c24_i32 = arith.constant 24 : i32
        %c11_i32 = arith.constant 11 : i32
        %c1_i32 = arith.constant 1 : i32
        func.call @"8d516248_conv2dk1_skip_ui8_i8_i8"(%collapse_shape, %view_54, %collapse_shape_55, %collapse_shape_56, %c56_i32, %c72_i32, %c24_i32, %c11_i32, %c1_i32) : (memref<4032xui8>, memref<1728xi8>, memref<1344xi8>, memref<1344xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of8(Consume, 1)
        aie.objectfifo.release @of7(Consume, 1)
        aie.objectfifo.release @of13(Produce, 1)
        %35 = aie.objectfifo.acquire @of7(Consume, 1) : memref<56x1x72xui8>
        %36 = aie.objectfifo.acquire @of13(Produce, 1) : memref<56x1x24xi8>
        %37 = aie.objectfifo.acquire @of8(Consume, 1) : memref<56x1x24xi8>
        %collapse_shape_57 = memref.collapse_shape %35 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_58 = memref.collapse_shape %36 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %collapse_shape_59 = memref.collapse_shape %37 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %c56_i32_60 = arith.constant 56 : i32
        %c72_i32_61 = arith.constant 72 : i32
        %c24_i32_62 = arith.constant 24 : i32
        %c11_i32_63 = arith.constant 11 : i32
        %c1_i32_64 = arith.constant 1 : i32
        func.call @"8d516248_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_57, %view_54, %collapse_shape_58, %collapse_shape_59, %c56_i32_60, %c72_i32_61, %c24_i32_62, %c11_i32_63, %c1_i32_64) : (memref<4032xui8>, memref<1728xi8>, memref<1344xi8>, memref<1344xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of8(Consume, 1)
        aie.objectfifo.release @of7(Consume, 1)
        aie.objectfifo.release @of13(Produce, 1)
        %38:2 = aie.objectfifo.acquire @of13(Consume, 2) : memref<56x1x24xi8>, memref<56x1x24xi8>
        %39:2 = aie.objectfifo.acquire @of9(Produce, 2) : memref<56x1x72xui8>, memref<56x1x72xui8>
        %collapse_shape_65 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %collapse_shape_66 = memref.collapse_shape %39#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %c56_i32_67 = arith.constant 56 : i32
        %c24_i32_68 = arith.constant 24 : i32
        %c72_i32_69 = arith.constant 72 : i32
        %c7_i32 = arith.constant 7 : i32
        func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_65, %view, %collapse_shape_66, %c56_i32_67, %c24_i32_68, %c72_i32_69, %c7_i32) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
        %collapse_shape_70 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
        %collapse_shape_71 = memref.collapse_shape %39#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %c56_i32_72 = arith.constant 56 : i32
        %c24_i32_73 = arith.constant 24 : i32
        %c72_i32_74 = arith.constant 72 : i32
        %c7_i32_75 = arith.constant 7 : i32
        func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_70, %view, %collapse_shape_71, %c56_i32_72, %c24_i32_73, %c72_i32_74, %c7_i32_75) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of9(Produce, 2)
        aie.objectfifo.release @of13(Consume, 2)
        %40:2 = aie.objectfifo.acquire @of9(Consume, 2) : memref<56x1x72xui8>, memref<56x1x72xui8>
        %41 = aie.objectfifo.acquire @of11(Produce, 1) : memref<28x1x72xui8>
        %collapse_shape_76 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_77 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_78 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
        %collapse_shape_79 = memref.collapse_shape %41 [[0, 1, 2]] : memref<28x1x72xui8> into memref<2016xui8>
        %c56_i32_80 = arith.constant 56 : i32
        %c1_i32_81 = arith.constant 1 : i32
        %c72_i32_82 = arith.constant 72 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_83 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_84 = arith.constant 0 : i32
        func.call @c374361d_conv2dk3_dw_stride2_relu_ui8_ui8(%collapse_shape_76, %collapse_shape_77, %collapse_shape_78, %view_51, %collapse_shape_79, %c56_i32_80, %c1_i32_81, %c72_i32_82, %c3_i32, %c3_i32_83, %c0_i32, %c8_i32, %c0_i32_84) : (memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<2016xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of9(Consume, 1)
        aie.objectfifo.release @of11(Produce, 1)
        %42 = aie.objectfifo.acquire @of11(Consume, 1) : memref<28x1x72xui8>
        %43 = aie.objectfifo.acquire @of10(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_85 = memref.collapse_shape %42 [[0, 1, 2]] : memref<28x1x72xui8> into memref<2016xui8>
        %collapse_shape_86 = memref.collapse_shape %43 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32 = arith.constant 28 : i32
        %c72_i32_87 = arith.constant 72 : i32
        %c40_i32 = arith.constant 40 : i32
        %c8_i32_88 = arith.constant 8 : i32
        func.call @aff12032_conv2dk1_ui8_i8(%collapse_shape_85, %view_52, %collapse_shape_86, %c28_i32, %c72_i32_87, %c40_i32, %c8_i32_88) : (memref<2016xui8>, memref<2880xi8>, memref<1120xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of11(Consume, 1)
        %44 = aie.objectfifo.acquire @of12(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_89 = memref.collapse_shape %43 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_90 = memref.collapse_shape %44 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_91 = arith.constant 28 : i32
        %c40_i32_92 = arith.constant 40 : i32
        %c120_i32 = arith.constant 120 : i32
        %c9_i32 = arith.constant 9 : i32
        func.call @"453c6c71_conv2dk1_relu_i8_ui8"(%collapse_shape_89, %view_53, %collapse_shape_90, %c28_i32_91, %c40_i32_92, %c120_i32, %c9_i32) : (memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of12(Produce, 1)
        aie.objectfifo.release @of10(Produce, 1)
        %c0_93 = arith.constant 0 : index
        %c27 = arith.constant 27 : index
        %c1_94 = arith.constant 1 : index
        scf.for %arg1 = %c0_93 to %c27 step %c1_94 {
          %45 = aie.objectfifo.acquire @of7(Consume, 1) : memref<56x1x72xui8>
          %46 = aie.objectfifo.acquire @of13(Produce, 1) : memref<56x1x24xi8>
          %47 = aie.objectfifo.acquire @of8(Consume, 1) : memref<56x1x24xi8>
          %collapse_shape_95 = memref.collapse_shape %45 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_96 = memref.collapse_shape %46 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %collapse_shape_97 = memref.collapse_shape %47 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %c56_i32_98 = arith.constant 56 : i32
          %c72_i32_99 = arith.constant 72 : i32
          %c24_i32_100 = arith.constant 24 : i32
          %c11_i32_101 = arith.constant 11 : i32
          %c1_i32_102 = arith.constant 1 : i32
          func.call @"8d516248_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_95, %view_54, %collapse_shape_96, %collapse_shape_97, %c56_i32_98, %c72_i32_99, %c24_i32_100, %c11_i32_101, %c1_i32_102) : (memref<4032xui8>, memref<1728xi8>, memref<1344xi8>, memref<1344xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of8(Consume, 1)
          aie.objectfifo.release @of7(Consume, 1)
          aie.objectfifo.release @of13(Produce, 1)
          %48 = aie.objectfifo.acquire @of7(Consume, 1) : memref<56x1x72xui8>
          %49 = aie.objectfifo.acquire @of13(Produce, 1) : memref<56x1x24xi8>
          %50 = aie.objectfifo.acquire @of8(Consume, 1) : memref<56x1x24xi8>
          %collapse_shape_103 = memref.collapse_shape %48 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_104 = memref.collapse_shape %49 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %collapse_shape_105 = memref.collapse_shape %50 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %c56_i32_106 = arith.constant 56 : i32
          %c72_i32_107 = arith.constant 72 : i32
          %c24_i32_108 = arith.constant 24 : i32
          %c11_i32_109 = arith.constant 11 : i32
          %c1_i32_110 = arith.constant 1 : i32
          func.call @"8d516248_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_103, %view_54, %collapse_shape_104, %collapse_shape_105, %c56_i32_106, %c72_i32_107, %c24_i32_108, %c11_i32_109, %c1_i32_110) : (memref<4032xui8>, memref<1728xi8>, memref<1344xi8>, memref<1344xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of8(Consume, 1)
          aie.objectfifo.release @of7(Consume, 1)
          aie.objectfifo.release @of13(Produce, 1)
          %51:2 = aie.objectfifo.acquire @of13(Consume, 2) : memref<56x1x24xi8>, memref<56x1x24xi8>
          %52:2 = aie.objectfifo.acquire @of9(Produce, 2) : memref<56x1x72xui8>, memref<56x1x72xui8>
          %collapse_shape_111 = memref.collapse_shape %51#0 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %collapse_shape_112 = memref.collapse_shape %52#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %c56_i32_113 = arith.constant 56 : i32
          %c24_i32_114 = arith.constant 24 : i32
          %c72_i32_115 = arith.constant 72 : i32
          %c7_i32_116 = arith.constant 7 : i32
          func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_111, %view, %collapse_shape_112, %c56_i32_113, %c24_i32_114, %c72_i32_115, %c7_i32_116) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
          %collapse_shape_117 = memref.collapse_shape %51#1 [[0, 1, 2]] : memref<56x1x24xi8> into memref<1344xi8>
          %collapse_shape_118 = memref.collapse_shape %52#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %c56_i32_119 = arith.constant 56 : i32
          %c24_i32_120 = arith.constant 24 : i32
          %c72_i32_121 = arith.constant 72 : i32
          %c7_i32_122 = arith.constant 7 : i32
          func.call @b33ddc7d_conv2dk1_relu_i8_ui8(%collapse_shape_117, %view, %collapse_shape_118, %c56_i32_119, %c24_i32_120, %c72_i32_121, %c7_i32_122) : (memref<1344xi8>, memref<1728xi8>, memref<4032xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of9(Produce, 2)
          aie.objectfifo.release @of13(Consume, 2)
          %53:3 = aie.objectfifo.acquire @of9(Consume, 3) : memref<56x1x72xui8>, memref<56x1x72xui8>, memref<56x1x72xui8>
          %54 = aie.objectfifo.acquire @of11(Produce, 1) : memref<28x1x72xui8>
          %collapse_shape_123 = memref.collapse_shape %53#0 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_124 = memref.collapse_shape %53#1 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_125 = memref.collapse_shape %53#2 [[0, 1, 2]] : memref<56x1x72xui8> into memref<4032xui8>
          %collapse_shape_126 = memref.collapse_shape %54 [[0, 1, 2]] : memref<28x1x72xui8> into memref<2016xui8>
          %c56_i32_127 = arith.constant 56 : i32
          %c1_i32_128 = arith.constant 1 : i32
          %c72_i32_129 = arith.constant 72 : i32
          %c3_i32_130 = arith.constant 3 : i32
          %c3_i32_131 = arith.constant 3 : i32
          %c1_i32_132 = arith.constant 1 : i32
          %c8_i32_133 = arith.constant 8 : i32
          %c0_i32_134 = arith.constant 0 : i32
          func.call @c374361d_conv2dk3_dw_stride2_relu_ui8_ui8(%collapse_shape_123, %collapse_shape_124, %collapse_shape_125, %view_51, %collapse_shape_126, %c56_i32_127, %c1_i32_128, %c72_i32_129, %c3_i32_130, %c3_i32_131, %c1_i32_132, %c8_i32_133, %c0_i32_134) : (memref<4032xui8>, memref<4032xui8>, memref<4032xui8>, memref<648xi8>, memref<2016xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of9(Consume, 2)
          aie.objectfifo.release @of11(Produce, 1)
          %55 = aie.objectfifo.acquire @of11(Consume, 1) : memref<28x1x72xui8>
          %56 = aie.objectfifo.acquire @of10(Produce, 1) : memref<28x1x40xi8>
          %collapse_shape_135 = memref.collapse_shape %55 [[0, 1, 2]] : memref<28x1x72xui8> into memref<2016xui8>
          %collapse_shape_136 = memref.collapse_shape %56 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %c28_i32_137 = arith.constant 28 : i32
          %c72_i32_138 = arith.constant 72 : i32
          %c40_i32_139 = arith.constant 40 : i32
          %c8_i32_140 = arith.constant 8 : i32
          func.call @aff12032_conv2dk1_ui8_i8(%collapse_shape_135, %view_52, %collapse_shape_136, %c28_i32_137, %c72_i32_138, %c40_i32_139, %c8_i32_140) : (memref<2016xui8>, memref<2880xi8>, memref<1120xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of11(Consume, 1)
          %57 = aie.objectfifo.acquire @of12(Produce, 1) : memref<28x1x120xui8>
          %collapse_shape_141 = memref.collapse_shape %56 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %collapse_shape_142 = memref.collapse_shape %57 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %c28_i32_143 = arith.constant 28 : i32
          %c40_i32_144 = arith.constant 40 : i32
          %c120_i32_145 = arith.constant 120 : i32
          %c9_i32_146 = arith.constant 9 : i32
          func.call @"453c6c71_conv2dk1_relu_i8_ui8"(%collapse_shape_141, %view_53, %collapse_shape_142, %c28_i32_143, %c40_i32_144, %c120_i32_145, %c9_i32_146) : (memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of12(Produce, 1)
          aie.objectfifo.release @of10(Produce, 1)
        }
        aie.objectfifo.release @of9(Consume, 1)
      }
      aie.end
    }
    %5 = aie.core(%logical_core_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_5[%c0_50][] : memref<16576xi8> to memref<1080xi8>
        %c1088 = arith.constant 1088 : index
        %view_51 = memref.view %buf_5[%c1088][] : memref<16576xi8> to memref<4800xi8>
        %c5888 = arith.constant 5888 : index
        %view_52 = memref.view %buf_5[%c5888][] : memref<16576xi8> to memref<4800xi8>
        %c10688 = arith.constant 10688 : index
        %view_53 = memref.view %buf_5[%c10688][] : memref<16576xi8> to memref<1080xi8>
        %c11776 = arith.constant 11776 : index
        %view_54 = memref.view %buf_5[%c11776][] : memref<16576xi8> to memref<4800xi8>
        %32:2 = aie.objectfifo.acquire @of10(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
        %33:2 = aie.objectfifo.acquire @of12(Consume, 2) : memref<28x1x120xui8>, memref<28x1x120xui8>
        %34 = aie.objectfifo.acquire @of15(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape = memref.collapse_shape %33#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_55 = memref.collapse_shape %33#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_56 = memref.collapse_shape %33#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_57 = memref.collapse_shape %34 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32 = arith.constant 28 : i32
        %c1_i32 = arith.constant 1 : i32
        %c120_i32 = arith.constant 120 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_58 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_59 = arith.constant 0 : i32
        func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape, %collapse_shape_55, %collapse_shape_56, %view, %collapse_shape_57, %c28_i32, %c1_i32, %c120_i32, %c3_i32, %c3_i32_58, %c0_i32, %c7_i32, %c0_i32_59) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of15(Produce, 1)
        %35 = aie.objectfifo.acquire @of15(Consume, 1) : memref<28x1x120xui8>
        %36 = aie.objectfifo.acquire @of16(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_60 = memref.collapse_shape %35 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_61 = memref.collapse_shape %36 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_62 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32_63 = arith.constant 28 : i32
        %c120_i32_64 = arith.constant 120 : i32
        %c40_i32 = arith.constant 40 : i32
        %c11_i32 = arith.constant 11 : i32
        %c0_i32_65 = arith.constant 0 : i32
        func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_60, %view_51, %collapse_shape_61, %collapse_shape_62, %c28_i32_63, %c120_i32_64, %c40_i32, %c11_i32, %c0_i32_65) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of10(Consume, 1)
        aie.objectfifo.release @of15(Consume, 1)
        aie.objectfifo.release @of16(Produce, 1)
        %37 = aie.objectfifo.acquire @of16(Consume, 1) : memref<28x1x40xi8>
        %38 = aie.objectfifo.acquire @of17(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_66 = memref.collapse_shape %37 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_67 = memref.collapse_shape %38 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_68 = arith.constant 28 : i32
        %c40_i32_69 = arith.constant 40 : i32
        %c120_i32_70 = arith.constant 120 : i32
        %c9_i32 = arith.constant 9 : i32
        func.call @"453c6c71_conv2dk1_relu_i8_ui8"(%collapse_shape_66, %view_52, %collapse_shape_67, %c28_i32_68, %c40_i32_69, %c120_i32_70, %c9_i32) : (memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of17(Produce, 1)
        %39:2 = aie.objectfifo.acquire @of10(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
        %40:3 = aie.objectfifo.acquire @of12(Consume, 3) : memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>
        %41 = aie.objectfifo.acquire @of15(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_71 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_72 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_73 = memref.collapse_shape %40#2 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_74 = memref.collapse_shape %41 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_75 = arith.constant 28 : i32
        %c1_i32_76 = arith.constant 1 : i32
        %c120_i32_77 = arith.constant 120 : i32
        %c3_i32_78 = arith.constant 3 : i32
        %c3_i32_79 = arith.constant 3 : i32
        %c1_i32_80 = arith.constant 1 : i32
        %c7_i32_81 = arith.constant 7 : i32
        %c0_i32_82 = arith.constant 0 : i32
        func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_71, %collapse_shape_72, %collapse_shape_73, %view, %collapse_shape_74, %c28_i32_75, %c1_i32_76, %c120_i32_77, %c3_i32_78, %c3_i32_79, %c1_i32_80, %c7_i32_81, %c0_i32_82) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of12(Consume, 1)
        aie.objectfifo.release @of15(Produce, 1)
        %42 = aie.objectfifo.acquire @of15(Consume, 1) : memref<28x1x120xui8>
        %43 = aie.objectfifo.acquire @of16(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_83 = memref.collapse_shape %42 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_84 = memref.collapse_shape %43 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_85 = memref.collapse_shape %39#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32_86 = arith.constant 28 : i32
        %c120_i32_87 = arith.constant 120 : i32
        %c40_i32_88 = arith.constant 40 : i32
        %c11_i32_89 = arith.constant 11 : i32
        %c0_i32_90 = arith.constant 0 : i32
        func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_83, %view_51, %collapse_shape_84, %collapse_shape_85, %c28_i32_86, %c120_i32_87, %c40_i32_88, %c11_i32_89, %c0_i32_90) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of10(Consume, 1)
        aie.objectfifo.release @of15(Consume, 1)
        aie.objectfifo.release @of16(Produce, 1)
        %44:2 = aie.objectfifo.acquire @of16(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
        %45 = aie.objectfifo.acquire @of17(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_91 = memref.collapse_shape %44#1 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_92 = memref.collapse_shape %45 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_93 = arith.constant 28 : i32
        %c40_i32_94 = arith.constant 40 : i32
        %c120_i32_95 = arith.constant 120 : i32
        %c9_i32_96 = arith.constant 9 : i32
        func.call @"453c6c71_conv2dk1_relu_i8_ui8"(%collapse_shape_91, %view_52, %collapse_shape_92, %c28_i32_93, %c40_i32_94, %c120_i32_95, %c9_i32_96) : (memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of17(Produce, 1)
        %46:2 = aie.objectfifo.acquire @of17(Consume, 2) : memref<28x1x120xui8>, memref<28x1x120xui8>
        %47 = aie.objectfifo.acquire @of18(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_97 = memref.collapse_shape %46#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_98 = memref.collapse_shape %46#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_99 = memref.collapse_shape %46#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_100 = memref.collapse_shape %47 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_101 = arith.constant 28 : i32
        %c1_i32_102 = arith.constant 1 : i32
        %c120_i32_103 = arith.constant 120 : i32
        %c3_i32_104 = arith.constant 3 : i32
        %c3_i32_105 = arith.constant 3 : i32
        %c0_i32_106 = arith.constant 0 : i32
        %c7_i32_107 = arith.constant 7 : i32
        %c0_i32_108 = arith.constant 0 : i32
        func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_97, %collapse_shape_98, %collapse_shape_99, %view_53, %collapse_shape_100, %c28_i32_101, %c1_i32_102, %c120_i32_103, %c3_i32_104, %c3_i32_105, %c0_i32_106, %c7_i32_107, %c0_i32_108) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of18(Produce, 1)
        %48 = aie.objectfifo.acquire @of18(Consume, 1) : memref<28x1x120xui8>
        %49 = aie.objectfifo.acquire @of14(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_109 = memref.collapse_shape %48 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_110 = memref.collapse_shape %49 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_111 = memref.collapse_shape %44#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32_112 = arith.constant 28 : i32
        %c120_i32_113 = arith.constant 120 : i32
        %c40_i32_114 = arith.constant 40 : i32
        %c11_i32_115 = arith.constant 11 : i32
        %c1_i32_116 = arith.constant 1 : i32
        func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_109, %view_54, %collapse_shape_110, %collapse_shape_111, %c28_i32_112, %c120_i32_113, %c40_i32_114, %c11_i32_115, %c1_i32_116) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of18(Consume, 1)
        aie.objectfifo.release @of16(Consume, 1)
        aie.objectfifo.release @of14(Produce, 1)
        %c0_117 = arith.constant 0 : index
        %c25 = arith.constant 25 : index
        %c1_118 = arith.constant 1 : index
        scf.for %arg1 = %c0_117 to %c25 step %c1_118 {
          %66:2 = aie.objectfifo.acquire @of10(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
          %67:3 = aie.objectfifo.acquire @of12(Consume, 3) : memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>
          %68 = aie.objectfifo.acquire @of15(Produce, 1) : memref<28x1x120xui8>
          %collapse_shape_184 = memref.collapse_shape %67#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_185 = memref.collapse_shape %67#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_186 = memref.collapse_shape %67#2 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_187 = memref.collapse_shape %68 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %c28_i32_188 = arith.constant 28 : i32
          %c1_i32_189 = arith.constant 1 : i32
          %c120_i32_190 = arith.constant 120 : i32
          %c3_i32_191 = arith.constant 3 : i32
          %c3_i32_192 = arith.constant 3 : i32
          %c1_i32_193 = arith.constant 1 : i32
          %c7_i32_194 = arith.constant 7 : i32
          %c0_i32_195 = arith.constant 0 : i32
          func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_184, %collapse_shape_185, %collapse_shape_186, %view, %collapse_shape_187, %c28_i32_188, %c1_i32_189, %c120_i32_190, %c3_i32_191, %c3_i32_192, %c1_i32_193, %c7_i32_194, %c0_i32_195) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of12(Consume, 1)
          aie.objectfifo.release @of15(Produce, 1)
          %69 = aie.objectfifo.acquire @of15(Consume, 1) : memref<28x1x120xui8>
          %70 = aie.objectfifo.acquire @of16(Produce, 1) : memref<28x1x40xi8>
          %collapse_shape_196 = memref.collapse_shape %69 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_197 = memref.collapse_shape %70 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %collapse_shape_198 = memref.collapse_shape %66#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %c28_i32_199 = arith.constant 28 : i32
          %c120_i32_200 = arith.constant 120 : i32
          %c40_i32_201 = arith.constant 40 : i32
          %c11_i32_202 = arith.constant 11 : i32
          %c0_i32_203 = arith.constant 0 : i32
          func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_196, %view_51, %collapse_shape_197, %collapse_shape_198, %c28_i32_199, %c120_i32_200, %c40_i32_201, %c11_i32_202, %c0_i32_203) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of10(Consume, 1)
          aie.objectfifo.release @of15(Consume, 1)
          aie.objectfifo.release @of16(Produce, 1)
          %71:2 = aie.objectfifo.acquire @of16(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
          %72 = aie.objectfifo.acquire @of17(Produce, 1) : memref<28x1x120xui8>
          %collapse_shape_204 = memref.collapse_shape %71#1 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %collapse_shape_205 = memref.collapse_shape %72 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %c28_i32_206 = arith.constant 28 : i32
          %c40_i32_207 = arith.constant 40 : i32
          %c120_i32_208 = arith.constant 120 : i32
          %c9_i32_209 = arith.constant 9 : i32
          func.call @"453c6c71_conv2dk1_relu_i8_ui8"(%collapse_shape_204, %view_52, %collapse_shape_205, %c28_i32_206, %c40_i32_207, %c120_i32_208, %c9_i32_209) : (memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of17(Produce, 1)
          %73:3 = aie.objectfifo.acquire @of17(Consume, 3) : memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>
          %74 = aie.objectfifo.acquire @of18(Produce, 1) : memref<28x1x120xui8>
          %collapse_shape_210 = memref.collapse_shape %73#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_211 = memref.collapse_shape %73#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_212 = memref.collapse_shape %73#2 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_213 = memref.collapse_shape %74 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %c28_i32_214 = arith.constant 28 : i32
          %c1_i32_215 = arith.constant 1 : i32
          %c120_i32_216 = arith.constant 120 : i32
          %c3_i32_217 = arith.constant 3 : i32
          %c3_i32_218 = arith.constant 3 : i32
          %c1_i32_219 = arith.constant 1 : i32
          %c7_i32_220 = arith.constant 7 : i32
          %c0_i32_221 = arith.constant 0 : i32
          func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_210, %collapse_shape_211, %collapse_shape_212, %view_53, %collapse_shape_213, %c28_i32_214, %c1_i32_215, %c120_i32_216, %c3_i32_217, %c3_i32_218, %c1_i32_219, %c7_i32_220, %c0_i32_221) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of17(Consume, 1)
          aie.objectfifo.release @of18(Produce, 1)
          %75 = aie.objectfifo.acquire @of18(Consume, 1) : memref<28x1x120xui8>
          %76 = aie.objectfifo.acquire @of14(Produce, 1) : memref<28x1x40xi8>
          %collapse_shape_222 = memref.collapse_shape %75 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
          %collapse_shape_223 = memref.collapse_shape %76 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %collapse_shape_224 = memref.collapse_shape %71#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %c28_i32_225 = arith.constant 28 : i32
          %c120_i32_226 = arith.constant 120 : i32
          %c40_i32_227 = arith.constant 40 : i32
          %c11_i32_228 = arith.constant 11 : i32
          %c1_i32_229 = arith.constant 1 : i32
          func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_222, %view_54, %collapse_shape_223, %collapse_shape_224, %c28_i32_225, %c120_i32_226, %c40_i32_227, %c11_i32_228, %c1_i32_229) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of18(Consume, 1)
          aie.objectfifo.release @of16(Consume, 1)
          aie.objectfifo.release @of14(Produce, 1)
        }
        %50:2 = aie.objectfifo.acquire @of12(Consume, 2) : memref<28x1x120xui8>, memref<28x1x120xui8>
        %51 = aie.objectfifo.acquire @of15(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_119 = memref.collapse_shape %50#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_120 = memref.collapse_shape %50#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_121 = memref.collapse_shape %50#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_122 = memref.collapse_shape %51 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_123 = arith.constant 28 : i32
        %c1_i32_124 = arith.constant 1 : i32
        %c120_i32_125 = arith.constant 120 : i32
        %c3_i32_126 = arith.constant 3 : i32
        %c3_i32_127 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c7_i32_128 = arith.constant 7 : i32
        %c0_i32_129 = arith.constant 0 : i32
        func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_119, %collapse_shape_120, %collapse_shape_121, %view, %collapse_shape_122, %c28_i32_123, %c1_i32_124, %c120_i32_125, %c3_i32_126, %c3_i32_127, %c2_i32, %c7_i32_128, %c0_i32_129) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of12(Consume, 2)
        aie.objectfifo.release @of15(Produce, 1)
        %52 = aie.objectfifo.acquire @of10(Consume, 1) : memref<28x1x40xi8>
        %53 = aie.objectfifo.acquire @of15(Consume, 1) : memref<28x1x120xui8>
        %54 = aie.objectfifo.acquire @of16(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_130 = memref.collapse_shape %53 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_131 = memref.collapse_shape %54 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_132 = memref.collapse_shape %52 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32_133 = arith.constant 28 : i32
        %c120_i32_134 = arith.constant 120 : i32
        %c40_i32_135 = arith.constant 40 : i32
        %c11_i32_136 = arith.constant 11 : i32
        %c0_i32_137 = arith.constant 0 : i32
        func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_130, %view_51, %collapse_shape_131, %collapse_shape_132, %c28_i32_133, %c120_i32_134, %c40_i32_135, %c11_i32_136, %c0_i32_137) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of10(Consume, 1)
        aie.objectfifo.release @of15(Consume, 1)
        aie.objectfifo.release @of16(Produce, 1)
        %55:2 = aie.objectfifo.acquire @of16(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
        %56 = aie.objectfifo.acquire @of17(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_138 = memref.collapse_shape %55#1 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_139 = memref.collapse_shape %56 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_140 = arith.constant 28 : i32
        %c40_i32_141 = arith.constant 40 : i32
        %c120_i32_142 = arith.constant 120 : i32
        %c9_i32_143 = arith.constant 9 : i32
        func.call @"453c6c71_conv2dk1_relu_i8_ui8"(%collapse_shape_138, %view_52, %collapse_shape_139, %c28_i32_140, %c40_i32_141, %c120_i32_142, %c9_i32_143) : (memref<1120xi8>, memref<4800xi8>, memref<3360xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of17(Produce, 1)
        %57:3 = aie.objectfifo.acquire @of17(Consume, 3) : memref<28x1x120xui8>, memref<28x1x120xui8>, memref<28x1x120xui8>
        %58 = aie.objectfifo.acquire @of18(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_144 = memref.collapse_shape %57#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_145 = memref.collapse_shape %57#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_146 = memref.collapse_shape %57#2 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_147 = memref.collapse_shape %58 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_148 = arith.constant 28 : i32
        %c1_i32_149 = arith.constant 1 : i32
        %c120_i32_150 = arith.constant 120 : i32
        %c3_i32_151 = arith.constant 3 : i32
        %c3_i32_152 = arith.constant 3 : i32
        %c1_i32_153 = arith.constant 1 : i32
        %c7_i32_154 = arith.constant 7 : i32
        %c0_i32_155 = arith.constant 0 : i32
        func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_144, %collapse_shape_145, %collapse_shape_146, %view_53, %collapse_shape_147, %c28_i32_148, %c1_i32_149, %c120_i32_150, %c3_i32_151, %c3_i32_152, %c1_i32_153, %c7_i32_154, %c0_i32_155) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of17(Consume, 1)
        aie.objectfifo.release @of18(Produce, 1)
        %59 = aie.objectfifo.acquire @of18(Consume, 1) : memref<28x1x120xui8>
        %60 = aie.objectfifo.acquire @of14(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_156 = memref.collapse_shape %59 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_157 = memref.collapse_shape %60 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_158 = memref.collapse_shape %55#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32_159 = arith.constant 28 : i32
        %c120_i32_160 = arith.constant 120 : i32
        %c40_i32_161 = arith.constant 40 : i32
        %c11_i32_162 = arith.constant 11 : i32
        %c1_i32_163 = arith.constant 1 : i32
        func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_156, %view_54, %collapse_shape_157, %collapse_shape_158, %c28_i32_159, %c120_i32_160, %c40_i32_161, %c11_i32_162, %c1_i32_163) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of18(Consume, 1)
        aie.objectfifo.release @of16(Consume, 1)
        aie.objectfifo.release @of14(Produce, 1)
        %61:2 = aie.objectfifo.acquire @of17(Consume, 2) : memref<28x1x120xui8>, memref<28x1x120xui8>
        %62 = aie.objectfifo.acquire @of18(Produce, 1) : memref<28x1x120xui8>
        %collapse_shape_164 = memref.collapse_shape %61#0 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_165 = memref.collapse_shape %61#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_166 = memref.collapse_shape %61#1 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_167 = memref.collapse_shape %62 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %c28_i32_168 = arith.constant 28 : i32
        %c1_i32_169 = arith.constant 1 : i32
        %c120_i32_170 = arith.constant 120 : i32
        %c3_i32_171 = arith.constant 3 : i32
        %c3_i32_172 = arith.constant 3 : i32
        %c2_i32_173 = arith.constant 2 : i32
        %c7_i32_174 = arith.constant 7 : i32
        %c0_i32_175 = arith.constant 0 : i32
        func.call @"3e7b1aaa_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_164, %collapse_shape_165, %collapse_shape_166, %view_53, %collapse_shape_167, %c28_i32_168, %c1_i32_169, %c120_i32_170, %c3_i32_171, %c3_i32_172, %c2_i32_173, %c7_i32_174, %c0_i32_175) : (memref<3360xui8>, memref<3360xui8>, memref<3360xui8>, memref<1080xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of17(Consume, 2)
        aie.objectfifo.release @of18(Produce, 1)
        %63 = aie.objectfifo.acquire @of16(Consume, 1) : memref<28x1x40xi8>
        %64 = aie.objectfifo.acquire @of18(Consume, 1) : memref<28x1x120xui8>
        %65 = aie.objectfifo.acquire @of14(Produce, 1) : memref<28x1x40xi8>
        %collapse_shape_176 = memref.collapse_shape %64 [[0, 1, 2]] : memref<28x1x120xui8> into memref<3360xui8>
        %collapse_shape_177 = memref.collapse_shape %65 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_178 = memref.collapse_shape %63 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %c28_i32_179 = arith.constant 28 : i32
        %c120_i32_180 = arith.constant 120 : i32
        %c40_i32_181 = arith.constant 40 : i32
        %c11_i32_182 = arith.constant 11 : i32
        %c1_i32_183 = arith.constant 1 : i32
        func.call @"6d1fcc7d_conv2dk1_skip_ui8_i8_i8"(%collapse_shape_176, %view_54, %collapse_shape_177, %collapse_shape_178, %c28_i32_179, %c120_i32_180, %c40_i32_181, %c11_i32_182, %c1_i32_183) : (memref<3360xui8>, memref<4800xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of18(Consume, 1)
        aie.objectfifo.release @of16(Consume, 1)
        aie.objectfifo.release @of14(Produce, 1)
      }
      aie.end
    }
    %6 = aie.core(%logical_core_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_6[%c0_50][] : memref<30976xi8> to memref<9600xi8>
        %c9600 = arith.constant 9600 : index
        %view_51 = memref.view %buf_6[%c9600][] : memref<30976xi8> to memref<2160xi8>
        %c11776 = arith.constant 11776 : index
        %view_52 = memref.view %buf_6[%c11776][] : memref<30976xi8> to memref<19200xi8>
        %32:2 = aie.objectfifo.acquire @of14(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
        %33:2 = aie.objectfifo.acquire @of19(Produce, 2) : memref<28x1x240xui8>, memref<28x1x240xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_53 = memref.collapse_shape %33#0 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
        %c28_i32 = arith.constant 28 : i32
        %c40_i32 = arith.constant 40 : i32
        %c240_i32 = arith.constant 240 : i32
        %c8_i32 = arith.constant 8 : i32
        func.call @b8f588ff_conv2dk1_relu_i8_ui8(%collapse_shape, %view, %collapse_shape_53, %c28_i32, %c40_i32, %c240_i32, %c8_i32) : (memref<1120xi8>, memref<9600xi8>, memref<6720xui8>, i32, i32, i32, i32) -> ()
        %collapse_shape_54 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
        %collapse_shape_55 = memref.collapse_shape %33#1 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
        %c28_i32_56 = arith.constant 28 : i32
        %c40_i32_57 = arith.constant 40 : i32
        %c240_i32_58 = arith.constant 240 : i32
        %c8_i32_59 = arith.constant 8 : i32
        func.call @b8f588ff_conv2dk1_relu_i8_ui8(%collapse_shape_54, %view, %collapse_shape_55, %c28_i32_56, %c40_i32_57, %c240_i32_58, %c8_i32_59) : (memref<1120xi8>, memref<9600xi8>, memref<6720xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of19(Produce, 2)
        aie.objectfifo.release @of14(Consume, 2)
        %34:2 = aie.objectfifo.acquire @of19(Consume, 2) : memref<28x1x240xui8>, memref<28x1x240xui8>
        %35 = aie.objectfifo.acquire @of21(Produce, 1) : memref<14x1x240xui8>
        %collapse_shape_60 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
        %collapse_shape_61 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
        %collapse_shape_62 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
        %collapse_shape_63 = memref.collapse_shape %35 [[0, 1, 2]] : memref<14x1x240xui8> into memref<3360xui8>
        %c28_i32_64 = arith.constant 28 : i32
        %c1_i32 = arith.constant 1 : i32
        %c240_i32_65 = arith.constant 240 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_66 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_67 = arith.constant 0 : i32
        func.call @"5a8aa48c_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape_60, %collapse_shape_61, %collapse_shape_62, %view_51, %collapse_shape_63, %c28_i32_64, %c1_i32, %c240_i32_65, %c3_i32, %c3_i32_66, %c0_i32, %c7_i32, %c0_i32_67) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<2160xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of19(Consume, 1)
        aie.objectfifo.release @of21(Produce, 1)
        %36 = aie.objectfifo.acquire @of21(Consume, 1) : memref<14x1x240xui8>
        %37 = aie.objectfifo.acquire @of20(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_68 = memref.collapse_shape %36 [[0, 1, 2]] : memref<14x1x240xui8> into memref<3360xui8>
        %collapse_shape_69 = memref.collapse_shape %37 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32 = arith.constant 14 : i32
        %c240_i32_70 = arith.constant 240 : i32
        %c80_i32 = arith.constant 80 : i32
        %c8_i32_71 = arith.constant 8 : i32
        func.call @a7cf1c86_conv2dk1_ui8_i8(%collapse_shape_68, %view_52, %collapse_shape_69, %c14_i32, %c240_i32_70, %c80_i32, %c8_i32_71) : (memref<3360xui8>, memref<19200xi8>, memref<1120xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of21(Consume, 1)
        aie.objectfifo.release @of20(Produce, 1)
        %c0_72 = arith.constant 0 : index
        %c13 = arith.constant 13 : index
        %c1_73 = arith.constant 1 : index
        scf.for %arg1 = %c0_72 to %c13 step %c1_73 {
          %38:2 = aie.objectfifo.acquire @of14(Consume, 2) : memref<28x1x40xi8>, memref<28x1x40xi8>
          %39:2 = aie.objectfifo.acquire @of19(Produce, 2) : memref<28x1x240xui8>, memref<28x1x240xui8>
          %collapse_shape_74 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %collapse_shape_75 = memref.collapse_shape %39#0 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
          %c28_i32_76 = arith.constant 28 : i32
          %c40_i32_77 = arith.constant 40 : i32
          %c240_i32_78 = arith.constant 240 : i32
          %c8_i32_79 = arith.constant 8 : i32
          func.call @b8f588ff_conv2dk1_relu_i8_ui8(%collapse_shape_74, %view, %collapse_shape_75, %c28_i32_76, %c40_i32_77, %c240_i32_78, %c8_i32_79) : (memref<1120xi8>, memref<9600xi8>, memref<6720xui8>, i32, i32, i32, i32) -> ()
          %collapse_shape_80 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<28x1x40xi8> into memref<1120xi8>
          %collapse_shape_81 = memref.collapse_shape %39#1 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
          %c28_i32_82 = arith.constant 28 : i32
          %c40_i32_83 = arith.constant 40 : i32
          %c240_i32_84 = arith.constant 240 : i32
          %c8_i32_85 = arith.constant 8 : i32
          func.call @b8f588ff_conv2dk1_relu_i8_ui8(%collapse_shape_80, %view, %collapse_shape_81, %c28_i32_82, %c40_i32_83, %c240_i32_84, %c8_i32_85) : (memref<1120xi8>, memref<9600xi8>, memref<6720xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of19(Produce, 2)
          aie.objectfifo.release @of14(Consume, 2)
          %40:3 = aie.objectfifo.acquire @of19(Consume, 3) : memref<28x1x240xui8>, memref<28x1x240xui8>, memref<28x1x240xui8>
          %41 = aie.objectfifo.acquire @of21(Produce, 1) : memref<14x1x240xui8>
          %collapse_shape_86 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
          %collapse_shape_87 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
          %collapse_shape_88 = memref.collapse_shape %40#2 [[0, 1, 2]] : memref<28x1x240xui8> into memref<6720xui8>
          %collapse_shape_89 = memref.collapse_shape %41 [[0, 1, 2]] : memref<14x1x240xui8> into memref<3360xui8>
          %c28_i32_90 = arith.constant 28 : i32
          %c1_i32_91 = arith.constant 1 : i32
          %c240_i32_92 = arith.constant 240 : i32
          %c3_i32_93 = arith.constant 3 : i32
          %c3_i32_94 = arith.constant 3 : i32
          %c1_i32_95 = arith.constant 1 : i32
          %c7_i32_96 = arith.constant 7 : i32
          %c0_i32_97 = arith.constant 0 : i32
          func.call @"5a8aa48c_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape_86, %collapse_shape_87, %collapse_shape_88, %view_51, %collapse_shape_89, %c28_i32_90, %c1_i32_91, %c240_i32_92, %c3_i32_93, %c3_i32_94, %c1_i32_95, %c7_i32_96, %c0_i32_97) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<2160xi8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of19(Consume, 2)
          aie.objectfifo.release @of21(Produce, 1)
          %42 = aie.objectfifo.acquire @of21(Consume, 1) : memref<14x1x240xui8>
          %43 = aie.objectfifo.acquire @of20(Produce, 1) : memref<14x1x80xi8>
          %collapse_shape_98 = memref.collapse_shape %42 [[0, 1, 2]] : memref<14x1x240xui8> into memref<3360xui8>
          %collapse_shape_99 = memref.collapse_shape %43 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %c14_i32_100 = arith.constant 14 : i32
          %c240_i32_101 = arith.constant 240 : i32
          %c80_i32_102 = arith.constant 80 : i32
          %c8_i32_103 = arith.constant 8 : i32
          func.call @a7cf1c86_conv2dk1_ui8_i8(%collapse_shape_98, %view_52, %collapse_shape_99, %c14_i32_100, %c240_i32_101, %c80_i32_102, %c8_i32_103) : (memref<3360xui8>, memref<19200xi8>, memref<1120xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of21(Consume, 1)
          aie.objectfifo.release @of20(Produce, 1)
        }
        aie.objectfifo.release @of19(Consume, 1)
      }
      aie.end
    }
    %7 = aie.core(%logical_core_6) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_7[%c0_50][] : memref<33856xi8> to memref<16000xi8>
        %c16000 = arith.constant 16000 : index
        %view_51 = memref.view %buf_7[%c16000][] : memref<33856xi8> to memref<1800xi8>
        %c17856 = arith.constant 17856 : index
        %view_52 = memref.view %buf_7[%c17856][] : memref<33856xi8> to memref<16000xi8>
        %32:2 = aie.objectfifo.acquire @of20(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
        %33:2 = aie.objectfifo.acquire @of22(Produce, 2) : memref<14x1x200xui8>, memref<14x1x200xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_53 = memref.collapse_shape %33#0 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %c14_i32 = arith.constant 14 : i32
        %c80_i32 = arith.constant 80 : i32
        %c200_i32 = arith.constant 200 : i32
        %c9_i32 = arith.constant 9 : i32
        func.call @ee5a41d0_conv2dk1_relu_i8_ui8(%collapse_shape, %view, %collapse_shape_53, %c14_i32, %c80_i32, %c200_i32, %c9_i32) : (memref<1120xi8>, memref<16000xi8>, memref<2800xui8>, i32, i32, i32, i32) -> ()
        %collapse_shape_54 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_55 = memref.collapse_shape %33#1 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %c14_i32_56 = arith.constant 14 : i32
        %c80_i32_57 = arith.constant 80 : i32
        %c200_i32_58 = arith.constant 200 : i32
        %c9_i32_59 = arith.constant 9 : i32
        func.call @ee5a41d0_conv2dk1_relu_i8_ui8(%collapse_shape_54, %view, %collapse_shape_55, %c14_i32_56, %c80_i32_57, %c200_i32_58, %c9_i32_59) : (memref<1120xi8>, memref<16000xi8>, memref<2800xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of22(Produce, 2)
        %34:2 = aie.objectfifo.acquire @of22(Consume, 2) : memref<14x1x200xui8>, memref<14x1x200xui8>
        %35 = aie.objectfifo.acquire @of24(Produce, 1) : memref<14x1x200xui8>
        %collapse_shape_60 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_61 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_62 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_63 = memref.collapse_shape %35 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %c14_i32_64 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c200_i32_65 = arith.constant 200 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_66 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_67 = arith.constant 0 : i32
        func.call @"4cb11a8b_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_60, %collapse_shape_61, %collapse_shape_62, %view_51, %collapse_shape_63, %c14_i32_64, %c1_i32, %c200_i32_65, %c3_i32, %c3_i32_66, %c0_i32, %c7_i32, %c0_i32_67) : (memref<2800xui8>, memref<2800xui8>, memref<2800xui8>, memref<1800xi8>, memref<2800xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of24(Produce, 1)
        %36 = aie.objectfifo.acquire @of24(Consume, 1) : memref<14x1x200xui8>
        %37 = aie.objectfifo.acquire @of23(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_68 = memref.collapse_shape %36 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_69 = memref.collapse_shape %37 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_70 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_71 = arith.constant 14 : i32
        %c200_i32_72 = arith.constant 200 : i32
        %c80_i32_73 = arith.constant 80 : i32
        %c12_i32 = arith.constant 12 : i32
        %c0_i32_74 = arith.constant 0 : i32
        func.call @f245367d_conv2dk1_skip_ui8_i8_i8(%collapse_shape_68, %view_52, %collapse_shape_69, %collapse_shape_70, %c14_i32_71, %c200_i32_72, %c80_i32_73, %c12_i32, %c0_i32_74) : (memref<2800xui8>, memref<16000xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of20(Consume, 1)
        aie.objectfifo.release @of24(Consume, 1)
        aie.objectfifo.release @of23(Produce, 1)
        %c0_75 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_76 = arith.constant 1 : index
        scf.for %arg1 = %c0_75 to %c12 step %c1_76 {
          %43:2 = aie.objectfifo.acquire @of20(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
          %44 = aie.objectfifo.acquire @of22(Produce, 1) : memref<14x1x200xui8>
          %collapse_shape_96 = memref.collapse_shape %43#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_97 = memref.collapse_shape %44 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
          %c14_i32_98 = arith.constant 14 : i32
          %c80_i32_99 = arith.constant 80 : i32
          %c200_i32_100 = arith.constant 200 : i32
          %c9_i32_101 = arith.constant 9 : i32
          func.call @ee5a41d0_conv2dk1_relu_i8_ui8(%collapse_shape_96, %view, %collapse_shape_97, %c14_i32_98, %c80_i32_99, %c200_i32_100, %c9_i32_101) : (memref<1120xi8>, memref<16000xi8>, memref<2800xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of22(Produce, 1)
          %45:3 = aie.objectfifo.acquire @of22(Consume, 3) : memref<14x1x200xui8>, memref<14x1x200xui8>, memref<14x1x200xui8>
          %46 = aie.objectfifo.acquire @of24(Produce, 1) : memref<14x1x200xui8>
          %collapse_shape_102 = memref.collapse_shape %45#0 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
          %collapse_shape_103 = memref.collapse_shape %45#1 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
          %collapse_shape_104 = memref.collapse_shape %45#2 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
          %collapse_shape_105 = memref.collapse_shape %46 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
          %c14_i32_106 = arith.constant 14 : i32
          %c1_i32_107 = arith.constant 1 : i32
          %c200_i32_108 = arith.constant 200 : i32
          %c3_i32_109 = arith.constant 3 : i32
          %c3_i32_110 = arith.constant 3 : i32
          %c1_i32_111 = arith.constant 1 : i32
          %c7_i32_112 = arith.constant 7 : i32
          %c0_i32_113 = arith.constant 0 : i32
          func.call @"4cb11a8b_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_102, %collapse_shape_103, %collapse_shape_104, %view_51, %collapse_shape_105, %c14_i32_106, %c1_i32_107, %c200_i32_108, %c3_i32_109, %c3_i32_110, %c1_i32_111, %c7_i32_112, %c0_i32_113) : (memref<2800xui8>, memref<2800xui8>, memref<2800xui8>, memref<1800xi8>, memref<2800xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of22(Consume, 1)
          aie.objectfifo.release @of24(Produce, 1)
          %47 = aie.objectfifo.acquire @of24(Consume, 1) : memref<14x1x200xui8>
          %48 = aie.objectfifo.acquire @of23(Produce, 1) : memref<14x1x80xi8>
          %collapse_shape_114 = memref.collapse_shape %47 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
          %collapse_shape_115 = memref.collapse_shape %48 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_116 = memref.collapse_shape %43#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %c14_i32_117 = arith.constant 14 : i32
          %c200_i32_118 = arith.constant 200 : i32
          %c80_i32_119 = arith.constant 80 : i32
          %c12_i32_120 = arith.constant 12 : i32
          %c0_i32_121 = arith.constant 0 : i32
          func.call @f245367d_conv2dk1_skip_ui8_i8_i8(%collapse_shape_114, %view_52, %collapse_shape_115, %collapse_shape_116, %c14_i32_117, %c200_i32_118, %c80_i32_119, %c12_i32_120, %c0_i32_121) : (memref<2800xui8>, memref<16000xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of20(Consume, 1)
          aie.objectfifo.release @of24(Consume, 1)
          aie.objectfifo.release @of23(Produce, 1)
        }
        %38:2 = aie.objectfifo.acquire @of22(Consume, 2) : memref<14x1x200xui8>, memref<14x1x200xui8>
        %39 = aie.objectfifo.acquire @of24(Produce, 1) : memref<14x1x200xui8>
        %collapse_shape_77 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_78 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_79 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_80 = memref.collapse_shape %39 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %c14_i32_81 = arith.constant 14 : i32
        %c1_i32_82 = arith.constant 1 : i32
        %c200_i32_83 = arith.constant 200 : i32
        %c3_i32_84 = arith.constant 3 : i32
        %c3_i32_85 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c7_i32_86 = arith.constant 7 : i32
        %c0_i32_87 = arith.constant 0 : i32
        func.call @"4cb11a8b_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_77, %collapse_shape_78, %collapse_shape_79, %view_51, %collapse_shape_80, %c14_i32_81, %c1_i32_82, %c200_i32_83, %c3_i32_84, %c3_i32_85, %c2_i32, %c7_i32_86, %c0_i32_87) : (memref<2800xui8>, memref<2800xui8>, memref<2800xui8>, memref<1800xi8>, memref<2800xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of22(Consume, 2)
        aie.objectfifo.release @of24(Produce, 1)
        %40 = aie.objectfifo.acquire @of20(Consume, 1) : memref<14x1x80xi8>
        %41 = aie.objectfifo.acquire @of24(Consume, 1) : memref<14x1x200xui8>
        %42 = aie.objectfifo.acquire @of23(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_88 = memref.collapse_shape %41 [[0, 1, 2]] : memref<14x1x200xui8> into memref<2800xui8>
        %collapse_shape_89 = memref.collapse_shape %42 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_90 = memref.collapse_shape %40 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_91 = arith.constant 14 : i32
        %c200_i32_92 = arith.constant 200 : i32
        %c80_i32_93 = arith.constant 80 : i32
        %c12_i32_94 = arith.constant 12 : i32
        %c0_i32_95 = arith.constant 0 : i32
        func.call @f245367d_conv2dk1_skip_ui8_i8_i8(%collapse_shape_88, %view_52, %collapse_shape_89, %collapse_shape_90, %c14_i32_91, %c200_i32_92, %c80_i32_93, %c12_i32_94, %c0_i32_95) : (memref<2800xui8>, memref<16000xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of20(Consume, 1)
        aie.objectfifo.release @of24(Consume, 1)
        aie.objectfifo.release @of23(Produce, 1)
      }
      aie.end
    }
    %8 = aie.core(%logical_core_7) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_8[%c0_50][] : memref<62208xi8> to memref<14720xi8>
        %c14720 = arith.constant 14720 : index
        %view_51 = memref.view %buf_8[%c14720][] : memref<62208xi8> to memref<1656xi8>
        %c16384 = arith.constant 16384 : index
        %view_52 = memref.view %buf_8[%c16384][] : memref<62208xi8> to memref<14720xi8>
        %c31104 = arith.constant 31104 : index
        %view_53 = memref.view %buf_8[%c31104][] : memref<62208xi8> to memref<14720xi8>
        %c45824 = arith.constant 45824 : index
        %view_54 = memref.view %buf_8[%c45824][] : memref<62208xi8> to memref<1656xi8>
        %c47488 = arith.constant 47488 : index
        %view_55 = memref.view %buf_8[%c47488][] : memref<62208xi8> to memref<14720xi8>
        %32:2 = aie.objectfifo.acquire @of23(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
        %33:2 = aie.objectfifo.acquire @of26(Produce, 2) : memref<14x1x184xui8>, memref<14x1x184xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_56 = memref.collapse_shape %33#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32 = arith.constant 14 : i32
        %c80_i32 = arith.constant 80 : i32
        %c184_i32 = arith.constant 184 : i32
        %c9_i32 = arith.constant 9 : i32
        func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape, %view, %collapse_shape_56, %c14_i32, %c80_i32, %c184_i32, %c9_i32) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
        %collapse_shape_57 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_58 = memref.collapse_shape %33#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_59 = arith.constant 14 : i32
        %c80_i32_60 = arith.constant 80 : i32
        %c184_i32_61 = arith.constant 184 : i32
        %c9_i32_62 = arith.constant 9 : i32
        func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_57, %view, %collapse_shape_58, %c14_i32_59, %c80_i32_60, %c184_i32_61, %c9_i32_62) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of26(Produce, 2)
        %34:2 = aie.objectfifo.acquire @of26(Consume, 2) : memref<14x1x184xui8>, memref<14x1x184xui8>
        %35 = aie.objectfifo.acquire @of27(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_63 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_64 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_65 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_66 = memref.collapse_shape %35 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_67 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c184_i32_68 = arith.constant 184 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_69 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_70 = arith.constant 0 : i32
        func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_63, %collapse_shape_64, %collapse_shape_65, %view_51, %collapse_shape_66, %c14_i32_67, %c1_i32, %c184_i32_68, %c3_i32, %c3_i32_69, %c0_i32, %c7_i32, %c0_i32_70) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of27(Produce, 1)
        %36 = aie.objectfifo.acquire @of27(Consume, 1) : memref<14x1x184xui8>
        %37 = aie.objectfifo.acquire @of28(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_71 = memref.collapse_shape %36 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_72 = memref.collapse_shape %37 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_73 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_74 = arith.constant 14 : i32
        %c184_i32_75 = arith.constant 184 : i32
        %c80_i32_76 = arith.constant 80 : i32
        %c12_i32 = arith.constant 12 : i32
        %c0_i32_77 = arith.constant 0 : i32
        func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_71, %view_52, %collapse_shape_72, %collapse_shape_73, %c14_i32_74, %c184_i32_75, %c80_i32_76, %c12_i32, %c0_i32_77) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of23(Consume, 1)
        aie.objectfifo.release @of27(Consume, 1)
        aie.objectfifo.release @of28(Produce, 1)
        %38 = aie.objectfifo.acquire @of28(Consume, 1) : memref<14x1x80xi8>
        %39 = aie.objectfifo.acquire @of29(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_78 = memref.collapse_shape %38 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_79 = memref.collapse_shape %39 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_80 = arith.constant 14 : i32
        %c80_i32_81 = arith.constant 80 : i32
        %c184_i32_82 = arith.constant 184 : i32
        %c9_i32_83 = arith.constant 9 : i32
        func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_78, %view_53, %collapse_shape_79, %c14_i32_80, %c80_i32_81, %c184_i32_82, %c9_i32_83) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of29(Produce, 1)
        %40:2 = aie.objectfifo.acquire @of23(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
        %41 = aie.objectfifo.acquire @of26(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_84 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_85 = memref.collapse_shape %41 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_86 = arith.constant 14 : i32
        %c80_i32_87 = arith.constant 80 : i32
        %c184_i32_88 = arith.constant 184 : i32
        %c9_i32_89 = arith.constant 9 : i32
        func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_84, %view, %collapse_shape_85, %c14_i32_86, %c80_i32_87, %c184_i32_88, %c9_i32_89) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of26(Produce, 1)
        %42:3 = aie.objectfifo.acquire @of26(Consume, 3) : memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>
        %43 = aie.objectfifo.acquire @of27(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_90 = memref.collapse_shape %42#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_91 = memref.collapse_shape %42#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_92 = memref.collapse_shape %42#2 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_93 = memref.collapse_shape %43 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_94 = arith.constant 14 : i32
        %c1_i32_95 = arith.constant 1 : i32
        %c184_i32_96 = arith.constant 184 : i32
        %c3_i32_97 = arith.constant 3 : i32
        %c3_i32_98 = arith.constant 3 : i32
        %c1_i32_99 = arith.constant 1 : i32
        %c7_i32_100 = arith.constant 7 : i32
        %c0_i32_101 = arith.constant 0 : i32
        func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_90, %collapse_shape_91, %collapse_shape_92, %view_51, %collapse_shape_93, %c14_i32_94, %c1_i32_95, %c184_i32_96, %c3_i32_97, %c3_i32_98, %c1_i32_99, %c7_i32_100, %c0_i32_101) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of26(Consume, 1)
        aie.objectfifo.release @of27(Produce, 1)
        %44 = aie.objectfifo.acquire @of27(Consume, 1) : memref<14x1x184xui8>
        %45 = aie.objectfifo.acquire @of28(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_102 = memref.collapse_shape %44 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_103 = memref.collapse_shape %45 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_104 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_105 = arith.constant 14 : i32
        %c184_i32_106 = arith.constant 184 : i32
        %c80_i32_107 = arith.constant 80 : i32
        %c12_i32_108 = arith.constant 12 : i32
        %c0_i32_109 = arith.constant 0 : i32
        func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_102, %view_52, %collapse_shape_103, %collapse_shape_104, %c14_i32_105, %c184_i32_106, %c80_i32_107, %c12_i32_108, %c0_i32_109) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of23(Consume, 1)
        aie.objectfifo.release @of27(Consume, 1)
        aie.objectfifo.release @of28(Produce, 1)
        %46:2 = aie.objectfifo.acquire @of28(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
        %47 = aie.objectfifo.acquire @of29(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_110 = memref.collapse_shape %46#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_111 = memref.collapse_shape %47 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_112 = arith.constant 14 : i32
        %c80_i32_113 = arith.constant 80 : i32
        %c184_i32_114 = arith.constant 184 : i32
        %c9_i32_115 = arith.constant 9 : i32
        func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_110, %view_53, %collapse_shape_111, %c14_i32_112, %c80_i32_113, %c184_i32_114, %c9_i32_115) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of29(Produce, 1)
        %48:2 = aie.objectfifo.acquire @of29(Consume, 2) : memref<14x1x184xui8>, memref<14x1x184xui8>
        %49 = aie.objectfifo.acquire @of30(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_116 = memref.collapse_shape %48#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_117 = memref.collapse_shape %48#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_118 = memref.collapse_shape %48#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_119 = memref.collapse_shape %49 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_120 = arith.constant 14 : i32
        %c1_i32_121 = arith.constant 1 : i32
        %c184_i32_122 = arith.constant 184 : i32
        %c3_i32_123 = arith.constant 3 : i32
        %c3_i32_124 = arith.constant 3 : i32
        %c0_i32_125 = arith.constant 0 : i32
        %c7_i32_126 = arith.constant 7 : i32
        %c0_i32_127 = arith.constant 0 : i32
        func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_116, %collapse_shape_117, %collapse_shape_118, %view_54, %collapse_shape_119, %c14_i32_120, %c1_i32_121, %c184_i32_122, %c3_i32_123, %c3_i32_124, %c0_i32_125, %c7_i32_126, %c0_i32_127) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of30(Produce, 1)
        %50 = aie.objectfifo.acquire @of30(Consume, 1) : memref<14x1x184xui8>
        %51 = aie.objectfifo.acquire @of25(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_128 = memref.collapse_shape %50 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_129 = memref.collapse_shape %51 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_130 = memref.collapse_shape %46#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_131 = arith.constant 14 : i32
        %c184_i32_132 = arith.constant 184 : i32
        %c80_i32_133 = arith.constant 80 : i32
        %c12_i32_134 = arith.constant 12 : i32
        %c1_i32_135 = arith.constant 1 : i32
        func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_128, %view_55, %collapse_shape_129, %collapse_shape_130, %c14_i32_131, %c184_i32_132, %c80_i32_133, %c12_i32_134, %c1_i32_135) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of30(Consume, 1)
        aie.objectfifo.release @of28(Consume, 1)
        aie.objectfifo.release @of25(Produce, 1)
        %c0_136 = arith.constant 0 : index
        %c11 = arith.constant 11 : index
        %c1_137 = arith.constant 1 : index
        scf.for %arg1 = %c0_136 to %c11 step %c1_137 {
          %68:2 = aie.objectfifo.acquire @of23(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
          %69 = aie.objectfifo.acquire @of26(Produce, 1) : memref<14x1x184xui8>
          %collapse_shape_203 = memref.collapse_shape %68#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_204 = memref.collapse_shape %69 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %c14_i32_205 = arith.constant 14 : i32
          %c80_i32_206 = arith.constant 80 : i32
          %c184_i32_207 = arith.constant 184 : i32
          %c9_i32_208 = arith.constant 9 : i32
          func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_203, %view, %collapse_shape_204, %c14_i32_205, %c80_i32_206, %c184_i32_207, %c9_i32_208) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of26(Produce, 1)
          %70:3 = aie.objectfifo.acquire @of26(Consume, 3) : memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>
          %71 = aie.objectfifo.acquire @of27(Produce, 1) : memref<14x1x184xui8>
          %collapse_shape_209 = memref.collapse_shape %70#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_210 = memref.collapse_shape %70#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_211 = memref.collapse_shape %70#2 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_212 = memref.collapse_shape %71 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %c14_i32_213 = arith.constant 14 : i32
          %c1_i32_214 = arith.constant 1 : i32
          %c184_i32_215 = arith.constant 184 : i32
          %c3_i32_216 = arith.constant 3 : i32
          %c3_i32_217 = arith.constant 3 : i32
          %c1_i32_218 = arith.constant 1 : i32
          %c7_i32_219 = arith.constant 7 : i32
          %c0_i32_220 = arith.constant 0 : i32
          func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_209, %collapse_shape_210, %collapse_shape_211, %view_51, %collapse_shape_212, %c14_i32_213, %c1_i32_214, %c184_i32_215, %c3_i32_216, %c3_i32_217, %c1_i32_218, %c7_i32_219, %c0_i32_220) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of26(Consume, 1)
          aie.objectfifo.release @of27(Produce, 1)
          %72 = aie.objectfifo.acquire @of27(Consume, 1) : memref<14x1x184xui8>
          %73 = aie.objectfifo.acquire @of28(Produce, 1) : memref<14x1x80xi8>
          %collapse_shape_221 = memref.collapse_shape %72 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_222 = memref.collapse_shape %73 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_223 = memref.collapse_shape %68#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %c14_i32_224 = arith.constant 14 : i32
          %c184_i32_225 = arith.constant 184 : i32
          %c80_i32_226 = arith.constant 80 : i32
          %c12_i32_227 = arith.constant 12 : i32
          %c0_i32_228 = arith.constant 0 : i32
          func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_221, %view_52, %collapse_shape_222, %collapse_shape_223, %c14_i32_224, %c184_i32_225, %c80_i32_226, %c12_i32_227, %c0_i32_228) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of23(Consume, 1)
          aie.objectfifo.release @of27(Consume, 1)
          aie.objectfifo.release @of28(Produce, 1)
          %74:2 = aie.objectfifo.acquire @of28(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
          %75 = aie.objectfifo.acquire @of29(Produce, 1) : memref<14x1x184xui8>
          %collapse_shape_229 = memref.collapse_shape %74#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_230 = memref.collapse_shape %75 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %c14_i32_231 = arith.constant 14 : i32
          %c80_i32_232 = arith.constant 80 : i32
          %c184_i32_233 = arith.constant 184 : i32
          %c9_i32_234 = arith.constant 9 : i32
          func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_229, %view_53, %collapse_shape_230, %c14_i32_231, %c80_i32_232, %c184_i32_233, %c9_i32_234) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of29(Produce, 1)
          %76:3 = aie.objectfifo.acquire @of29(Consume, 3) : memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>
          %77 = aie.objectfifo.acquire @of30(Produce, 1) : memref<14x1x184xui8>
          %collapse_shape_235 = memref.collapse_shape %76#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_236 = memref.collapse_shape %76#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_237 = memref.collapse_shape %76#2 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_238 = memref.collapse_shape %77 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %c14_i32_239 = arith.constant 14 : i32
          %c1_i32_240 = arith.constant 1 : i32
          %c184_i32_241 = arith.constant 184 : i32
          %c3_i32_242 = arith.constant 3 : i32
          %c3_i32_243 = arith.constant 3 : i32
          %c1_i32_244 = arith.constant 1 : i32
          %c7_i32_245 = arith.constant 7 : i32
          %c0_i32_246 = arith.constant 0 : i32
          func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_235, %collapse_shape_236, %collapse_shape_237, %view_54, %collapse_shape_238, %c14_i32_239, %c1_i32_240, %c184_i32_241, %c3_i32_242, %c3_i32_243, %c1_i32_244, %c7_i32_245, %c0_i32_246) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of29(Consume, 1)
          aie.objectfifo.release @of30(Produce, 1)
          %78 = aie.objectfifo.acquire @of30(Consume, 1) : memref<14x1x184xui8>
          %79 = aie.objectfifo.acquire @of25(Produce, 1) : memref<14x1x80xi8>
          %collapse_shape_247 = memref.collapse_shape %78 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
          %collapse_shape_248 = memref.collapse_shape %79 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_249 = memref.collapse_shape %74#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %c14_i32_250 = arith.constant 14 : i32
          %c184_i32_251 = arith.constant 184 : i32
          %c80_i32_252 = arith.constant 80 : i32
          %c12_i32_253 = arith.constant 12 : i32
          %c1_i32_254 = arith.constant 1 : i32
          func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_247, %view_55, %collapse_shape_248, %collapse_shape_249, %c14_i32_250, %c184_i32_251, %c80_i32_252, %c12_i32_253, %c1_i32_254) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of30(Consume, 1)
          aie.objectfifo.release @of28(Consume, 1)
          aie.objectfifo.release @of25(Produce, 1)
        }
        %52:2 = aie.objectfifo.acquire @of26(Consume, 2) : memref<14x1x184xui8>, memref<14x1x184xui8>
        %53 = aie.objectfifo.acquire @of27(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_138 = memref.collapse_shape %52#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_139 = memref.collapse_shape %52#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_140 = memref.collapse_shape %52#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_141 = memref.collapse_shape %53 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_142 = arith.constant 14 : i32
        %c1_i32_143 = arith.constant 1 : i32
        %c184_i32_144 = arith.constant 184 : i32
        %c3_i32_145 = arith.constant 3 : i32
        %c3_i32_146 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c7_i32_147 = arith.constant 7 : i32
        %c0_i32_148 = arith.constant 0 : i32
        func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_138, %collapse_shape_139, %collapse_shape_140, %view_51, %collapse_shape_141, %c14_i32_142, %c1_i32_143, %c184_i32_144, %c3_i32_145, %c3_i32_146, %c2_i32, %c7_i32_147, %c0_i32_148) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of26(Consume, 2)
        aie.objectfifo.release @of27(Produce, 1)
        %54 = aie.objectfifo.acquire @of23(Consume, 1) : memref<14x1x80xi8>
        %55 = aie.objectfifo.acquire @of27(Consume, 1) : memref<14x1x184xui8>
        %56 = aie.objectfifo.acquire @of28(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_149 = memref.collapse_shape %55 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_150 = memref.collapse_shape %56 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_151 = memref.collapse_shape %54 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_152 = arith.constant 14 : i32
        %c184_i32_153 = arith.constant 184 : i32
        %c80_i32_154 = arith.constant 80 : i32
        %c12_i32_155 = arith.constant 12 : i32
        %c0_i32_156 = arith.constant 0 : i32
        func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_149, %view_52, %collapse_shape_150, %collapse_shape_151, %c14_i32_152, %c184_i32_153, %c80_i32_154, %c12_i32_155, %c0_i32_156) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of23(Consume, 1)
        aie.objectfifo.release @of27(Consume, 1)
        aie.objectfifo.release @of28(Produce, 1)
        %57:2 = aie.objectfifo.acquire @of28(Consume, 2) : memref<14x1x80xi8>, memref<14x1x80xi8>
        %58 = aie.objectfifo.acquire @of29(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_157 = memref.collapse_shape %57#1 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_158 = memref.collapse_shape %58 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_159 = arith.constant 14 : i32
        %c80_i32_160 = arith.constant 80 : i32
        %c184_i32_161 = arith.constant 184 : i32
        %c9_i32_162 = arith.constant 9 : i32
        func.call @"14a4ba36_conv2dk1_relu_i8_ui8"(%collapse_shape_157, %view_53, %collapse_shape_158, %c14_i32_159, %c80_i32_160, %c184_i32_161, %c9_i32_162) : (memref<1120xi8>, memref<14720xi8>, memref<2576xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of29(Produce, 1)
        %59:3 = aie.objectfifo.acquire @of29(Consume, 3) : memref<14x1x184xui8>, memref<14x1x184xui8>, memref<14x1x184xui8>
        %60 = aie.objectfifo.acquire @of30(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_163 = memref.collapse_shape %59#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_164 = memref.collapse_shape %59#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_165 = memref.collapse_shape %59#2 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_166 = memref.collapse_shape %60 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_167 = arith.constant 14 : i32
        %c1_i32_168 = arith.constant 1 : i32
        %c184_i32_169 = arith.constant 184 : i32
        %c3_i32_170 = arith.constant 3 : i32
        %c3_i32_171 = arith.constant 3 : i32
        %c1_i32_172 = arith.constant 1 : i32
        %c7_i32_173 = arith.constant 7 : i32
        %c0_i32_174 = arith.constant 0 : i32
        func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_163, %collapse_shape_164, %collapse_shape_165, %view_54, %collapse_shape_166, %c14_i32_167, %c1_i32_168, %c184_i32_169, %c3_i32_170, %c3_i32_171, %c1_i32_172, %c7_i32_173, %c0_i32_174) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of29(Consume, 1)
        aie.objectfifo.release @of30(Produce, 1)
        %61 = aie.objectfifo.acquire @of30(Consume, 1) : memref<14x1x184xui8>
        %62 = aie.objectfifo.acquire @of25(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_175 = memref.collapse_shape %61 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_176 = memref.collapse_shape %62 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_177 = memref.collapse_shape %57#0 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_178 = arith.constant 14 : i32
        %c184_i32_179 = arith.constant 184 : i32
        %c80_i32_180 = arith.constant 80 : i32
        %c12_i32_181 = arith.constant 12 : i32
        %c1_i32_182 = arith.constant 1 : i32
        func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_175, %view_55, %collapse_shape_176, %collapse_shape_177, %c14_i32_178, %c184_i32_179, %c80_i32_180, %c12_i32_181, %c1_i32_182) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of30(Consume, 1)
        aie.objectfifo.release @of28(Consume, 1)
        aie.objectfifo.release @of25(Produce, 1)
        %63:2 = aie.objectfifo.acquire @of29(Consume, 2) : memref<14x1x184xui8>, memref<14x1x184xui8>
        %64 = aie.objectfifo.acquire @of30(Produce, 1) : memref<14x1x184xui8>
        %collapse_shape_183 = memref.collapse_shape %63#0 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_184 = memref.collapse_shape %63#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_185 = memref.collapse_shape %63#1 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_186 = memref.collapse_shape %64 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %c14_i32_187 = arith.constant 14 : i32
        %c1_i32_188 = arith.constant 1 : i32
        %c184_i32_189 = arith.constant 184 : i32
        %c3_i32_190 = arith.constant 3 : i32
        %c3_i32_191 = arith.constant 3 : i32
        %c2_i32_192 = arith.constant 2 : i32
        %c7_i32_193 = arith.constant 7 : i32
        %c0_i32_194 = arith.constant 0 : i32
        func.call @a74d4d9a_conv2dk3_dw_stride1_relu_ui8_ui8(%collapse_shape_183, %collapse_shape_184, %collapse_shape_185, %view_54, %collapse_shape_186, %c14_i32_187, %c1_i32_188, %c184_i32_189, %c3_i32_190, %c3_i32_191, %c2_i32_192, %c7_i32_193, %c0_i32_194) : (memref<2576xui8>, memref<2576xui8>, memref<2576xui8>, memref<1656xi8>, memref<2576xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of29(Consume, 2)
        aie.objectfifo.release @of30(Produce, 1)
        %65 = aie.objectfifo.acquire @of28(Consume, 1) : memref<14x1x80xi8>
        %66 = aie.objectfifo.acquire @of30(Consume, 1) : memref<14x1x184xui8>
        %67 = aie.objectfifo.acquire @of25(Produce, 1) : memref<14x1x80xi8>
        %collapse_shape_195 = memref.collapse_shape %66 [[0, 1, 2]] : memref<14x1x184xui8> into memref<2576xui8>
        %collapse_shape_196 = memref.collapse_shape %67 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %collapse_shape_197 = memref.collapse_shape %65 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
        %c14_i32_198 = arith.constant 14 : i32
        %c184_i32_199 = arith.constant 184 : i32
        %c80_i32_200 = arith.constant 80 : i32
        %c12_i32_201 = arith.constant 12 : i32
        %c1_i32_202 = arith.constant 1 : i32
        func.call @a7eba513_conv2dk1_skip_ui8_i8_i8(%collapse_shape_195, %view_55, %collapse_shape_196, %collapse_shape_197, %c14_i32_198, %c184_i32_199, %c80_i32_200, %c12_i32_201, %c1_i32_202) : (memref<2576xui8>, memref<14720xi8>, memref<1120xi8>, memref<1120xi8>, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of30(Consume, 1)
        aie.objectfifo.release @of28(Consume, 1)
        aie.objectfifo.release @of25(Produce, 1)
      }
      aie.end
    }
    %9 = aie.core(%logical_core_8) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c14 step %c1_51 {
          %32 = aie.objectfifo.acquire @of25(Consume, 1) : memref<14x1x80xi8>
          %33 = aie.objectfifo.acquire @of31(Produce, 1) : memref<14x1x480xui8>
          %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<14x1x80xi8> into memref<1120xi8>
          %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
          %c14_i32 = arith.constant 14 : i32
          %c80_i32 = arith.constant 80 : i32
          %c480_i32 = arith.constant 480 : i32
          %c8_i32 = arith.constant 8 : i32
          func.call @"62fc3ee5_conv2dk1_relu_i8_ui8"(%collapse_shape, %buf_9, %collapse_shape_52, %c14_i32, %c80_i32, %c480_i32, %c8_i32) : (memref<1120xi8>, memref<38400xi8>, memref<6720xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of25(Consume, 1)
          aie.objectfifo.release @of31(Produce, 1)
        }
      }
      aie.end
    }
    %10 = aie.core(%logical_core_9) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32:2 = aie.objectfifo.acquire @of31(Consume, 2) : memref<14x1x480xui8>, memref<14x1x480xui8>
        %33 = aie.objectfifo.acquire @of32(Produce, 1) : memref<14x1x480xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %collapse_shape_50 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %collapse_shape_51 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c480_i32 = arith.constant 480 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_53 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32 = arith.constant 7 : i32
        %c0_i32_54 = arith.constant 0 : i32
        func.call @"61f9b950_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape, %collapse_shape_50, %collapse_shape_51, %buf_10, %collapse_shape_52, %c14_i32, %c1_i32, %c480_i32, %c3_i32, %c3_i32_53, %c0_i32, %c7_i32, %c0_i32_54) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<4320xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of32(Produce, 1)
        %c0_55 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %c12 step %c1_56 {
          %36:3 = aie.objectfifo.acquire @of31(Consume, 3) : memref<14x1x480xui8>, memref<14x1x480xui8>, memref<14x1x480xui8>
          %37 = aie.objectfifo.acquire @of32(Produce, 1) : memref<14x1x480xui8>
          %collapse_shape_68 = memref.collapse_shape %36#0 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
          %collapse_shape_69 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
          %collapse_shape_70 = memref.collapse_shape %36#2 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
          %collapse_shape_71 = memref.collapse_shape %37 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
          %c14_i32_72 = arith.constant 14 : i32
          %c1_i32_73 = arith.constant 1 : i32
          %c480_i32_74 = arith.constant 480 : i32
          %c3_i32_75 = arith.constant 3 : i32
          %c3_i32_76 = arith.constant 3 : i32
          %c1_i32_77 = arith.constant 1 : i32
          %c7_i32_78 = arith.constant 7 : i32
          %c0_i32_79 = arith.constant 0 : i32
          func.call @"61f9b950_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_68, %collapse_shape_69, %collapse_shape_70, %buf_10, %collapse_shape_71, %c14_i32_72, %c1_i32_73, %c480_i32_74, %c3_i32_75, %c3_i32_76, %c1_i32_77, %c7_i32_78, %c0_i32_79) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<4320xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of31(Consume, 1)
          aie.objectfifo.release @of32(Produce, 1)
        }
        %34:2 = aie.objectfifo.acquire @of31(Consume, 2) : memref<14x1x480xui8>, memref<14x1x480xui8>
        %35 = aie.objectfifo.acquire @of32(Produce, 1) : memref<14x1x480xui8>
        %collapse_shape_57 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %collapse_shape_58 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %collapse_shape_59 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %collapse_shape_60 = memref.collapse_shape %35 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
        %c14_i32_61 = arith.constant 14 : i32
        %c1_i32_62 = arith.constant 1 : i32
        %c480_i32_63 = arith.constant 480 : i32
        %c3_i32_64 = arith.constant 3 : i32
        %c3_i32_65 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c7_i32_66 = arith.constant 7 : i32
        %c0_i32_67 = arith.constant 0 : i32
        func.call @"61f9b950_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_57, %collapse_shape_58, %collapse_shape_59, %buf_10, %collapse_shape_60, %c14_i32_61, %c1_i32_62, %c480_i32_63, %c3_i32_64, %c3_i32_65, %c2_i32, %c7_i32_66, %c0_i32_67) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<4320xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of31(Consume, 2)
        aie.objectfifo.release @of32(Produce, 1)
      }
      aie.end
    }
    %11 = aie.core(%logical_core_10) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c14 step %c1_51 {
          %32 = aie.objectfifo.acquire @of32(Consume, 1) : memref<14x1x480xui8>
          %33 = aie.objectfifo.acquire @of33(Produce, 1) : memref<14x1x112xi8>
          %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<14x1x480xui8> into memref<6720xui8>
          %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x112xi8> into memref<1568xi8>
          %c14_i32 = arith.constant 14 : i32
          %c480_i32 = arith.constant 480 : i32
          %c112_i32 = arith.constant 112 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @c261f057_conv2dk1_ui8_i8(%collapse_shape, %buf_11, %collapse_shape_52, %c14_i32, %c480_i32, %c112_i32, %c9_i32) : (memref<6720xui8>, memref<53760xi8>, memref<1568xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of32(Consume, 1)
          aie.objectfifo.release @of33(Produce, 1)
        }
      }
      aie.end
    }
    %12 = aie.core(%logical_core_11) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c14 step %c1_51 {
          %32 = aie.objectfifo.acquire @of33(Consume, 1) : memref<14x1x112xi8>
          %33 = aie.objectfifo.acquire @of34(Produce, 1) : memref<14x1x336xui8>
          %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<14x1x112xi8> into memref<1568xi8>
          %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %c14_i32 = arith.constant 14 : i32
          %c112_i32 = arith.constant 112 : i32
          %c336_i32 = arith.constant 336 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @"9c08375a_conv2dk1_relu_i8_ui8"(%collapse_shape, %buf_12, %collapse_shape_52, %c14_i32, %c112_i32, %c336_i32, %c9_i32) : (memref<1568xi8>, memref<37632xi8>, memref<4704xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of33(Consume, 1)
          aie.objectfifo.release @of34(Produce, 1)
        }
      }
      aie.end
    }
    %13 = aie.core(%logical_core_12) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32:2 = aie.objectfifo.acquire @of34(Consume, 2) : memref<14x1x336xui8>, memref<14x1x336xui8>
        %33 = aie.objectfifo.acquire @of35(Produce, 1) : memref<14x1x336xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_50 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_51 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c336_i32 = arith.constant 336 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_53 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_54 = arith.constant 0 : i32
        func.call @"1461e5eb_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape, %collapse_shape_50, %collapse_shape_51, %buf_13, %collapse_shape_52, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_53, %c0_i32, %c8_i32, %c0_i32_54) : (memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<4704xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of35(Produce, 1)
        %c0_55 = arith.constant 0 : index
        %c12 = arith.constant 12 : index
        %c1_56 = arith.constant 1 : index
        scf.for %arg1 = %c0_55 to %c12 step %c1_56 {
          %36:3 = aie.objectfifo.acquire @of34(Consume, 3) : memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>
          %37 = aie.objectfifo.acquire @of35(Produce, 1) : memref<14x1x336xui8>
          %collapse_shape_68 = memref.collapse_shape %36#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_69 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_70 = memref.collapse_shape %36#2 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_71 = memref.collapse_shape %37 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %c14_i32_72 = arith.constant 14 : i32
          %c1_i32_73 = arith.constant 1 : i32
          %c336_i32_74 = arith.constant 336 : i32
          %c3_i32_75 = arith.constant 3 : i32
          %c3_i32_76 = arith.constant 3 : i32
          %c1_i32_77 = arith.constant 1 : i32
          %c8_i32_78 = arith.constant 8 : i32
          %c0_i32_79 = arith.constant 0 : i32
          func.call @"1461e5eb_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_68, %collapse_shape_69, %collapse_shape_70, %buf_13, %collapse_shape_71, %c14_i32_72, %c1_i32_73, %c336_i32_74, %c3_i32_75, %c3_i32_76, %c1_i32_77, %c8_i32_78, %c0_i32_79) : (memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<4704xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of34(Consume, 1)
          aie.objectfifo.release @of35(Produce, 1)
        }
        %34:2 = aie.objectfifo.acquire @of34(Consume, 2) : memref<14x1x336xui8>, memref<14x1x336xui8>
        %35 = aie.objectfifo.acquire @of35(Produce, 1) : memref<14x1x336xui8>
        %collapse_shape_57 = memref.collapse_shape %34#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_58 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_59 = memref.collapse_shape %34#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_60 = memref.collapse_shape %35 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %c14_i32_61 = arith.constant 14 : i32
        %c1_i32_62 = arith.constant 1 : i32
        %c336_i32_63 = arith.constant 336 : i32
        %c3_i32_64 = arith.constant 3 : i32
        %c3_i32_65 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c8_i32_66 = arith.constant 8 : i32
        %c0_i32_67 = arith.constant 0 : i32
        func.call @"1461e5eb_conv2dk3_dw_stride1_relu_ui8_ui8"(%collapse_shape_57, %collapse_shape_58, %collapse_shape_59, %buf_13, %collapse_shape_60, %c14_i32_61, %c1_i32_62, %c336_i32_63, %c3_i32_64, %c3_i32_65, %c2_i32, %c8_i32_66, %c0_i32_67) : (memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<4704xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of34(Consume, 2)
        aie.objectfifo.release @of35(Produce, 1)
      }
      aie.end
    }
    %14 = aie.core(%logical_core_13) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c14 step %c1_51 {
          %32 = aie.objectfifo.acquire @of35(Consume, 1) : memref<14x1x336xui8>
          %33 = aie.objectfifo.acquire @of36(Produce, 1) : memref<14x1x112xi8>
          %34 = aie.objectfifo.acquire @of33_fwd(Consume, 1) : memref<14x1x112xi8>
          %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x112xi8> into memref<1568xi8>
          %collapse_shape_53 = memref.collapse_shape %34 [[0, 1, 2]] : memref<14x1x112xi8> into memref<1568xi8>
          %c14_i32 = arith.constant 14 : i32
          %c336_i32 = arith.constant 336 : i32
          %c112_i32 = arith.constant 112 : i32
          %c12_i32 = arith.constant 12 : i32
          %c1_i32 = arith.constant 1 : i32
          func.call @"0de09bc1_conv2dk1_skip_ui8_i8_i8"(%collapse_shape, %buf_14, %collapse_shape_52, %collapse_shape_53, %c14_i32, %c336_i32, %c112_i32, %c12_i32, %c1_i32) : (memref<4704xui8>, memref<37632xi8>, memref<1568xi8>, memref<1568xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of35(Consume, 1)
          aie.objectfifo.release @of36(Produce, 1)
          aie.objectfifo.release @of33_fwd(Consume, 1)
        }
      }
      aie.end
    }
    %15 = aie.core(%logical_core_14) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c14 = arith.constant 14 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c14 step %c1_51 {
          %32 = aie.objectfifo.acquire @of36(Consume, 1) : memref<14x1x112xi8>
          %33 = aie.objectfifo.acquire @of37(Produce, 1) : memref<14x1x336xui8>
          %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<14x1x112xi8> into memref<1568xi8>
          %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %c14_i32 = arith.constant 14 : i32
          %c112_i32 = arith.constant 112 : i32
          %c336_i32 = arith.constant 336 : i32
          %c8_i32 = arith.constant 8 : i32
          func.call @"9c08375a_conv2dk1_relu_i8_ui8"(%collapse_shape, %buf_15, %collapse_shape_52, %c14_i32, %c112_i32, %c336_i32, %c8_i32) : (memref<1568xi8>, memref<37632xi8>, memref<4704xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of36(Consume, 1)
          aie.objectfifo.release @of37(Produce, 1)
        }
      }
      aie.end
    }
    %16 = aie.core(%logical_core_15) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %view = memref.view %buf_16[%c0_50][] : memref<29952xi8> to memref<3024xi8>
        %c3072 = arith.constant 3072 : index
        %view_51 = memref.view %buf_16[%c3072][] : memref<29952xi8> to memref<26880xi8>
        %32:2 = aie.objectfifo.acquire @of37(Consume, 2) : memref<14x1x336xui8>, memref<14x1x336xui8>
        %33 = aie.objectfifo.acquire @of38(Produce, 1) : memref<7x1x336xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_52 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_53 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x336xui8> into memref<2352xui8>
        %c14_i32 = arith.constant 14 : i32
        %c1_i32 = arith.constant 1 : i32
        %c336_i32 = arith.constant 336 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_55 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_56 = arith.constant 0 : i32
        func.call @"2aac7224_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape, %collapse_shape_52, %collapse_shape_53, %view, %collapse_shape_54, %c14_i32, %c1_i32, %c336_i32, %c3_i32, %c3_i32_55, %c0_i32, %c8_i32, %c0_i32_56) : (memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<2352xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of37(Consume, 1)
        aie.objectfifo.release @of38(Produce, 1)
        %34 = aie.objectfifo.acquire @of38(Consume, 1) : memref<7x1x336xui8>
        %35 = aie.objectfifo.acquire @of39(Produce, 1) : memref<7x1x80xi8>
        %collapse_shape_57 = memref.collapse_shape %34 [[0, 1, 2]] : memref<7x1x336xui8> into memref<2352xui8>
        %collapse_shape_58 = memref.collapse_shape %35 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
        %c7_i32 = arith.constant 7 : i32
        %c336_i32_59 = arith.constant 336 : i32
        %c80_i32 = arith.constant 80 : i32
        %c9_i32 = arith.constant 9 : i32
        func.call @"6a5d1c89_conv2dk1_ui8_i8"(%collapse_shape_57, %view_51, %collapse_shape_58, %c7_i32, %c336_i32_59, %c80_i32, %c9_i32) : (memref<2352xui8>, memref<26880xi8>, memref<560xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of38(Consume, 1)
        aie.objectfifo.release @of39(Produce, 1)
        %c0_60 = arith.constant 0 : index
        %c5 = arith.constant 5 : index
        %c1_61 = arith.constant 1 : index
        scf.for %arg1 = %c0_60 to %c5 step %c1_61 {
          %40:3 = aie.objectfifo.acquire @of37(Consume, 3) : memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>
          %41 = aie.objectfifo.acquire @of38(Produce, 1) : memref<7x1x336xui8>
          %collapse_shape_80 = memref.collapse_shape %40#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_81 = memref.collapse_shape %40#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_82 = memref.collapse_shape %40#2 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
          %collapse_shape_83 = memref.collapse_shape %41 [[0, 1, 2]] : memref<7x1x336xui8> into memref<2352xui8>
          %c14_i32_84 = arith.constant 14 : i32
          %c1_i32_85 = arith.constant 1 : i32
          %c336_i32_86 = arith.constant 336 : i32
          %c3_i32_87 = arith.constant 3 : i32
          %c3_i32_88 = arith.constant 3 : i32
          %c1_i32_89 = arith.constant 1 : i32
          %c8_i32_90 = arith.constant 8 : i32
          %c0_i32_91 = arith.constant 0 : i32
          func.call @"2aac7224_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape_80, %collapse_shape_81, %collapse_shape_82, %view, %collapse_shape_83, %c14_i32_84, %c1_i32_85, %c336_i32_86, %c3_i32_87, %c3_i32_88, %c1_i32_89, %c8_i32_90, %c0_i32_91) : (memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<2352xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of37(Consume, 2)
          aie.objectfifo.release @of38(Produce, 1)
          %42 = aie.objectfifo.acquire @of38(Consume, 1) : memref<7x1x336xui8>
          %43 = aie.objectfifo.acquire @of39(Produce, 1) : memref<7x1x80xi8>
          %collapse_shape_92 = memref.collapse_shape %42 [[0, 1, 2]] : memref<7x1x336xui8> into memref<2352xui8>
          %collapse_shape_93 = memref.collapse_shape %43 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
          %c7_i32_94 = arith.constant 7 : i32
          %c336_i32_95 = arith.constant 336 : i32
          %c80_i32_96 = arith.constant 80 : i32
          %c9_i32_97 = arith.constant 9 : i32
          func.call @"6a5d1c89_conv2dk1_ui8_i8"(%collapse_shape_92, %view_51, %collapse_shape_93, %c7_i32_94, %c336_i32_95, %c80_i32_96, %c9_i32_97) : (memref<2352xui8>, memref<26880xi8>, memref<560xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of38(Consume, 1)
          aie.objectfifo.release @of39(Produce, 1)
        }
        %36:3 = aie.objectfifo.acquire @of37(Consume, 3) : memref<14x1x336xui8>, memref<14x1x336xui8>, memref<14x1x336xui8>
        %37 = aie.objectfifo.acquire @of38(Produce, 1) : memref<7x1x336xui8>
        %collapse_shape_62 = memref.collapse_shape %36#0 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_63 = memref.collapse_shape %36#1 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_64 = memref.collapse_shape %36#2 [[0, 1, 2]] : memref<14x1x336xui8> into memref<4704xui8>
        %collapse_shape_65 = memref.collapse_shape %37 [[0, 1, 2]] : memref<7x1x336xui8> into memref<2352xui8>
        %c14_i32_66 = arith.constant 14 : i32
        %c1_i32_67 = arith.constant 1 : i32
        %c336_i32_68 = arith.constant 336 : i32
        %c3_i32_69 = arith.constant 3 : i32
        %c3_i32_70 = arith.constant 3 : i32
        %c1_i32_71 = arith.constant 1 : i32
        %c8_i32_72 = arith.constant 8 : i32
        %c0_i32_73 = arith.constant 0 : i32
        func.call @"2aac7224_conv2dk3_dw_stride2_relu_ui8_ui8"(%collapse_shape_62, %collapse_shape_63, %collapse_shape_64, %view, %collapse_shape_65, %c14_i32_66, %c1_i32_67, %c336_i32_68, %c3_i32_69, %c3_i32_70, %c1_i32_71, %c8_i32_72, %c0_i32_73) : (memref<4704xui8>, memref<4704xui8>, memref<4704xui8>, memref<3024xi8>, memref<2352xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of37(Consume, 3)
        aie.objectfifo.release @of38(Produce, 1)
        %38 = aie.objectfifo.acquire @of38(Consume, 1) : memref<7x1x336xui8>
        %39 = aie.objectfifo.acquire @of39(Produce, 1) : memref<7x1x80xi8>
        %collapse_shape_74 = memref.collapse_shape %38 [[0, 1, 2]] : memref<7x1x336xui8> into memref<2352xui8>
        %collapse_shape_75 = memref.collapse_shape %39 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
        %c7_i32_76 = arith.constant 7 : i32
        %c336_i32_77 = arith.constant 336 : i32
        %c80_i32_78 = arith.constant 80 : i32
        %c9_i32_79 = arith.constant 9 : i32
        func.call @"6a5d1c89_conv2dk1_ui8_i8"(%collapse_shape_74, %view_51, %collapse_shape_75, %c7_i32_76, %c336_i32_77, %c80_i32_78, %c9_i32_79) : (memref<2352xui8>, memref<26880xi8>, memref<560xi8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of38(Consume, 1)
        aie.objectfifo.release @of39(Produce, 1)
      }
      aie.end
    }
    %17 = aie.core(%logical_core_16) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of39(Consume, 1) : memref<7x1x80xi8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %33 = aie.objectfifo.acquire @bn13_l1_put_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %c7_i32 = arith.constant 7 : i32
            %c80_i32 = arith.constant 80 : i32
            %c960_i32 = arith.constant 960 : i32
            %c2_i32 = arith.constant 2 : i32
            %34 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_54 = arith.constant 0 : i32
            func.call @e10f2832_bn13_1_conv2dk1_i8_ui8_partial_width_put_new(%collapse_shape, %33, %c7_i32, %c80_i32, %c960_i32, %c2_i32, %34, %c0_i32, %c0_i32_54) : (memref<560xi8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn13_l1_put_wts(Consume, 1)
          }
          aie.objectfifo.release @of39(Consume, 1)
        }
      }
      aie.end
    }
    %18 = aie.core(%logical_core_17) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of39(Consume, 1) : memref<7x1x80xi8>
          %33 = aie.objectfifo.acquire @of40(Produce, 1) : memref<7x1x960xui8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %34 = aie.objectfifo.acquire @bn13_l1_get_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
            %c7_i32 = arith.constant 7 : i32
            %c80_i32 = arith.constant 80 : i32
            %c960_i32 = arith.constant 960 : i32
            %c9_i32 = arith.constant 9 : i32
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_55 = arith.constant 2 : i32
            %35 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_56 = arith.constant 0 : i32
            func.call @ef8b7952_bn13_1_conv2dk1_i8_ui8_partial_width_get_new(%collapse_shape, %34, %collapse_shape_54, %c7_i32, %c80_i32, %c960_i32, %c9_i32, %c2_i32, %c2_i32_55, %35, %c0_i32, %c0_i32_56) : (memref<560xi8>, memref<19200xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn13_l1_get_wts(Consume, 1)
          }
          aie.objectfifo.release @of39(Consume, 1)
          aie.objectfifo.release @of40(Produce, 1)
        }
      }
      aie.end
    }
    %19 = aie.core(%logical_core_18) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32:2 = aie.objectfifo.acquire @of40(Consume, 2) : memref<7x1x960xui8>, memref<7x1x960xui8>
        %33 = aie.objectfifo.acquire @of41(Produce, 1) : memref<7x1x480xui8>
        %34 = aie.objectfifo.acquire @of42(Produce, 1) : memref<7x1x480xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_50 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_51 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %collapse_shape_53 = memref.collapse_shape %34 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %c7_i32 = arith.constant 7 : i32
        %c1_i32 = arith.constant 1 : i32
        %c960_i32 = arith.constant 960 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_54 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_55 = arith.constant 0 : i32
        func.call @db3aae29_bn13_conv2dk3_ui8_out_split(%collapse_shape, %collapse_shape_50, %collapse_shape_51, %bn13_2_wts_static, %collapse_shape_52, %collapse_shape_53, %c7_i32, %c1_i32, %c960_i32, %c3_i32, %c3_i32_54, %c0_i32, %c8_i32, %c0_i32_55) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of41(Produce, 1)
        aie.objectfifo.release @of42(Produce, 1)
        %c0_56 = arith.constant 0 : index
        %c5 = arith.constant 5 : index
        %c1_57 = arith.constant 1 : index
        scf.for %arg1 = %c0_56 to %c5 step %c1_57 {
          %38:3 = aie.objectfifo.acquire @of40(Consume, 3) : memref<7x1x960xui8>, memref<7x1x960xui8>, memref<7x1x960xui8>
          %39 = aie.objectfifo.acquire @of41(Produce, 1) : memref<7x1x480xui8>
          %40 = aie.objectfifo.acquire @of42(Produce, 1) : memref<7x1x480xui8>
          %collapse_shape_70 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
          %collapse_shape_71 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
          %collapse_shape_72 = memref.collapse_shape %38#2 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
          %collapse_shape_73 = memref.collapse_shape %39 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
          %collapse_shape_74 = memref.collapse_shape %40 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
          %c7_i32_75 = arith.constant 7 : i32
          %c1_i32_76 = arith.constant 1 : i32
          %c960_i32_77 = arith.constant 960 : i32
          %c3_i32_78 = arith.constant 3 : i32
          %c3_i32_79 = arith.constant 3 : i32
          %c1_i32_80 = arith.constant 1 : i32
          %c8_i32_81 = arith.constant 8 : i32
          %c0_i32_82 = arith.constant 0 : i32
          func.call @db3aae29_bn13_conv2dk3_ui8_out_split(%collapse_shape_70, %collapse_shape_71, %collapse_shape_72, %bn13_2_wts_static, %collapse_shape_73, %collapse_shape_74, %c7_i32_75, %c1_i32_76, %c960_i32_77, %c3_i32_78, %c3_i32_79, %c1_i32_80, %c8_i32_81, %c0_i32_82) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of41(Produce, 1)
          aie.objectfifo.release @of42(Produce, 1)
          aie.objectfifo.release @of40(Consume, 1)
        }
        %35:2 = aie.objectfifo.acquire @of40(Consume, 2) : memref<7x1x960xui8>, memref<7x1x960xui8>
        %36 = aie.objectfifo.acquire @of41(Produce, 1) : memref<7x1x480xui8>
        %37 = aie.objectfifo.acquire @of42(Produce, 1) : memref<7x1x480xui8>
        %collapse_shape_58 = memref.collapse_shape %35#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_59 = memref.collapse_shape %35#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_60 = memref.collapse_shape %35#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_61 = memref.collapse_shape %36 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %collapse_shape_62 = memref.collapse_shape %37 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %c7_i32_63 = arith.constant 7 : i32
        %c1_i32_64 = arith.constant 1 : i32
        %c960_i32_65 = arith.constant 960 : i32
        %c3_i32_66 = arith.constant 3 : i32
        %c3_i32_67 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c8_i32_68 = arith.constant 8 : i32
        %c0_i32_69 = arith.constant 0 : i32
        func.call @db3aae29_bn13_conv2dk3_ui8_out_split(%collapse_shape_58, %collapse_shape_59, %collapse_shape_60, %bn13_2_wts_static, %collapse_shape_61, %collapse_shape_62, %c7_i32_63, %c1_i32_64, %c960_i32_65, %c3_i32_66, %c3_i32_67, %c2_i32, %c8_i32_68, %c0_i32_69) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of41(Produce, 1)
        aie.objectfifo.release @of42(Produce, 1)
        aie.objectfifo.release @of40(Consume, 2)
      }
      aie.end
    }
    %20 = aie.core(%logical_core_19) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of41(Consume, 1) : memref<7x1x480xui8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %33 = aie.objectfifo.acquire @bn13_l3_put_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
            %c7_i32 = arith.constant 7 : i32
            %c960_i32 = arith.constant 960 : i32
            %c80_i32 = arith.constant 80 : i32
            %c2_i32 = arith.constant 2 : i32
            %34 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_54 = arith.constant 0 : i32
            func.call @"612b8218_bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new"(%collapse_shape, %33, %c7_i32, %c960_i32, %c80_i32, %c2_i32, %34, %c0_i32, %c0_i32_54) : (memref<3360xui8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn13_l3_put_wts(Consume, 1)
          }
          aie.objectfifo.release @of41(Consume, 1)
        }
      }
      aie.end
    }
    %21 = aie.core(%logical_core_20) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of42(Consume, 1) : memref<7x1x480xui8>
          %33 = aie.objectfifo.acquire @of43(Produce, 1) : memref<7x1x80xi8>
          %34 = aie.objectfifo.acquire @of39_fwd(Consume, 1) : memref<7x1x80xi8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %35 = aie.objectfifo.acquire @bn13_l3_get_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
            %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %collapse_shape_55 = memref.collapse_shape %34 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %c7_i32 = arith.constant 7 : i32
            %c960_i32 = arith.constant 960 : i32
            %c80_i32 = arith.constant 80 : i32
            %c12_i32 = arith.constant 12 : i32
            %c0_i32 = arith.constant 0 : i32
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_56 = arith.constant 2 : i32
            %36 = arith.index_cast %arg2 : index to i32
            %c0_i32_57 = arith.constant 0 : i32
            %c0_i32_58 = arith.constant 0 : i32
            func.call @f8d47821_bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new(%collapse_shape, %35, %collapse_shape_54, %collapse_shape_55, %c7_i32, %c960_i32, %c80_i32, %c12_i32, %c0_i32, %c2_i32, %c2_i32_56, %36, %c0_i32_57, %c0_i32_58) : (memref<3360xui8>, memref<19200xi8>, memref<560xi8>, memref<560xi8>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn13_l3_get_wts(Consume, 1)
          }
          aie.objectfifo.release @of42(Consume, 1)
          aie.objectfifo.release @of43(Produce, 1)
          aie.objectfifo.release @of39_fwd(Consume, 1)
        }
      }
      aie.end
    }
    %22 = aie.core(%logical_core_21) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of43(Consume, 1) : memref<7x1x80xi8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %33 = aie.objectfifo.acquire @bn14_l1_put_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %c7_i32 = arith.constant 7 : i32
            %c80_i32 = arith.constant 80 : i32
            %c960_i32 = arith.constant 960 : i32
            %c2_i32 = arith.constant 2 : i32
            %34 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_54 = arith.constant 0 : i32
            func.call @d244edb6_bn14_1_conv2dk1_i8_ui8_partial_width_put_new(%collapse_shape, %33, %c7_i32, %c80_i32, %c960_i32, %c2_i32, %34, %c0_i32, %c0_i32_54) : (memref<560xi8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn14_l1_put_wts(Consume, 1)
          }
          aie.objectfifo.release @of43(Consume, 1)
        }
      }
      aie.end
    }
    %23 = aie.core(%logical_core_22) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of43(Consume, 1) : memref<7x1x80xi8>
          %33 = aie.objectfifo.acquire @of44(Produce, 1) : memref<7x1x960xui8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %34 = aie.objectfifo.acquire @bn14_l1_get_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
            %c7_i32 = arith.constant 7 : i32
            %c80_i32 = arith.constant 80 : i32
            %c960_i32 = arith.constant 960 : i32
            %c9_i32 = arith.constant 9 : i32
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_55 = arith.constant 2 : i32
            %35 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_56 = arith.constant 0 : i32
            func.call @"7a24c72f_bn14_1_conv2dk1_i8_ui8_partial_width_get_new"(%collapse_shape, %34, %collapse_shape_54, %c7_i32, %c80_i32, %c960_i32, %c9_i32, %c2_i32, %c2_i32_55, %35, %c0_i32, %c0_i32_56) : (memref<560xi8>, memref<19200xi8>, memref<6720xui8>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn14_l1_get_wts(Consume, 1)
          }
          aie.objectfifo.release @of43(Consume, 1)
          aie.objectfifo.release @of44(Produce, 1)
        }
      }
      aie.end
    }
    %24 = aie.core(%logical_core_23) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32:2 = aie.objectfifo.acquire @of44(Consume, 2) : memref<7x1x960xui8>, memref<7x1x960xui8>
        %33 = aie.objectfifo.acquire @of45(Produce, 1) : memref<7x1x480xui8>
        %34 = aie.objectfifo.acquire @of46(Produce, 1) : memref<7x1x480xui8>
        %collapse_shape = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_50 = memref.collapse_shape %32#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_51 = memref.collapse_shape %32#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_52 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %collapse_shape_53 = memref.collapse_shape %34 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %c7_i32 = arith.constant 7 : i32
        %c1_i32 = arith.constant 1 : i32
        %c960_i32 = arith.constant 960 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_54 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c7_i32_55 = arith.constant 7 : i32
        %c0_i32_56 = arith.constant 0 : i32
        func.call @a1fb6ef1_bn14_conv2dk3_ui8_out_split(%collapse_shape, %collapse_shape_50, %collapse_shape_51, %bn14_2_wts_static, %collapse_shape_52, %collapse_shape_53, %c7_i32, %c1_i32, %c960_i32, %c3_i32, %c3_i32_54, %c0_i32, %c7_i32_55, %c0_i32_56) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of45(Produce, 1)
        aie.objectfifo.release @of46(Produce, 1)
        %c0_57 = arith.constant 0 : index
        %c5 = arith.constant 5 : index
        %c1_58 = arith.constant 1 : index
        scf.for %arg1 = %c0_57 to %c5 step %c1_58 {
          %38:3 = aie.objectfifo.acquire @of44(Consume, 3) : memref<7x1x960xui8>, memref<7x1x960xui8>, memref<7x1x960xui8>
          %39 = aie.objectfifo.acquire @of45(Produce, 1) : memref<7x1x480xui8>
          %40 = aie.objectfifo.acquire @of46(Produce, 1) : memref<7x1x480xui8>
          %collapse_shape_71 = memref.collapse_shape %38#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
          %collapse_shape_72 = memref.collapse_shape %38#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
          %collapse_shape_73 = memref.collapse_shape %38#2 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
          %collapse_shape_74 = memref.collapse_shape %39 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
          %collapse_shape_75 = memref.collapse_shape %40 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
          %c7_i32_76 = arith.constant 7 : i32
          %c1_i32_77 = arith.constant 1 : i32
          %c960_i32_78 = arith.constant 960 : i32
          %c3_i32_79 = arith.constant 3 : i32
          %c3_i32_80 = arith.constant 3 : i32
          %c1_i32_81 = arith.constant 1 : i32
          %c7_i32_82 = arith.constant 7 : i32
          %c0_i32_83 = arith.constant 0 : i32
          func.call @a1fb6ef1_bn14_conv2dk3_ui8_out_split(%collapse_shape_71, %collapse_shape_72, %collapse_shape_73, %bn14_2_wts_static, %collapse_shape_74, %collapse_shape_75, %c7_i32_76, %c1_i32_77, %c960_i32_78, %c3_i32_79, %c3_i32_80, %c1_i32_81, %c7_i32_82, %c0_i32_83) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @of45(Produce, 1)
          aie.objectfifo.release @of46(Produce, 1)
          aie.objectfifo.release @of44(Consume, 1)
        }
        %35:2 = aie.objectfifo.acquire @of44(Consume, 2) : memref<7x1x960xui8>, memref<7x1x960xui8>
        %36 = aie.objectfifo.acquire @of45(Produce, 1) : memref<7x1x480xui8>
        %37 = aie.objectfifo.acquire @of46(Produce, 1) : memref<7x1x480xui8>
        %collapse_shape_59 = memref.collapse_shape %35#0 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_60 = memref.collapse_shape %35#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_61 = memref.collapse_shape %35#1 [[0, 1, 2]] : memref<7x1x960xui8> into memref<6720xui8>
        %collapse_shape_62 = memref.collapse_shape %36 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %collapse_shape_63 = memref.collapse_shape %37 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
        %c7_i32_64 = arith.constant 7 : i32
        %c1_i32_65 = arith.constant 1 : i32
        %c960_i32_66 = arith.constant 960 : i32
        %c3_i32_67 = arith.constant 3 : i32
        %c3_i32_68 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c7_i32_69 = arith.constant 7 : i32
        %c0_i32_70 = arith.constant 0 : i32
        func.call @a1fb6ef1_bn14_conv2dk3_ui8_out_split(%collapse_shape_59, %collapse_shape_60, %collapse_shape_61, %bn14_2_wts_static, %collapse_shape_62, %collapse_shape_63, %c7_i32_64, %c1_i32_65, %c960_i32_66, %c3_i32_67, %c3_i32_68, %c2_i32, %c7_i32_69, %c0_i32_70) : (memref<6720xui8>, memref<6720xui8>, memref<6720xui8>, memref<8640xi8>, memref<3360xui8>, memref<3360xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @of45(Produce, 1)
        aie.objectfifo.release @of46(Produce, 1)
        aie.objectfifo.release @of44(Consume, 2)
      }
      aie.end
    }
    %25 = aie.core(%logical_core_24) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of45(Consume, 1) : memref<7x1x480xui8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %33 = aie.objectfifo.acquire @bn14_l3_put_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
            %c7_i32 = arith.constant 7 : i32
            %c960_i32 = arith.constant 960 : i32
            %c80_i32 = arith.constant 80 : i32
            %c2_i32 = arith.constant 2 : i32
            %34 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_54 = arith.constant 0 : i32
            func.call @aa85880e_bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new(%collapse_shape, %33, %c7_i32, %c960_i32, %c80_i32, %c2_i32, %34, %c0_i32, %c0_i32_54) : (memref<3360xui8>, memref<19200xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn14_l3_put_wts(Consume, 1)
          }
          aie.objectfifo.release @of45(Consume, 1)
        }
      }
      aie.end
    }
    %26 = aie.core(%logical_core_25) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %32 = aie.objectfifo.acquire @of46(Consume, 1) : memref<7x1x480xui8>
          %33 = aie.objectfifo.acquire @of47(Produce, 1) : memref<7x1x80xi8>
          %34 = aie.objectfifo.acquire @of43_fwd(Consume, 1) : memref<7x1x80xi8>
          %c0_52 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c2 step %c1_53 {
            %35 = aie.objectfifo.acquire @bn14_l3_get_wts(Consume, 1) : memref<19200xi8>
            %collapse_shape = memref.collapse_shape %32 [[0, 1, 2]] : memref<7x1x480xui8> into memref<3360xui8>
            %collapse_shape_54 = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %collapse_shape_55 = memref.collapse_shape %34 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %c7_i32 = arith.constant 7 : i32
            %c960_i32 = arith.constant 960 : i32
            %c80_i32 = arith.constant 80 : i32
            %c13_i32 = arith.constant 13 : i32
            %c1_i32 = arith.constant 1 : i32
            %c2_i32 = arith.constant 2 : i32
            %c2_i32_56 = arith.constant 2 : i32
            %36 = arith.index_cast %arg2 : index to i32
            %c0_i32 = arith.constant 0 : i32
            %c0_i32_57 = arith.constant 0 : i32
            func.call @"9336eff4_bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new"(%collapse_shape, %35, %collapse_shape_54, %collapse_shape_55, %c7_i32, %c960_i32, %c80_i32, %c13_i32, %c1_i32, %c2_i32, %c2_i32_56, %36, %c0_i32, %c0_i32_57) : (memref<3360xui8>, memref<19200xi8>, memref<560xi8>, memref<560xi8>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @bn14_l3_get_wts(Consume, 1)
          }
          aie.objectfifo.release @of46(Consume, 1)
          aie.objectfifo.release @of47(Produce, 1)
          aie.objectfifo.release @of43_fwd(Consume, 1)
        }
      }
      aie.end
    }
    %27 = aie.core(%logical_core_26) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32 = aie.objectfifo.acquire @of48(Produce, 1) : memref<1280xui16>
        %c0_50 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c7 step %c1_51 {
          %33 = aie.objectfifo.acquire @of47(Consume, 1) : memref<7x1x80xi8>
          %c0_52 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_53 = arith.constant 1 : index
          scf.for %arg2 = %c0_52 to %c8 step %c1_53 {
            %34 = aie.objectfifo.acquire @post_L1_wts(Consume, 1) : memref<9600xi8>
            %collapse_shape = memref.collapse_shape %33 [[0, 1, 2]] : memref<7x1x80xi8> into memref<560xi8>
            %c7_i32 = arith.constant 7 : i32
            %c80_i32 = arith.constant 80 : i32
            %c960_i32 = arith.constant 960 : i32
            %c1280_i32 = arith.constant 1280 : i32
            %c8_i32 = arith.constant 8 : i32
            %35 = arith.index_cast %arg1 : index to i32
            %c8_i32_54 = arith.constant 8 : i32
            %36 = arith.index_cast %arg2 : index to i32
            func.call @"613bef8c_conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8"(%collapse_shape, %34, %32, %c7_i32, %c80_i32, %c960_i32, %c1280_i32, %c8_i32, %35, %c8_i32_54, %36) : (memref<560xi8>, memref<9600xi8>, memref<1280xui16>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            aie.objectfifo.release @post_L1_wts(Consume, 1)
          }
          aie.objectfifo.release @of47(Consume, 1)
        }
        aie.objectfifo.release @of48(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %28 = aie.core(%logical_core_27) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_50 = arith.constant 0 : index
        %c40 = arith.constant 40 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c40 step %c1_51 {
          %34 = aie.objectfifo.acquire @of50_join0(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_1(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c960_i32 = arith.constant 960 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%32, %35, %34, %c1_i32, %c960_i32, %c1280_i32, %c8_i32, %c9_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_1(Consume, 1)
          aie.objectfifo.release @of50_join0(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
        %33 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_52 = arith.constant 0 : index
        %c40_53 = arith.constant 40 : index
        %c1_54 = arith.constant 1 : index
        scf.for %arg1 = %c0_52 to %c40_53 step %c1_54 {
          %34 = aie.objectfifo.acquire @of50_join0(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_1(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c1280_i32_55 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c11_i32 = arith.constant 11 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%33, %35, %34, %c1_i32, %c1280_i32, %c1280_i32_55, %c8_i32, %c11_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_1(Consume, 1)
          aie.objectfifo.release @of50_join0(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
      }
      aie.end
    }
    %29 = aie.core(%logical_core_28) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_50 = arith.constant 0 : index
        %c40 = arith.constant 40 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c40 step %c1_51 {
          %34 = aie.objectfifo.acquire @of50_join1(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_2(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c960_i32 = arith.constant 960 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%32, %35, %34, %c1_i32, %c960_i32, %c1280_i32, %c8_i32, %c9_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_2(Consume, 1)
          aie.objectfifo.release @of50_join1(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
        %33 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_52 = arith.constant 0 : index
        %c40_53 = arith.constant 40 : index
        %c1_54 = arith.constant 1 : index
        scf.for %arg1 = %c0_52 to %c40_53 step %c1_54 {
          %34 = aie.objectfifo.acquire @of50_join1(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_2(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c1280_i32_55 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c11_i32 = arith.constant 11 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%33, %35, %34, %c1_i32, %c1280_i32, %c1280_i32_55, %c8_i32, %c11_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_2(Consume, 1)
          aie.objectfifo.release @of50_join1(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
      }
      aie.end
    }
    %30 = aie.core(%logical_core_29) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_50 = arith.constant 0 : index
        %c40 = arith.constant 40 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c40 step %c1_51 {
          %34 = aie.objectfifo.acquire @of50_join2(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_3(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c960_i32 = arith.constant 960 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%32, %35, %34, %c1_i32, %c960_i32, %c1280_i32, %c8_i32, %c9_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_3(Consume, 1)
          aie.objectfifo.release @of50_join2(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
        %33 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_52 = arith.constant 0 : index
        %c40_53 = arith.constant 40 : index
        %c1_54 = arith.constant 1 : index
        scf.for %arg1 = %c0_52 to %c40_53 step %c1_54 {
          %34 = aie.objectfifo.acquire @of50_join2(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_3(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c1280_i32_55 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c11_i32 = arith.constant 11 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%33, %35, %34, %c1_i32, %c1280_i32, %c1280_i32_55, %c8_i32, %c11_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_3(Consume, 1)
          aie.objectfifo.release @of50_join2(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
      }
      aie.end
    }
    %31 = aie.core(%logical_core_30) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %32 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_50 = arith.constant 0 : index
        %c40 = arith.constant 40 : index
        %c1_51 = arith.constant 1 : index
        scf.for %arg1 = %c0_50 to %c40 step %c1_51 {
          %34 = aie.objectfifo.acquire @of50_join3(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_4(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c960_i32 = arith.constant 960 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c9_i32 = arith.constant 9 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%32, %35, %34, %c1_i32, %c960_i32, %c1280_i32, %c8_i32, %c9_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_4(Consume, 1)
          aie.objectfifo.release @of50_join3(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
        %33 = aie.objectfifo.acquire @of49(Consume, 1) : memref<1280xui16>
        %c0_52 = arith.constant 0 : index
        %c40_53 = arith.constant 40 : index
        %c1_54 = arith.constant 1 : index
        scf.for %arg1 = %c0_52 to %c40_53 step %c1_54 {
          %34 = aie.objectfifo.acquire @of50_join3(Produce, 1) : memref<8xui16>
          %35 = aie.objectfifo.acquire @post_L2_wts_4(Consume, 1) : memref<10240xi8>
          %c1_i32 = arith.constant 1 : i32
          %c1280_i32 = arith.constant 1280 : i32
          %c1280_i32_55 = arith.constant 1280 : i32
          %c8_i32 = arith.constant 8 : i32
          %c11_i32 = arith.constant 11 : i32
          func.call @c7b77a1e_post_L2_conv2dk1_relu_i16_ui16_pad(%33, %35, %34, %c1_i32, %c1280_i32, %c1280_i32_55, %c8_i32, %c11_i32) : (memref<1280xui16>, memref<10240xi8>, memref<8xui16>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @post_L2_wts_4(Consume, 1)
          aie.objectfifo.release @of50_join3(Produce, 1)
        }
        aie.objectfifo.release @of49(Consume, 1)
      }
      aie.end
    }
    aie.cascade_flow(%logical_core_16, %logical_core_17)
    aie.cascade_flow(%logical_core_19, %logical_core_20)
    aie.cascade_flow(%logical_core_21, %logical_core_22)
    aie.cascade_flow(%logical_core_24, %logical_core_25)
    aie.runtime_sequence(%arg0: memref<100352xi32>, %arg1: memref<1280xi32>, %arg2: memref<640xi32>) {
      %32 = aiex.dma_configure_task_for @of0 {
        aie.dma_bd(%arg0 : memref<100352xi32> offset = 0 len = 100352 sizes = [1, 1, 1, 100352] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @of48 {
        aie.dma_bd(%arg1 : memref<1280xi32> offset = 0 len = 640 sizes = [1, 1, 1, 640] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%33)
      aiex.dma_await_task(%33)
      aiex.dma_free_task(%32)
      aiex.dma_free_task(%33)
      %34 = aiex.dma_configure_task_for @of49 {
        aie.dma_bd(%arg1 : memref<1280xi32> offset = 0 len = 640 sizes = [1, 1, 1, 640] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @of50 {
        aie.dma_bd(%arg1 : memref<1280xi32> offset = 640 len = 640 sizes = [1, 1, 1, 640] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%35)
      aiex.dma_await_task(%35)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      %36 = aiex.dma_configure_task_for @of49 {
        aie.dma_bd(%arg1 : memref<1280xi32> offset = 640 len = 640 sizes = [1, 1, 1, 640] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @of50 {
        aie.dma_bd(%arg2 : memref<640xi32> offset = 0 len = 640 sizes = [1, 1, 1, 640] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%37)
      aiex.dma_await_task(%37)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%37)
    }
  }
}
