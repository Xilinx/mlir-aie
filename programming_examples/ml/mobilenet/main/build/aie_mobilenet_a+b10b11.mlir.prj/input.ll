; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@rtp23 = external global [16 x i32]
@rtp22 = external global [16 x i32]
@rtp13 = external global [16 x i32]
@rtp12 = external global [16 x i32]
@rtp14 = external global [16 x i32]
@rtp15 = external global [16 x i32]
@rtp05 = external global [16 x i32]
@rtp04 = external global [16 x i32]
@rtp03 = external global [16 x i32]
@bn11_3_rtp = external global [16 x i32]
@bn11_2_rtp = external global [16 x i32]
@bn11_1_rtp = external global [16 x i32]
@bn10_3_rtp = external global [16 x i32]
@bn10_2_rtp = external global [16 x i32]
@bn10_1_rtp = external global [16 x i32]
@wts_b10_L3L2_cons_buff_0 = external global [96480 x i8]
@weightsInBN10_layer1_cons_buff_0 = external global [38400 x i8]
@weightsInBN10_layer2_cons_buff_0 = external global [4320 x i8]
@weightsInBN10_layer3_cons_buff_0 = external global [53760 x i8]
@wts_b11_L3L2_cons_buff_0 = external global [78288 x i8]
@weightsInBN11_layer1_cons_buff_0 = external global [37632 x i8]
@weightsInBN11_layer2_cons_buff_0 = external global [3024 x i8]
@weightsInBN11_layer3_cons_buff_0 = external global [37632 x i8]
@act_in_cons_buff_2 = external global [112 x [1 x [16 x i8]]]
@act_in_cons_buff_1 = external global [112 x [1 x [16 x i8]]]
@act_in_cons_buff_0 = external global [112 x [1 x [16 x i8]]]
@wts_OF_01_L3L2_cons_buff_0 = external global [34256 x i8]
@bn0_1_wts_OF_L2L1_cons_buff_0 = external global [3536 x i8]
@bn2_wts_OF_L2L1_cons_buff_0 = external global [4104 x i8]
@bn3_wts_OF_L2L1_cons_buff_0 = external global [5256 x i8]
@bn4_wts_OF_L2L1_cons_buff_0 = external global [10680 x i8]
@bn5_wts_OF_L2L1_cons_buff_0 = external global [10680 x i8]
@wts_OF_11_L3L2_cons_buff_0 = external global [126952 x i8]
@bn6_wts_OF_L2L1_cons_buff_0 = external global [30960 x i8]
@bn7_wts_OF_L2L1_cons_buff_0 = external global [33800 x i8]
@bn8_wts_OF_L2L1_cons_buff_0 = external global [31096 x i8]
@bn9_wts_OF_L2L1_cons_buff_0 = external global [31096 x i8]
@act_bn01_bn2_buff_2 = external global [56 x [1 x [24 x i8]]]
@act_bn01_bn2_buff_1 = external global [56 x [1 x [24 x i8]]]
@act_bn01_bn2_buff_0 = external global [56 x [1 x [24 x i8]]]
@bn01_act_bn0_2_3_buff_0 = external global [112 x [1 x [16 x i8]]]
@bn01_act_bn0_bn1_buff_0 = external global [112 x [1 x [16 x i8]]]
@bn01_act_bn1_1_2_buff_2 = external global [112 x [1 x [64 x i8]]]
@bn01_act_bn1_1_2_buff_1 = external global [112 x [1 x [64 x i8]]]
@bn01_act_bn1_1_2_buff_0 = external global [112 x [1 x [64 x i8]]]
@bn01_act_bn1_2_3_buff_0 = external global [56 x [1 x [64 x i8]]]
@act_bn2_bn3_buff_2 = external global [56 x [1 x [24 x i8]]]
@act_bn2_bn3_buff_1 = external global [56 x [1 x [24 x i8]]]
@act_bn2_bn3_buff_0 = external global [56 x [1 x [24 x i8]]]
@bn2_act_1_2_buff_2 = external global [56 x [1 x [72 x i8]]]
@bn2_act_1_2_buff_1 = external global [56 x [1 x [72 x i8]]]
@bn2_act_1_2_buff_0 = external global [56 x [1 x [72 x i8]]]
@bn2_act_2_3_buff_0 = external global [56 x [1 x [72 x i8]]]
@act_bn3_bn4_buff_2 = external global [28 x [1 x [40 x i8]]]
@act_bn3_bn4_buff_1 = external global [28 x [1 x [40 x i8]]]
@act_bn3_bn4_buff_0 = external global [28 x [1 x [40 x i8]]]
@bn3_act_1_2_buff_2 = external global [56 x [1 x [72 x i8]]]
@bn3_act_1_2_buff_1 = external global [56 x [1 x [72 x i8]]]
@bn3_act_1_2_buff_0 = external global [56 x [1 x [72 x i8]]]
@bn3_act_2_3_buff_0 = external global [28 x [1 x [72 x i8]]]
@act_bn4_bn5_buff_2 = external global [28 x [1 x [40 x i8]]]
@act_bn4_bn5_buff_1 = external global [28 x [1 x [40 x i8]]]
@act_bn4_bn5_buff_0 = external global [28 x [1 x [40 x i8]]]
@bn4_act_1_2_buff_2 = external global [28 x [1 x [120 x i8]]]
@bn4_act_1_2_buff_1 = external global [28 x [1 x [120 x i8]]]
@bn4_act_1_2_buff_0 = external global [28 x [1 x [120 x i8]]]
@bn4_act_2_3_buff_0 = external global [28 x [1 x [120 x i8]]]
@act_bn5_bn6_buff_1 = external global [28 x [1 x [40 x i8]]]
@act_bn5_bn6_buff_0 = external global [28 x [1 x [40 x i8]]]
@act_bn5_bn6_cons_buff_2 = external global [28 x [1 x [40 x i8]]]
@act_bn5_bn6_cons_buff_1 = external global [28 x [1 x [40 x i8]]]
@act_bn5_bn6_cons_buff_0 = external global [28 x [1 x [40 x i8]]]
@bn5_act_1_2_buff_2 = external global [28 x [1 x [120 x i8]]]
@bn5_act_1_2_buff_1 = external global [28 x [1 x [120 x i8]]]
@bn5_act_1_2_buff_0 = external global [28 x [1 x [120 x i8]]]
@bn5_act_2_3_buff_0 = external global [28 x [1 x [120 x i8]]]
@act_bn6_bn7_buff_1 = external global [14 x [1 x [80 x i8]]]
@act_bn6_bn7_buff_0 = external global [14 x [1 x [80 x i8]]]
@bn6_act_1_2_buff_2 = external global [28 x [1 x [240 x i8]]]
@bn6_act_1_2_buff_1 = external global [28 x [1 x [240 x i8]]]
@bn6_act_1_2_buff_0 = external global [28 x [1 x [240 x i8]]]
@bn6_act_2_3_buff_0 = external global [14 x [1 x [240 x i8]]]
@act_bn7_bn8_buff_1 = external global [14 x [1 x [80 x i8]]]
@act_bn7_bn8_buff_0 = external global [14 x [1 x [80 x i8]]]
@act_bn7_bn8_cons_buff_2 = external global [14 x [1 x [80 x i8]]]
@act_bn7_bn8_cons_buff_1 = external global [14 x [1 x [80 x i8]]]
@act_bn7_bn8_cons_buff_0 = external global [14 x [1 x [80 x i8]]]
@bn7_act_1_2_buff_2 = external global [14 x [1 x [200 x i8]]]
@bn7_act_1_2_buff_1 = external global [14 x [1 x [200 x i8]]]
@bn7_act_1_2_buff_0 = external global [14 x [1 x [200 x i8]]]
@bn7_act_2_3_buff_0 = external global [14 x [1 x [200 x i8]]]
@act_bn8_bn9_buff_1 = external global [14 x [1 x [80 x i8]]]
@act_bn8_bn9_buff_0 = external global [14 x [1 x [80 x i8]]]
@bn8_act_1_2_buff_2 = external global [14 x [1 x [184 x i8]]]
@bn8_act_1_2_buff_1 = external global [14 x [1 x [184 x i8]]]
@bn8_act_1_2_buff_0 = external global [14 x [1 x [184 x i8]]]
@bn8_act_2_3_buff_0 = external global [14 x [1 x [184 x i8]]]
@act_bn9_bn10_buff_1 = external global [14 x [1 x [80 x i8]]]
@act_bn9_bn10_buff_0 = external global [14 x [1 x [80 x i8]]]
@bn9_act_1_2_buff_2 = external global [14 x [1 x [184 x i8]]]
@bn9_act_1_2_buff_1 = external global [14 x [1 x [184 x i8]]]
@bn9_act_1_2_buff_0 = external global [14 x [1 x [184 x i8]]]
@bn9_act_2_3_buff_0 = external global [14 x [1 x [184 x i8]]]
@act_out_buff_1 = external global [14 x [1 x [112 x i8]]]
@act_out_buff_0 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_act_layer1_layer2_buff_1 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2_buff_0 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2_cons_buff_3 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2_cons_buff_2 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2_cons_buff_1 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2_cons_buff_0 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer2_layer3_buff_1 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer2_layer3_buff_0 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_buff_1 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_buff_0 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons_buff_5 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons_buff_4 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons_buff_3 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons_buff_2 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons_buff_1 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons_buff_0 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0 = external global [14 x [1 x [112 x i8]]]
@OF_b11_skip_cons_buff_1 = external global [14 x [1 x [112 x i8]]]
@OF_b11_skip_cons_buff_0 = external global [14 x [1 x [112 x i8]]]
@B_OF_b11_act_layer1_layer2_buff_1 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2_buff_0 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2_cons_buff_3 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2_cons_buff_2 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2_cons_buff_1 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2_cons_buff_0 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer2_layer3_buff_1 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer2_layer3_buff_0 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer2_layer3 = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2_cons = external global [14 x [1 x [336 x i8]]]
@B_OF_b11_act_layer1_layer2 = external global [14 x [1 x [336 x i8]]]
@OF_b11_skip_cons = external global [14 x [1 x [112 x i8]]]
@OF_b11_skip = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_0_cons = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1_1_cons = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_layer3_bn_11_layer1 = external global [14 x [1 x [112 x i8]]]
@B_OF_b10_act_layer2_layer3 = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2_cons = external global [14 x [1 x [480 x i8]]]
@B_OF_b10_act_layer1_layer2 = external global [14 x [1 x [480 x i8]]]
@act_out_cons = external global [14 x [1 x [112 x i8]]]
@act_out = external global [14 x [1 x [112 x i8]]]
@bn9_act_2_3 = external global [14 x [1 x [184 x i8]]]
@bn9_act_1_2 = external global [14 x [1 x [184 x i8]]]
@act_bn9_bn10 = external global [14 x [1 x [80 x i8]]]
@bn8_act_2_3 = external global [14 x [1 x [184 x i8]]]
@bn8_act_1_2 = external global [14 x [1 x [184 x i8]]]
@act_bn8_bn9 = external global [14 x [1 x [80 x i8]]]
@bn7_act_2_3 = external global [14 x [1 x [200 x i8]]]
@bn7_act_1_2 = external global [14 x [1 x [200 x i8]]]
@act_bn7_bn8_cons = external global [14 x [1 x [80 x i8]]]
@act_bn7_bn8 = external global [14 x [1 x [80 x i8]]]
@bn6_act_2_3 = external global [14 x [1 x [240 x i8]]]
@bn6_act_1_2 = external global [28 x [1 x [240 x i8]]]
@act_bn6_bn7 = external global [14 x [1 x [80 x i8]]]
@bn5_act_2_3 = external global [28 x [1 x [120 x i8]]]
@bn5_act_1_2 = external global [28 x [1 x [120 x i8]]]
@act_bn5_bn6_cons = external global [28 x [1 x [40 x i8]]]
@act_bn5_bn6 = external global [28 x [1 x [40 x i8]]]
@bn4_act_2_3 = external global [28 x [1 x [120 x i8]]]
@bn4_act_1_2 = external global [28 x [1 x [120 x i8]]]
@act_bn4_bn5 = external global [28 x [1 x [40 x i8]]]
@bn3_act_2_3 = external global [28 x [1 x [72 x i8]]]
@bn3_act_1_2 = external global [56 x [1 x [72 x i8]]]
@act_bn3_bn4 = external global [28 x [1 x [40 x i8]]]
@bn2_act_2_3 = external global [56 x [1 x [72 x i8]]]
@bn2_act_1_2 = external global [56 x [1 x [72 x i8]]]
@act_bn2_bn3 = external global [56 x [1 x [24 x i8]]]
@bn01_act_bn1_2_3 = external global [56 x [1 x [64 x i8]]]
@bn01_act_bn1_1_2 = external global [112 x [1 x [64 x i8]]]
@bn01_act_bn0_bn1 = external global [112 x [1 x [16 x i8]]]
@bn01_act_bn0_2_3 = external global [112 x [1 x [16 x i8]]]
@act_bn01_bn2 = external global [56 x [1 x [24 x i8]]]
@bn9_wts_OF_L2L1_cons = external global [31096 x i8]
@bn9_wts_OF_L2L1 = external global [31096 x i8]
@bn8_wts_OF_L2L1_cons = external global [31096 x i8]
@bn8_wts_OF_L2L1 = external global [31096 x i8]
@bn7_wts_OF_L2L1_cons = external global [33800 x i8]
@bn7_wts_OF_L2L1 = external global [33800 x i8]
@bn6_wts_OF_L2L1_cons = external global [30960 x i8]
@bn6_wts_OF_L2L1 = external global [30960 x i8]
@wts_OF_11_L3L2_cons = external global [126952 x i8]
@wts_OF_11_L3L2 = external global [126952 x i8]
@bn5_wts_OF_L2L1_cons = external global [10680 x i8]
@bn5_wts_OF_L2L1 = external global [10680 x i8]
@bn4_wts_OF_L2L1_cons = external global [10680 x i8]
@bn4_wts_OF_L2L1 = external global [10680 x i8]
@bn3_wts_OF_L2L1_cons = external global [5256 x i8]
@bn3_wts_OF_L2L1 = external global [5256 x i8]
@bn2_wts_OF_L2L1_cons = external global [4104 x i8]
@bn2_wts_OF_L2L1 = external global [4104 x i8]
@bn0_1_wts_OF_L2L1_cons = external global [3536 x i8]
@bn0_1_wts_OF_L2L1 = external global [3536 x i8]
@wts_OF_01_L3L2_cons = external global [34256 x i8]
@wts_OF_01_L3L2 = external global [34256 x i8]
@act_in_cons = external global [112 x [1 x [16 x i8]]]
@act_in = external global [112 x [1 x [16 x i8]]]
@weightsInBN11_layer3_cons = external global [37632 x i8]
@weightsInBN11_layer3 = external global [37632 x i8]
@weightsInBN11_layer2_cons = external global [3024 x i8]
@weightsInBN11_layer2 = external global [3024 x i8]
@weightsInBN11_layer1_cons = external global [37632 x i8]
@weightsInBN11_layer1 = external global [37632 x i8]
@wts_b11_L3L2_cons = external global [78288 x i8]
@wts_b11_L3L2 = external global [78288 x i8]
@weightsInBN10_layer3_cons = external global [53760 x i8]
@weightsInBN10_layer3 = external global [53760 x i8]
@weightsInBN10_layer2_cons = external global [4320 x i8]
@weightsInBN10_layer2 = external global [4320 x i8]
@weightsInBN10_layer1_cons = external global [38400 x i8]
@weightsInBN10_layer1 = external global [38400 x i8]
@wts_b10_L3L2_cons = external global [96480 x i8]
@wts_b10_L3L2 = external global [96480 x i8]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn1_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn1_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn2_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn2_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn2_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn2_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn3_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn3_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn3_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn3_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn4_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn4_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn4_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn4_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn5_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn5_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn5_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn5_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn6_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn6_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn6_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn6_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn7_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn7_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn7_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn7_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn8_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn8_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn8_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn8_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn9_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn9_conv2dk3_dw_stride2_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn9_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @bn9_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn10_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn10_conv2dk1_ui8_i8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn11_conv2dk1_relu_i8_ui8(ptr, ptr, ptr, i32, i32, i32, i32)

declare void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32)

declare void @bn11_conv2dk1_skip_ui8_i8_i8(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

define void @sequence(ptr %0, ptr %1, ptr %2) {
  ret void
}

define void @core_3_2() {
  br label %1

1:                                                ; preds = %24, %0
  %2 = phi i64 [ %25, %24 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %26

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %23, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 14
  br i1 %7, label %8, label %24

8:                                                ; preds = %5
  call void @llvm.aie2.acquire(i32 37, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  %9 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @OF_b11_skip_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  %13 = and i64 ptrtoint (ptr @act_out_buff_0 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @weightsInBN11_layer3_cons_buff_0 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  call void @bn11_conv2dk1_skip_ui8_i8_i8(ptr @B_OF_b11_act_layer2_layer3_buff_0, ptr @weightsInBN11_layer3_cons_buff_0, ptr @act_out_buff_0, ptr @OF_b11_skip_cons_buff_0, i32 14, i32 336, i32 112, i32 12, i32 1)
  call void @llvm.aie2.release(i32 36, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.acquire(i32 37, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  %17 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  %19 = and i64 ptrtoint (ptr @OF_b11_skip_cons_buff_1 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  %21 = and i64 ptrtoint (ptr @act_out_buff_1 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %16)
  call void @bn11_conv2dk1_skip_ui8_i8_i8(ptr @B_OF_b11_act_layer2_layer3_buff_1, ptr @weightsInBN11_layer3_cons_buff_0, ptr @act_out_buff_1, ptr @OF_b11_skip_cons_buff_1, i32 14, i32 336, i32 112, i32 12, i32 1)
  call void @llvm.aie2.release(i32 36, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 52, i32 1)
  %23 = add i64 %6, 2
  br label %5

24:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  %25 = add i64 %2, 1
  br label %1

26:                                               ; preds = %1
  ret void
}

define void @core_3_3() {
  br label %1

1:                                                ; preds = %48, %0
  %2 = phi i64 [ %49, %48 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775804
  br i1 %3, label %4, label %50

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %5 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_0 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_0 to i64), 31
  %8 = icmp eq i64 %7, 0
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  %9 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_1 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @weightsInBN11_layer2_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %13

13:                                               ; preds = %16, %4
  %14 = phi i64 [ %23, %16 ], [ 0, %4 ]
  %15 = icmp slt i64 %14, 12
  br i1 %15, label %16, label %24

16:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %17 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  %19 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_2 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %20)
  %21 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_3 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %23 = add i64 %14, 4
  br label %13

24:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_1 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  %27 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_2 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %28)
  %29 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_3 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %31

31:                                               ; preds = %34, %24
  %32 = phi i64 [ %35, %34 ], [ 0, %24 ]
  %33 = icmp slt i64 %32, 12
  br i1 %33, label %34, label %36

34:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %35 = add i64 %32, 4
  br label %31

36:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %41, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 12
  br i1 %39, label %40, label %42

40:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %41 = add i64 %38, 4
  br label %37

42:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %43

43:                                               ; preds = %46, %42
  %44 = phi i64 [ %47, %46 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, 12
  br i1 %45, label %46, label %48

46:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %47 = add i64 %44, 4
  br label %43

48:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  %49 = add i64 %2, 4
  br label %1

50:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %51 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_0 to i64), 31
  %52 = icmp eq i64 %51, 0
  call void @llvm.assume(i1 %52)
  %53 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_0 to i64), 31
  %54 = icmp eq i64 %53, 0
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  %55 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_1 to i64), 31
  %56 = icmp eq i64 %55, 0
  call void @llvm.assume(i1 %56)
  %57 = and i64 ptrtoint (ptr @weightsInBN11_layer2_cons_buff_0 to i64), 31
  %58 = icmp eq i64 %57, 0
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %59

59:                                               ; preds = %62, %50
  %60 = phi i64 [ %69, %62 ], [ 0, %50 ]
  %61 = icmp slt i64 %60, 12
  br i1 %61, label %62, label %70

62:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %63 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_1 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  %65 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_2 to i64), 31
  %66 = icmp eq i64 %65, 0
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %66)
  %67 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_3 to i64), 31
  %68 = icmp eq i64 %67, 0
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %69 = add i64 %60, 4
  br label %59

70:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %71 = and i64 ptrtoint (ptr @B_OF_b11_act_layer2_layer3_buff_1 to i64), 31
  %72 = icmp eq i64 %71, 0
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  %73 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_2 to i64), 31
  %74 = icmp eq i64 %73, 0
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %74)
  %75 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_cons_buff_3 to i64), 31
  %76 = icmp eq i64 %75, 0
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %77

77:                                               ; preds = %80, %70
  %78 = phi i64 [ %81, %80 ], [ 0, %70 ]
  %79 = icmp slt i64 %78, 12
  br i1 %79, label %80, label %82

80:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %81 = add i64 %78, 4
  br label %77

82:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 0, i32 8, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %83

83:                                               ; preds = %86, %82
  %84 = phi i64 [ %87, %86 ], [ 0, %82 ]
  %85 = icmp slt i64 %84, 12
  br i1 %85, label %86, label %88

86:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_2, ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_3, ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_0, i32 14, i32 1, i32 336, i32 3, i32 3, i32 1, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %87 = add i64 %84, 4
  br label %83

88:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @bn11_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b11_act_layer1_layer2_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @B_OF_b11_act_layer1_layer2_cons_buff_1, ptr @weightsInBN11_layer2_cons_buff_0, ptr @B_OF_b11_act_layer2_layer3_buff_1, i32 14, i32 1, i32 336, i32 3, i32 3, i32 2, i32 8, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_3_4() {
  br label %1

1:                                                ; preds = %20, %0
  %2 = phi i64 [ %21, %20 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %22

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %19, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 14
  br i1 %7, label %8, label %20

8:                                                ; preds = %5
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %9 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  %13 = and i64 ptrtoint (ptr @weightsInBN11_layer1_cons_buff_0 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  call void @bn11_conv2dk1_relu_i8_ui8(ptr @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_0, ptr @weightsInBN11_layer1_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_buff_0, i32 14, i32 112, i32 336, i32 9)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %15 = and i64 ptrtoint (ptr @B_OF_b11_act_layer1_layer2_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = and i64 ptrtoint (ptr @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %14)
  call void @bn11_conv2dk1_relu_i8_ui8(ptr @B_OF_b10_layer3_bn_11_layer1_0_cons_buff_1, ptr @weightsInBN11_layer1_cons_buff_0, ptr @B_OF_b11_act_layer1_layer2_buff_1, i32 14, i32 112, i32 336, i32 9)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %19 = add i64 %6, 2
  br label %5

20:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  %21 = add i64 %2, 1
  br label %1

22:                                               ; preds = %1
  ret void
}

define void @core_3_5() {
  br label %1

1:                                                ; preds = %20, %0
  %2 = phi i64 [ %21, %20 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %22

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %19, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 14
  br i1 %7, label %8, label %20

8:                                                ; preds = %5
  call void @llvm.aie2.acquire(i32 21, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  %9 = and i64 ptrtoint (ptr @B_OF_b10_layer3_bn_11_layer1_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  %13 = and i64 ptrtoint (ptr @weightsInBN10_layer3_cons_buff_0 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  call void @bn10_conv2dk1_ui8_i8(ptr @B_OF_b10_act_layer2_layer3_buff_0, ptr @weightsInBN10_layer3_cons_buff_0, ptr @B_OF_b10_layer3_bn_11_layer1_buff_0, i32 14, i32 480, i32 112, i32 10)
  call void @llvm.aie2.release(i32 20, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 21, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  %15 = and i64 ptrtoint (ptr @B_OF_b10_layer3_bn_11_layer1_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %14)
  call void @bn10_conv2dk1_ui8_i8(ptr @B_OF_b10_act_layer2_layer3_buff_1, ptr @weightsInBN10_layer3_cons_buff_0, ptr @B_OF_b10_layer3_bn_11_layer1_buff_1, i32 14, i32 480, i32 112, i32 10)
  call void @llvm.aie2.release(i32 20, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %19 = add i64 %6, 2
  br label %5

20:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  %21 = add i64 %2, 1
  br label %1

22:                                               ; preds = %1
  ret void
}

define void @core_2_5() {
  br label %1

1:                                                ; preds = %48, %0
  %2 = phi i64 [ %49, %48 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775804
  br i1 %3, label %4, label %50

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %5 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_0 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_0 to i64), 31
  %8 = icmp eq i64 %7, 0
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  %9 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_1 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @weightsInBN10_layer2_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %13

13:                                               ; preds = %16, %4
  %14 = phi i64 [ %23, %16 ], [ 0, %4 ]
  %15 = icmp slt i64 %14, 12
  br i1 %15, label %16, label %24

16:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %17 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  %19 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_2 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %20)
  %21 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_3 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %23 = add i64 %14, 4
  br label %13

24:                                               ; preds = %13
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_1 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  %27 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_2 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %28)
  %29 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_3 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %31

31:                                               ; preds = %34, %24
  %32 = phi i64 [ %35, %34 ], [ 0, %24 ]
  %33 = icmp slt i64 %32, 12
  br i1 %33, label %34, label %36

34:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %35 = add i64 %32, 4
  br label %31

36:                                               ; preds = %31
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %37

37:                                               ; preds = %40, %36
  %38 = phi i64 [ %41, %40 ], [ 0, %36 ]
  %39 = icmp slt i64 %38, 12
  br i1 %39, label %40, label %42

40:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %41 = add i64 %38, 4
  br label %37

42:                                               ; preds = %37
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %43

43:                                               ; preds = %46, %42
  %44 = phi i64 [ %47, %46 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, 12
  br i1 %45, label %46, label %48

46:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %8)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %6)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %47 = add i64 %44, 4
  br label %43

48:                                               ; preds = %43
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %12)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  %49 = add i64 %2, 4
  br label %1

50:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %51 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_0 to i64), 31
  %52 = icmp eq i64 %51, 0
  call void @llvm.assume(i1 %52)
  %53 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_0 to i64), 31
  %54 = icmp eq i64 %53, 0
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  %55 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_1 to i64), 31
  %56 = icmp eq i64 %55, 0
  call void @llvm.assume(i1 %56)
  %57 = and i64 ptrtoint (ptr @weightsInBN10_layer2_cons_buff_0 to i64), 31
  %58 = icmp eq i64 %57, 0
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %59

59:                                               ; preds = %62, %50
  %60 = phi i64 [ %69, %62 ], [ 0, %50 ]
  %61 = icmp slt i64 %60, 12
  br i1 %61, label %62, label %70

62:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %63 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_1 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  %65 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_2 to i64), 31
  %66 = icmp eq i64 %65, 0
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %66)
  %67 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_3 to i64), 31
  %68 = icmp eq i64 %67, 0
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %68)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %69 = add i64 %60, 4
  br label %59

70:                                               ; preds = %59
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %71 = and i64 ptrtoint (ptr @B_OF_b10_act_layer2_layer3_buff_1 to i64), 31
  %72 = icmp eq i64 %71, 0
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  %73 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_2 to i64), 31
  %74 = icmp eq i64 %73, 0
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %74)
  %75 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_cons_buff_3 to i64), 31
  %76 = icmp eq i64 %75, 0
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %77

77:                                               ; preds = %80, %70
  %78 = phi i64 [ %81, %80 ], [ 0, %70 ]
  %79 = icmp slt i64 %78, 12
  br i1 %79, label %80, label %82

80:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %81 = add i64 %78, 4
  br label %77

82:                                               ; preds = %77
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 0, i32 7, i32 0)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %83

83:                                               ; preds = %86, %82
  %84 = phi i64 [ %87, %86 ], [ 0, %82 ]
  %85 = icmp slt i64 %84, 12
  br i1 %85, label %86, label %88

86:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_2, ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %52)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %76)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_3, ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_0, i32 14, i32 1, i32 480, i32 3, i32 3, i32 1, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %87 = add i64 %84, 4
  br label %83

88:                                               ; preds = %83
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %72)
  call void @llvm.assume(i1 %54)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %56)
  call void @llvm.assume(i1 %58)
  call void @bn10_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @B_OF_b10_act_layer1_layer2_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @B_OF_b10_act_layer1_layer2_cons_buff_1, ptr @weightsInBN10_layer2_cons_buff_0, ptr @B_OF_b10_act_layer2_layer3_buff_1, i32 14, i32 1, i32 480, i32 3, i32 3, i32 2, i32 7, i32 0)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_2_4() {
  br label %1

1:                                                ; preds = %20, %0
  %2 = phi i64 [ %21, %20 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %22

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %19, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 14
  br i1 %7, label %8, label %20

8:                                                ; preds = %5
  call void @llvm.aie2.acquire(i32 3, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  %9 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @act_bn9_bn10_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  %13 = and i64 ptrtoint (ptr @weightsInBN10_layer1_cons_buff_0 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  call void @bn10_conv2dk1_relu_i8_ui8(ptr @act_bn9_bn10_buff_0, ptr @weightsInBN10_layer1_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_buff_0, i32 14, i32 80, i32 480, i32 8)
  call void @llvm.aie2.release(i32 2, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 3, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  %15 = and i64 ptrtoint (ptr @B_OF_b10_act_layer1_layer2_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = and i64 ptrtoint (ptr @act_bn9_bn10_buff_1 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %14)
  call void @bn10_conv2dk1_relu_i8_ui8(ptr @act_bn9_bn10_buff_1, ptr @weightsInBN10_layer1_cons_buff_0, ptr @B_OF_b10_act_layer1_layer2_buff_1, i32 14, i32 80, i32 480, i32 8)
  call void @llvm.aie2.release(i32 2, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %19 = add i64 %6, 2
  br label %5

20:                                               ; preds = %5
  call void @llvm.aie2.release(i32 48, i32 1)
  %21 = add i64 %2, 1
  br label %1

22:                                               ; preds = %1
  ret void
}

define void @core_2_3() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn9_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp23 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp23, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp23, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp23, i32 2), align 4
  call void @llvm.assume(i1 %4)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp23, i32 3), align 4
  call void @llvm.aie2.acquire(i32 5, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %9 = and i64 ptrtoint (ptr @bn9_act_1_2_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @act_bn8_bn9_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_0, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_0, i32 14, i32 80, i32 184, i32 %5)
  %13 = and i64 ptrtoint (ptr @bn9_act_1_2_buff_1 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @act_bn8_bn9_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_1, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_1, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %17 = and i64 ptrtoint (ptr @bn9_act_2_3_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %19 = and i64 ptrtoint (ptr @act_bn9_bn10_buff_0 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_0, ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  br label %21

21:                                               ; preds = %24, %0
  %22 = phi i64 [ %29, %24 ], [ 0, %0 ]
  %23 = icmp slt i64 %22, 12
  br i1 %23, label %24, label %30

24:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @bn9_act_1_2_buff_2 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_0, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_2, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_1, ptr @bn9_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %27 = and i64 ptrtoint (ptr @act_bn9_bn10_buff_1 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_1, ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_1, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_0, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_1, ptr @bn9_act_1_2_buff_2, ptr @bn9_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_0, ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_0, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_1, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_2, ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_1, ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_1, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_2, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_1, ptr @bn9_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_0, ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_0, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_0, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_1, ptr @bn9_act_1_2_buff_2, ptr @bn9_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_1, ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_relu_i8_ui8(ptr @act_bn8_bn9_buff_1, ptr @bn9_wts_OF_L2L1_cons_buff_0, ptr @bn9_act_1_2_buff_1, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_2, ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_0, ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %29 = add i64 %22, 6
  br label %21

30:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %14)
  call void @bn9_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn9_act_1_2_buff_0, ptr @bn9_act_1_2_buff_1, ptr @bn9_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn9_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 2, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %31 = and i64 ptrtoint (ptr @act_bn9_bn10_buff_1 to i64), 31
  %32 = icmp eq i64 %31, 0
  call void @llvm.assume(i1 %32)
  call void @llvm.assume(i1 %16)
  call void @bn9_conv2dk1_skip_ui8_i8_i8(ptr @bn9_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn9_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn9_bn10_buff_1, ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_2_2() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn8_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp22 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp22, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp22, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp22, i32 2), align 4
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  %8 = and i64 ptrtoint (ptr @bn8_act_1_2_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  %10 = and i64 ptrtoint (ptr @act_bn7_bn8_cons_buff_0 to i64), 31
  %11 = icmp eq i64 %10, 0
  call void @llvm.assume(i1 %11)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_0, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_0, i32 14, i32 80, i32 184, i32 %5)
  %12 = and i64 ptrtoint (ptr @bn8_act_1_2_buff_1 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = and i64 ptrtoint (ptr @act_bn7_bn8_cons_buff_1 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_1, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_1, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  %16 = and i64 ptrtoint (ptr @bn8_act_2_3_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  %18 = and i64 ptrtoint (ptr @act_bn8_bn9_buff_0 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %20

20:                                               ; preds = %23, %0
  %21 = phi i64 [ %30, %23 ], [ 0, %0 ]
  %22 = icmp slt i64 %21, 12
  br i1 %22, label %23, label %31

23:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %24 = and i64 ptrtoint (ptr @bn8_act_1_2_buff_2 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  %26 = and i64 ptrtoint (ptr @act_bn7_bn8_cons_buff_2 to i64), 31
  %27 = icmp eq i64 %26, 0
  call void @llvm.assume(i1 %27)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_2, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_2, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_1, ptr @bn8_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  %28 = and i64 ptrtoint (ptr @act_bn8_bn9_buff_1 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_0, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_0, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_1, ptr @bn8_act_1_2_buff_2, ptr @bn8_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_1, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_1, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_2, ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %29)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %25)
  call void @llvm.assume(i1 %27)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_2, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_2, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_1, ptr @bn8_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_0, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_0, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_1, ptr @bn8_act_1_2_buff_2, ptr @bn8_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %29)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn8_conv2dk1_relu_i8_ui8(ptr @act_bn7_bn8_cons_buff_1, ptr @bn8_wts_OF_L2L1_cons_buff_0, ptr @bn8_act_1_2_buff_1, i32 14, i32 80, i32 184, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_2, ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_0, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %30 = add i64 %21, 6
  br label %20

31:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %13)
  call void @bn8_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn8_act_1_2_buff_0, ptr @bn8_act_1_2_buff_1, ptr @bn8_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 14720), ptr @bn8_act_2_3_buff_0, i32 14, i32 1, i32 184, i32 3, i32 3, i32 2, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  %32 = and i64 ptrtoint (ptr @act_bn8_bn9_buff_1 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  call void @bn8_conv2dk1_ui8_i8(ptr @bn8_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn8_wts_OF_L2L1_cons_buff_0, i32 16376), ptr @act_bn8_bn9_buff_1, i32 14, i32 184, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_1_3() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn7_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp13 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp13, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp13, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp13, i32 2), align 4
  call void @llvm.assume(i1 %4)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp13, i32 3), align 4
  call void @llvm.aie2.acquire(i32 5, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %9 = and i64 ptrtoint (ptr @bn7_act_1_2_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @act_bn6_bn7_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_0, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_0, i32 14, i32 80, i32 200, i32 %5)
  %13 = and i64 ptrtoint (ptr @bn7_act_1_2_buff_1 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @act_bn6_bn7_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_1, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_1, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %17 = and i64 ptrtoint (ptr @bn7_act_2_3_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %19 = and i64 ptrtoint (ptr @act_bn7_bn8_buff_0 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_0, ptr @act_bn6_bn7_buff_0, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  br label %21

21:                                               ; preds = %24, %0
  %22 = phi i64 [ %29, %24 ], [ 0, %0 ]
  %23 = icmp slt i64 %22, 12
  br i1 %23, label %24, label %30

24:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @bn7_act_1_2_buff_2 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_0, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_2, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_1, ptr @bn7_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %27 = and i64 ptrtoint (ptr @act_bn7_bn8_buff_1 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_1, ptr @act_bn6_bn7_buff_1, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_1, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_0, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_1, ptr @bn7_act_1_2_buff_2, ptr @bn7_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_0, ptr @act_bn6_bn7_buff_0, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_0, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_1, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_2, ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_1, ptr @act_bn6_bn7_buff_1, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_1, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_2, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_1, ptr @bn7_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_0, ptr @act_bn6_bn7_buff_0, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_0, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_0, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_1, ptr @bn7_act_1_2_buff_2, ptr @bn7_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_1, ptr @act_bn6_bn7_buff_1, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_relu_i8_ui8(ptr @act_bn6_bn7_buff_1, ptr @bn7_wts_OF_L2L1_cons_buff_0, ptr @bn7_act_1_2_buff_1, i32 14, i32 80, i32 200, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_2, ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_0, ptr @act_bn6_bn7_buff_0, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %29 = add i64 %22, 6
  br label %21

30:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %14)
  call void @bn7_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn7_act_1_2_buff_0, ptr @bn7_act_1_2_buff_1, ptr @bn7_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 16000), ptr @bn7_act_2_3_buff_0, i32 14, i32 1, i32 200, i32 3, i32 3, i32 2, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %31 = and i64 ptrtoint (ptr @act_bn7_bn8_buff_1 to i64), 31
  %32 = icmp eq i64 %31, 0
  call void @llvm.assume(i1 %32)
  call void @llvm.assume(i1 %16)
  call void @bn7_conv2dk1_skip_ui8_i8_i8(ptr @bn7_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn7_wts_OF_L2L1_cons_buff_0, i32 17800), ptr @act_bn7_bn8_buff_1, ptr @act_bn6_bn7_buff_1, i32 14, i32 200, i32 80, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_1_2() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn6_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp12 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp12, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp12, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp12, i32 2), align 4
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  %8 = and i64 ptrtoint (ptr @bn6_act_1_2_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  %10 = and i64 ptrtoint (ptr @act_bn5_bn6_cons_buff_0 to i64), 31
  %11 = icmp eq i64 %10, 0
  call void @llvm.assume(i1 %11)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_0, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_0, i32 28, i32 40, i32 240, i32 %5)
  %12 = and i64 ptrtoint (ptr @bn6_act_1_2_buff_1 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = and i64 ptrtoint (ptr @act_bn5_bn6_cons_buff_1 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_1, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_1, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  %16 = and i64 ptrtoint (ptr @bn6_act_2_3_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_0, ptr @bn6_act_1_2_buff_0, ptr @bn6_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  %18 = and i64 ptrtoint (ptr @act_bn6_bn7_buff_0 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_0, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %20

20:                                               ; preds = %23, %0
  %21 = phi i64 [ %30, %23 ], [ 0, %0 ]
  %22 = icmp slt i64 %21, 12
  br i1 %22, label %23, label %31

23:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  %24 = and i64 ptrtoint (ptr @bn6_act_1_2_buff_2 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  %26 = and i64 ptrtoint (ptr @act_bn5_bn6_cons_buff_2 to i64), 31
  %27 = icmp eq i64 %26, 0
  call void @llvm.assume(i1 %27)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_2, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_2, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_0, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_0, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_1, ptr @bn6_act_1_2_buff_2, ptr @bn6_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  %28 = and i64 ptrtoint (ptr @act_bn6_bn7_buff_1 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_1, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_1, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_1, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %25)
  call void @llvm.assume(i1 %27)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_2, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_2, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_0, ptr @bn6_act_1_2_buff_1, ptr @bn6_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_0, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_0, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_0, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_1, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_1, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_2, ptr @bn6_act_1_2_buff_0, ptr @bn6_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %29)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_1, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  call void @llvm.assume(i1 %25)
  call void @llvm.assume(i1 %27)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_2, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_2, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_0, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_0, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_1, ptr @bn6_act_1_2_buff_2, ptr @bn6_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_0, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_1, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_1, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %25)
  call void @llvm.assume(i1 %27)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_2, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_2, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_0, ptr @bn6_act_1_2_buff_1, ptr @bn6_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %29)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_1, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_0, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_0, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_1, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_1, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_2, ptr @bn6_act_1_2_buff_0, ptr @bn6_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_0, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %30 = add i64 %21, 6
  br label %20

31:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 51, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -2)
  %32 = and i64 ptrtoint (ptr @bn6_act_1_2_buff_2 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  %34 = and i64 ptrtoint (ptr @act_bn5_bn6_cons_buff_2 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_2, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_2, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn6_conv2dk1_relu_i8_ui8(ptr @act_bn5_bn6_cons_buff_0, ptr @bn6_wts_OF_L2L1_cons_buff_0, ptr @bn6_act_1_2_buff_0, i32 28, i32 40, i32 240, i32 %5)
  call void @llvm.aie2.release(i32 55, i32 2)
  call void @llvm.aie2.release(i32 50, i32 2)
  call void @llvm.aie2.acquire(i32 55, i32 -2)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %33)
  call void @bn6_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn6_act_1_2_buff_1, ptr @bn6_act_1_2_buff_2, ptr @bn6_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 9600), ptr @bn6_act_2_3_buff_0, i32 28, i32 1, i32 240, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 54, i32 2)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %17)
  %36 = and i64 ptrtoint (ptr @act_bn6_bn7_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @bn6_conv2dk1_ui8_i8(ptr @bn6_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn6_wts_OF_L2L1_cons_buff_0, i32 11760), ptr @act_bn6_bn7_buff_1, i32 14, i32 240, i32 80, i32 %7)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_1_4() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn5_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp14 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp14, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp14, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp14, i32 2), align 4
  call void @llvm.aie2.acquire(i32 35, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %8 = and i64 ptrtoint (ptr @bn5_act_1_2_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  %10 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_0 to i64), 31
  %11 = icmp eq i64 %10, 0
  call void @llvm.assume(i1 %11)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_0, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  %12 = and i64 ptrtoint (ptr @bn5_act_1_2_buff_1 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_1 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_1, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_1, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.release(i32 34, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %16 = and i64 ptrtoint (ptr @bn5_act_2_3_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  %18 = and i64 ptrtoint (ptr @act_bn5_bn6_buff_0 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_0, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  br label %20

20:                                               ; preds = %23, %0
  %21 = phi i64 [ %30, %23 ], [ 0, %0 ]
  %22 = icmp slt i64 %21, 24
  br i1 %22, label %23, label %31

23:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %24 = and i64 ptrtoint (ptr @bn5_act_1_2_buff_2 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  %26 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_2 to i64), 31
  %27 = icmp eq i64 %26, 0
  call void @llvm.assume(i1 %27)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_2, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_2, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_1, ptr @bn5_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  %28 = and i64 ptrtoint (ptr @act_bn5_bn6_buff_1 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_1, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_0, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_1, ptr @bn5_act_1_2_buff_2, ptr @bn5_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_0, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_1, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_1, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_2, ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %29)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_1, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %25)
  call void @llvm.assume(i1 %27)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_2, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_2, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_1, ptr @bn5_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_0, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_0, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_1, ptr @bn5_act_1_2_buff_2, ptr @bn5_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %29)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_1, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_1, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_1, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_2, ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_0, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %30 = add i64 %21, 6
  br label %20

31:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %32 = and i64 ptrtoint (ptr @bn5_act_1_2_buff_2 to i64), 31
  %33 = icmp eq i64 %32, 0
  call void @llvm.assume(i1 %33)
  %34 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_2 to i64), 31
  %35 = icmp eq i64 %34, 0
  call void @llvm.assume(i1 %35)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_2, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_2, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %33)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_1, ptr @bn5_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  %36 = and i64 ptrtoint (ptr @act_bn5_bn6_buff_1 to i64), 31
  %37 = icmp eq i64 %36, 0
  call void @llvm.assume(i1 %37)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_1, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 35, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn5_conv2dk1_relu_i8_ui8(ptr @act_bn4_bn5_buff_0, ptr @bn5_wts_OF_L2L1_cons_buff_0, ptr @bn5_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 34, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %33)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_1, ptr @bn5_act_1_2_buff_2, ptr @bn5_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_0, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %33)
  call void @bn5_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn5_act_1_2_buff_2, ptr @bn5_act_1_2_buff_0, ptr @bn5_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn5_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 2, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %37)
  call void @bn5_conv2dk1_ui8_i8(ptr @bn5_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn5_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn5_bn6_buff_1, i32 28, i32 120, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_1_5() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn4_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp15 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp15, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp15, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp15, i32 2), align 4
  call void @llvm.assume(i1 %4)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp15, i32 3), align 4
  call void @llvm.aie2.acquire(i32 19, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %9 = and i64 ptrtoint (ptr @bn4_act_1_2_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_0, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  %13 = and i64 ptrtoint (ptr @bn4_act_1_2_buff_1 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_1, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_1, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %17 = and i64 ptrtoint (ptr @bn4_act_2_3_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_0, ptr @bn4_act_1_2_buff_0, ptr @bn4_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %19 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_0 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_0, ptr @act_bn3_bn4_buff_0, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  br label %21

21:                                               ; preds = %24, %0
  %22 = phi i64 [ %33, %24 ], [ 0, %0 ]
  %23 = icmp slt i64 %22, 24
  br i1 %23, label %24, label %34

24:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 19, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @bn4_act_1_2_buff_2 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  %27 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_2 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_2, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_2, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_0, ptr @bn4_act_1_2_buff_1, ptr @bn4_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %29 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_1 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %16)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_1, ptr @act_bn3_bn4_buff_1, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 19, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_0, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_1, ptr @bn4_act_1_2_buff_2, ptr @bn4_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %31 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_2 to i64), 31
  %32 = icmp eq i64 %31, 0
  call void @llvm.assume(i1 %32)
  call void @llvm.assume(i1 %28)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_2, ptr @act_bn3_bn4_buff_2, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 19, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_1, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_1, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_2, ptr @bn4_act_1_2_buff_0, ptr @bn4_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_0, ptr @act_bn3_bn4_buff_0, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %33 = add i64 %22, 3
  br label %21

34:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 19, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %35 = and i64 ptrtoint (ptr @bn4_act_1_2_buff_2 to i64), 31
  %36 = icmp eq i64 %35, 0
  call void @llvm.assume(i1 %36)
  %37 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_2 to i64), 31
  %38 = icmp eq i64 %37, 0
  call void @llvm.assume(i1 %38)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_2, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_2, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %36)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_0, ptr @bn4_act_1_2_buff_1, ptr @bn4_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %39 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_1 to i64), 31
  %40 = icmp eq i64 %39, 0
  call void @llvm.assume(i1 %40)
  call void @llvm.assume(i1 %16)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_1, ptr @act_bn3_bn4_buff_1, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 19, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn4_conv2dk1_relu_i8_ui8(ptr @act_bn3_bn4_buff_0, ptr @bn4_wts_OF_L2L1_cons_buff_0, ptr @bn4_act_1_2_buff_0, i32 28, i32 40, i32 120, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %36)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_1, ptr @bn4_act_1_2_buff_2, ptr @bn4_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %41 = and i64 ptrtoint (ptr @act_bn4_bn5_buff_2 to i64), 31
  %42 = icmp eq i64 %41, 0
  call void @llvm.assume(i1 %42)
  call void @llvm.assume(i1 %38)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_2, ptr @act_bn3_bn4_buff_2, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %36)
  call void @bn4_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn4_act_1_2_buff_2, ptr @bn4_act_1_2_buff_0, ptr @bn4_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 4800), ptr @bn4_act_2_3_buff_0, i32 28, i32 1, i32 120, i32 3, i32 3, i32 2, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn4_conv2dk1_skip_ui8_i8_i8(ptr @bn4_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn4_wts_OF_L2L1_cons_buff_0, i32 5880), ptr @act_bn4_bn5_buff_0, ptr @act_bn3_bn4_buff_0, i32 28, i32 120, i32 40, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 18, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_0_5() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn3_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp05 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp05, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp05, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp05, i32 2), align 4
  call void @llvm.aie2.acquire(i32 3, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %8 = and i64 ptrtoint (ptr @bn3_act_1_2_buff_0 to i64), 31
  %9 = icmp eq i64 %8, 0
  call void @llvm.assume(i1 %9)
  %10 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_0 to i64), 31
  %11 = icmp eq i64 %10, 0
  call void @llvm.assume(i1 %11)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_0, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_0, i32 56, i32 24, i32 72, i32 %5)
  %12 = and i64 ptrtoint (ptr @bn3_act_1_2_buff_1 to i64), 31
  %13 = icmp eq i64 %12, 0
  call void @llvm.assume(i1 %13)
  %14 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_1 to i64), 31
  %15 = icmp eq i64 %14, 0
  call void @llvm.assume(i1 %15)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_1, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_1, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.release(i32 2, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %16 = and i64 ptrtoint (ptr @bn3_act_2_3_buff_0 to i64), 31
  %17 = icmp eq i64 %16, 0
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn3_act_1_2_buff_0, ptr @bn3_act_1_2_buff_0, ptr @bn3_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn3_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  %18 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_0 to i64), 31
  %19 = icmp eq i64 %18, 0
  call void @llvm.assume(i1 %19)
  call void @bn3_conv2dk1_ui8_i8(ptr @bn3_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn3_bn4_buff_0, i32 28, i32 72, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  br label %20

20:                                               ; preds = %23, %0
  %21 = phi i64 [ %32, %23 ], [ 0, %0 ]
  %22 = icmp slt i64 %21, 27
  br i1 %22, label %23, label %33

23:                                               ; preds = %20
  call void @llvm.aie2.acquire(i32 3, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %24 = and i64 ptrtoint (ptr @bn3_act_1_2_buff_2 to i64), 31
  %25 = icmp eq i64 %24, 0
  call void @llvm.assume(i1 %25)
  %26 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_2 to i64), 31
  %27 = icmp eq i64 %26, 0
  call void @llvm.assume(i1 %27)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_2, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_2, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_0, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_0, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.release(i32 2, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn3_act_1_2_buff_1, ptr @bn3_act_1_2_buff_2, ptr @bn3_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn3_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  %28 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_1 to i64), 31
  %29 = icmp eq i64 %28, 0
  call void @llvm.assume(i1 %29)
  call void @bn3_conv2dk1_ui8_i8(ptr @bn3_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn3_bn4_buff_1, i32 28, i32 72, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 3, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_1, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_1, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.assume(i1 %25)
  call void @llvm.assume(i1 %27)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_2, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_2, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.release(i32 2, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn3_act_1_2_buff_0, ptr @bn3_act_1_2_buff_1, ptr @bn3_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn3_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  %30 = and i64 ptrtoint (ptr @act_bn3_bn4_buff_2 to i64), 31
  %31 = icmp eq i64 %30, 0
  call void @llvm.assume(i1 %31)
  call void @bn3_conv2dk1_ui8_i8(ptr @bn3_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn3_bn4_buff_2, i32 28, i32 72, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 3, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %11)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_0, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_0, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %15)
  call void @bn3_conv2dk1_relu_i8_ui8(ptr @act_bn2_bn3_buff_1, ptr @bn3_wts_OF_L2L1_cons_buff_0, ptr @bn3_act_1_2_buff_1, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.release(i32 2, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %9)
  call void @llvm.assume(i1 %13)
  call void @llvm.assume(i1 %25)
  call void @bn3_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn3_act_1_2_buff_2, ptr @bn3_act_1_2_buff_0, ptr @bn3_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn3_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %17)
  call void @llvm.assume(i1 %19)
  call void @bn3_conv2dk1_ui8_i8(ptr @bn3_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn3_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn3_bn4_buff_0, i32 28, i32 72, i32 40, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %32 = add i64 %21, 3
  br label %20

33:                                               ; preds = %20
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_0_4() {
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn2_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp04 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp04, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp04, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp04, i32 2), align 4
  call void @llvm.assume(i1 %4)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp04, i32 3), align 4
  call void @llvm.aie2.acquire(i32 5, i32 -2)
  call void @llvm.aie2.acquire(i32 52, i32 -2)
  %9 = and i64 ptrtoint (ptr @bn2_act_1_2_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @bn2_conv2dk1_relu_i8_ui8(ptr @act_bn01_bn2_buff_0, ptr @bn2_wts_OF_L2L1_cons_buff_0, ptr @bn2_act_1_2_buff_0, i32 56, i32 24, i32 72, i32 %5)
  %13 = and i64 ptrtoint (ptr @bn2_act_1_2_buff_1 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  call void @bn2_conv2dk1_relu_i8_ui8(ptr @act_bn01_bn2_buff_1, ptr @bn2_wts_OF_L2L1_cons_buff_0, ptr @bn2_act_1_2_buff_1, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 2)
  call void @llvm.aie2.acquire(i32 53, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %17 = and i64 ptrtoint (ptr @bn2_act_2_3_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn2_act_1_2_buff_0, ptr @bn2_act_1_2_buff_0, ptr @bn2_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn2_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 0, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %19 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_0 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn2_conv2dk1_skip_ui8_i8_i8(ptr @bn2_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn2_bn3_buff_0, ptr @act_bn01_bn2_buff_0, i32 56, i32 72, i32 24, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  br label %21

21:                                               ; preds = %24, %0
  %22 = phi i64 [ %33, %24 ], [ 0, %0 ]
  %23 = icmp slt i64 %22, 54
  br i1 %23, label %24, label %34

24:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @bn2_act_1_2_buff_2 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  %27 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_2 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @bn2_conv2dk1_relu_i8_ui8(ptr @act_bn01_bn2_buff_2, ptr @bn2_wts_OF_L2L1_cons_buff_0, ptr @bn2_act_1_2_buff_2, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn2_act_1_2_buff_0, ptr @bn2_act_1_2_buff_1, ptr @bn2_act_1_2_buff_2, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn2_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %29 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_1 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %16)
  call void @bn2_conv2dk1_skip_ui8_i8_i8(ptr @bn2_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn2_bn3_buff_1, ptr @act_bn01_bn2_buff_1, i32 56, i32 72, i32 24, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %12)
  call void @bn2_conv2dk1_relu_i8_ui8(ptr @act_bn01_bn2_buff_0, ptr @bn2_wts_OF_L2L1_cons_buff_0, ptr @bn2_act_1_2_buff_0, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn2_act_1_2_buff_1, ptr @bn2_act_1_2_buff_2, ptr @bn2_act_1_2_buff_0, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn2_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %31 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_2 to i64), 31
  %32 = icmp eq i64 %31, 0
  call void @llvm.assume(i1 %32)
  call void @llvm.assume(i1 %28)
  call void @bn2_conv2dk1_skip_ui8_i8_i8(ptr @bn2_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn2_bn3_buff_2, ptr @act_bn01_bn2_buff_2, i32 56, i32 72, i32 24, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.acquire(i32 5, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @bn2_conv2dk1_relu_i8_ui8(ptr @act_bn01_bn2_buff_1, ptr @bn2_wts_OF_L2L1_cons_buff_0, ptr @bn2_act_1_2_buff_1, i32 56, i32 24, i32 72, i32 %5)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 53, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %26)
  call void @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn2_act_1_2_buff_2, ptr @bn2_act_1_2_buff_0, ptr @bn2_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn2_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 1, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 1)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %12)
  call void @bn2_conv2dk1_skip_ui8_i8_i8(ptr @bn2_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn2_bn3_buff_0, ptr @act_bn01_bn2_buff_0, i32 56, i32 72, i32 24, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %33 = add i64 %22, 3
  br label %21

34:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %10)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %14)
  call void @bn2_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @bn2_act_1_2_buff_0, ptr @bn2_act_1_2_buff_1, ptr @bn2_act_1_2_buff_1, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 1728), ptr @bn2_act_2_3_buff_0, i32 56, i32 1, i32 72, i32 3, i32 3, i32 2, i32 %6, i32 0)
  call void @llvm.aie2.release(i32 52, i32 2)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 %18)
  %35 = and i64 ptrtoint (ptr @act_bn2_bn3_buff_1 to i64), 31
  %36 = icmp eq i64 %35, 0
  call void @llvm.assume(i1 %36)
  call void @llvm.assume(i1 %16)
  call void @bn2_conv2dk1_skip_ui8_i8_i8(ptr @bn2_act_2_3_buff_0, ptr getelementptr (i8, ptr @bn2_wts_OF_L2L1_cons_buff_0, i32 2376), ptr @act_bn2_bn3_buff_1, ptr @act_bn01_bn2_buff_1, i32 56, i32 72, i32 24, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 4, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  call void @llvm.aie2.release(i32 48, i32 1)
  ret void
}

define void @core_0_3() {
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  %1 = and i64 ptrtoint (ptr @bn0_1_wts_OF_L2L1_cons_buff_0 to i64), 31
  %2 = icmp eq i64 %1, 0
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  call void @llvm.assume(i1 %2)
  %3 = and i64 ptrtoint (ptr @rtp03 to i64), 31
  %4 = icmp eq i64 %3, 0
  call void @llvm.assume(i1 %4)
  %5 = load i32, ptr @rtp03, align 4
  call void @llvm.assume(i1 %4)
  %6 = load i32, ptr getelementptr (i32, ptr @rtp03, i32 1), align 4
  call void @llvm.assume(i1 %4)
  %7 = load i32, ptr getelementptr (i32, ptr @rtp03, i32 2), align 4
  call void @llvm.assume(i1 %4)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp03, i32 3), align 4
  call void @llvm.assume(i1 %4)
  %9 = load i32, ptr getelementptr (i32, ptr @rtp03, i32 4), align 4
  call void @llvm.assume(i1 %4)
  %10 = load i32, ptr getelementptr (i32, ptr @rtp03, i32 5), align 4
  call void @llvm.aie2.acquire(i32 49, i32 -2)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  %11 = and i64 ptrtoint (ptr @bn01_act_bn0_2_3_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  %13 = and i64 ptrtoint (ptr @act_in_cons_buff_0 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @act_in_cons_buff_1 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_1, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 0, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  %17 = and i64 ptrtoint (ptr @bn01_act_bn0_bn1_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_0, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  %19 = and i64 ptrtoint (ptr @bn01_act_bn1_1_2_buff_0 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_0, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  %21 = and i64 ptrtoint (ptr @act_in_cons_buff_2 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_1, ptr @act_in_cons_buff_2, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %16)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_1, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  %23 = and i64 ptrtoint (ptr @bn01_act_bn1_1_2_buff_1 to i64), 31
  %24 = icmp eq i64 %23, 0
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_1, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 59, i32 -2)
  call void @llvm.aie2.acquire(i32 60, i32 -1)
  %25 = and i64 ptrtoint (ptr @bn01_act_bn1_2_3_buff_0 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %24)
  call void @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn01_act_bn1_1_2_buff_0, ptr @bn01_act_bn1_1_2_buff_0, ptr @bn01_act_bn1_1_2_buff_1, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 1424), ptr @bn01_act_bn1_2_3_buff_0, i32 112, i32 1, i32 64, i32 3, i32 3, i32 0, i32 %9, i32 0)
  call void @llvm.aie2.release(i32 58, i32 1)
  call void @llvm.aie2.release(i32 61, i32 1)
  call void @llvm.aie2.acquire(i32 61, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  %27 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_0 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  call void @bn1_conv2dk1_ui8_i8(ptr @bn01_act_bn1_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 2000), ptr @act_bn01_bn2_buff_0, i32 56, i32 64, i32 24, i32 %10)
  call void @llvm.aie2.release(i32 60, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  br label %29

29:                                               ; preds = %32, %0
  %30 = phi i64 [ %39, %32 ], [ 0, %0 ]
  %31 = icmp slt i64 %30, 54
  br i1 %31, label %32, label %40

32:                                               ; preds = %29
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_1, ptr @act_in_cons_buff_2, ptr @act_in_cons_buff_0, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_2, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  %33 = and i64 ptrtoint (ptr @bn01_act_bn1_1_2_buff_2 to i64), 31
  %34 = icmp eq i64 %33, 0
  call void @llvm.assume(i1 %34)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_2, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_2, ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_1, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_0, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_0, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 59, i32 -2)
  call void @llvm.aie2.acquire(i32 60, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %34)
  call void @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn01_act_bn1_1_2_buff_1, ptr @bn01_act_bn1_1_2_buff_2, ptr @bn01_act_bn1_1_2_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 1424), ptr @bn01_act_bn1_2_3_buff_0, i32 112, i32 1, i32 64, i32 3, i32 3, i32 1, i32 %9, i32 0)
  call void @llvm.aie2.release(i32 58, i32 2)
  call void @llvm.aie2.release(i32 61, i32 1)
  call void @llvm.aie2.acquire(i32 61, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  %35 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_1 to i64), 31
  %36 = icmp eq i64 %35, 0
  call void @llvm.assume(i1 %36)
  call void @bn1_conv2dk1_ui8_i8(ptr @bn01_act_bn1_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 2000), ptr @act_bn01_bn2_buff_1, i32 56, i32 64, i32 24, i32 %10)
  call void @llvm.aie2.release(i32 60, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_1, ptr @act_in_cons_buff_2, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %16)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_1, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_1, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_1, ptr @act_in_cons_buff_2, ptr @act_in_cons_buff_0, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_2, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  call void @llvm.assume(i1 %34)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_2, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 59, i32 -2)
  call void @llvm.aie2.acquire(i32 60, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %34)
  call void @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn01_act_bn1_1_2_buff_0, ptr @bn01_act_bn1_1_2_buff_1, ptr @bn01_act_bn1_1_2_buff_2, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 1424), ptr @bn01_act_bn1_2_3_buff_0, i32 112, i32 1, i32 64, i32 3, i32 3, i32 1, i32 %9, i32 0)
  call void @llvm.aie2.release(i32 58, i32 2)
  call void @llvm.aie2.release(i32 61, i32 1)
  call void @llvm.aie2.acquire(i32 61, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  %37 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_2 to i64), 31
  %38 = icmp eq i64 %37, 0
  call void @llvm.assume(i1 %38)
  call void @bn1_conv2dk1_ui8_i8(ptr @bn01_act_bn1_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 2000), ptr @act_bn01_bn2_buff_2, i32 56, i32 64, i32 24, i32 %10)
  call void @llvm.aie2.release(i32 60, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_2, ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_1, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_0, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_0, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_1, ptr @act_in_cons_buff_2, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %16)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_1, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_1, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 59, i32 -2)
  call void @llvm.aie2.acquire(i32 60, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %34)
  call void @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn01_act_bn1_1_2_buff_2, ptr @bn01_act_bn1_1_2_buff_0, ptr @bn01_act_bn1_1_2_buff_1, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 1424), ptr @bn01_act_bn1_2_3_buff_0, i32 112, i32 1, i32 64, i32 3, i32 3, i32 1, i32 %9, i32 0)
  call void @llvm.aie2.release(i32 58, i32 2)
  call void @llvm.aie2.release(i32 61, i32 1)
  call void @llvm.aie2.acquire(i32 61, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @bn1_conv2dk1_ui8_i8(ptr @bn01_act_bn1_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 2000), ptr @act_bn01_bn2_buff_0, i32 56, i32 64, i32 24, i32 %10)
  call void @llvm.aie2.release(i32 60, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %39 = add i64 %30, 3
  br label %29

40:                                               ; preds = %29
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %16)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_1, ptr @act_in_cons_buff_2, ptr @act_in_cons_buff_0, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 1, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_2, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  %41 = and i64 ptrtoint (ptr @bn01_act_bn1_1_2_buff_2 to i64), 31
  %42 = icmp eq i64 %41, 0
  call void @llvm.assume(i1 %42)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_2, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 54, i32 -1)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %14)
  call void @llvm.assume(i1 %22)
  call void @bn0_conv2dk3_dw_stride1_relu_ui8_ui8(ptr @act_in_cons_buff_2, ptr @act_in_cons_buff_0, ptr @act_in_cons_buff_0, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, ptr @bn01_act_bn0_2_3_buff_0, i32 112, i32 1, i32 16, i32 3, i32 3, i32 2, i32 %5, i32 0)
  call void @llvm.aie2.release(i32 55, i32 1)
  call void @llvm.aie2.acquire(i32 55, i32 -1)
  call void @llvm.aie2.acquire(i32 56, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %12)
  call void @llvm.assume(i1 %14)
  call void @bn0_conv2dk1_skip_ui8_ui8_i8(ptr @bn01_act_bn0_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 144), ptr @bn01_act_bn0_bn1_buff_0, ptr @act_in_cons_buff_0, i32 112, i32 16, i32 16, i32 %6, i32 %7)
  call void @llvm.aie2.release(i32 48, i32 2)
  call void @llvm.aie2.release(i32 54, i32 1)
  call void @llvm.aie2.release(i32 57, i32 1)
  call void @llvm.aie2.acquire(i32 57, i32 -1)
  call void @llvm.aie2.acquire(i32 58, i32 -1)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %18)
  call void @bn1_conv2dk1_relu_i8_ui8(ptr @bn01_act_bn0_bn1_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 400), ptr @bn01_act_bn1_1_2_buff_0, i32 112, i32 16, i32 64, i32 %8)
  call void @llvm.aie2.release(i32 56, i32 1)
  call void @llvm.aie2.release(i32 59, i32 1)
  call void @llvm.aie2.acquire(i32 59, i32 -2)
  call void @llvm.aie2.acquire(i32 60, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %24)
  call void @llvm.assume(i1 %42)
  call void @bn1_conv2dk3_dw_stride2_relu_ui8_ui8(ptr @bn01_act_bn1_1_2_buff_1, ptr @bn01_act_bn1_1_2_buff_2, ptr @bn01_act_bn1_1_2_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 1424), ptr @bn01_act_bn1_2_3_buff_0, i32 112, i32 1, i32 64, i32 3, i32 3, i32 1, i32 %9, i32 0)
  call void @llvm.aie2.release(i32 58, i32 3)
  call void @llvm.aie2.release(i32 61, i32 1)
  call void @llvm.aie2.acquire(i32 61, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  %43 = and i64 ptrtoint (ptr @act_bn01_bn2_buff_1 to i64), 31
  %44 = icmp eq i64 %43, 0
  call void @llvm.assume(i1 %44)
  call void @bn1_conv2dk1_ui8_i8(ptr @bn01_act_bn1_2_3_buff_0, ptr getelementptr (i8, ptr @bn0_1_wts_OF_L2L1_cons_buff_0, i32 2000), ptr @act_bn01_bn2_buff_1, i32 56, i32 64, i32 24, i32 %10)
  call void @llvm.aie2.release(i32 60, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
